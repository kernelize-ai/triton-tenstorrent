#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CORESPECIALIZE
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

static bool isDepenendetLoad(triton::LoadOp load) {
  for (auto &use : load->getUses()) {
    if (isa<triton::LoadOp>(use.getOwner()))
      return true;
  }
  return false;
}

class Specializer {
  triton::FuncOp func;
  SmallVector<triton::gpu::LocalAllocOp> allocs;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  DenseMap<Operation *, triton::gpu::LocalAllocOp> allocMap;

public:
  Specializer(ModuleOp m, triton::FuncOp _func) : func(_func) {
    // collect all tile sizes and create multi-buffers
    func.walk([&](Operation *op) {
      if (auto store = dyn_cast<triton::StoreOp>(op)) {
        auto alloc = createSharedBuffer(store, store.getValue().getType());
        allocMap[store] = alloc;
        stores.push_back(store);
      } else if (auto load = dyn_cast<triton::LoadOp>(op)) {
        if (!isDepenendetLoad(load)) {
          auto alloc = createSharedBuffer(load, load.getType());
          allocMap[load] = alloc;
          loads.push_back(load);
        }
      }
    });

    // Create Reader Program
    auto readerFunc = makeReader(func);
    m.insert(func, readerFunc);

    // Create Compute Program
    auto computeFunc = makeCompute(func);
    m.insert(func, computeFunc);

    // Create Writer Program
    auto writerFunc = makeWriter(func);
    m.insert(func, writerFunc);
  }
  ~Specializer() {
    allocs.clear();
    loads.clear();
    stores.clear();
    allocMap.clear();
    func.erase();
  }

  triton::gpu::LocalAllocOp createSharedBuffer(Operation *op, Type type) {
    auto rtType = cast<RankedTensorType>(type);
    int idx = allocs.size();
    auto ctx = func.getContext();
    OpBuilder b(func.getBody());
    auto shape = rtType.getShape();
    auto encoding =
        cast<triton::gpu::DistributedEncodingTrait>(rtType.getEncoding());
    SmallVector<std::pair<unsigned, unsigned>> intervalPads{{1, 1}};
    auto order = encoding.getRepOrder();
    auto ctaLayout = getCTALayout(encoding);
    auto sharedEncoding = triton::gpu::PaddedSharedEncodingAttr::get(
        ctx, intervalPads, order, shape, ctaLayout);
    auto memdesc = triton::gpu::MemDescType::get(
        shape, rtType.getElementType(), sharedEncoding,
        triton::gpu::SharedMemorySpaceAttr::get(ctx), true);
    auto alloc = triton::gpu::LocalAllocOp::create(b, op->getLoc(), memdesc);
    alloc->setAttr("alloc_idx", b.getI32IntegerAttr(idx));

    allocs.push_back(alloc);
    return alloc;
  }

  // Create a global tensor shared-buffer that is externally visible
  //
  triton::FuncOp makeReader(triton::FuncOp func) {
    IRMapping map;
    auto readerFunc = cast<triton::FuncOp>(func->clone(map));
    readerFunc.setName(func.getName().str() + "__reader");

    // Replace all loads with DMA to SRAM
    // TODO: handle address loads for subsequent loads
    // TODO: preload first N tiles (needs address logic..)
    for (auto load : loads) {
      auto alloc = allocMap[load];
      auto cload = cast<triton::LoadOp>(map.lookup(load).getDefiningOp());
      auto calloc =
          cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cload);
      auto loc = cload.getLoc();
      // replace uses with null value
      auto nullValue = arith::ConstantOp::create(
          b, loc, cload.getType(), b.getZeroAttr(cload.getType()));
      cload.replaceAllUsesWith(nullValue.getResult());
      // Use local store until async copy is supported
      b.setInsertionPointAfter(cload);
      auto lstore =
          triton::gpu::LocalStoreOp::create(b, loc, cload.getResult(), calloc);
    }
    // Erase all stores
    for (auto store : stores) {
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      cstore.erase();
    }
    return readerFunc;
  }

  triton::FuncOp makeCompute(triton::FuncOp func) {
    IRMapping map;
    auto computeFunc = cast<triton::FuncOp>(func->clone(map));
    computeFunc.setName(func.getName().str() + "__compute");

    // Replace all loads with local loads
    for (auto load : loads) {
      auto alloc = allocMap[load];
      auto cload = cast<triton::LoadOp>(map.lookup(load).getDefiningOp());
      auto calloc =
          cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cload);
      auto loc = cload.getLoc();
      auto lload =
          triton::gpu::LocalLoadOp::create(b, loc, cload.getType(), calloc);
      cload.replaceAllUsesWith(lload.getResult());
    }
    // Erase all stores
    for (auto store : stores) {
      auto alloc = allocMap[store];
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      auto calloc =
          cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cstore);
      auto loc = cstore.getLoc();
      auto lstore =
          triton::gpu::LocalStoreOp::create(b, loc, cstore.getValue(), calloc);
      cstore.erase();
    }
    return computeFunc;
  }

  triton::FuncOp makeWriter(triton::FuncOp func) {
    IRMapping map;
    auto writerFunc = cast<triton::FuncOp>(func->clone(map));
    writerFunc.setName(func.getName().str() + "__writer");

    // Replace all stores with local stores
    for (auto store : stores) {
      auto alloc = allocMap[store];
      auto cstore = cast<triton::StoreOp>(map.lookup(store));
      auto calloc =
          cast<triton::gpu::LocalAllocOp>(map.lookup(alloc).getDefiningOp());
      auto b = OpBuilder(cstore);
      auto loc = cstore.getLoc();
      // Use local load until async store is supported
      auto lload = triton::gpu::LocalLoadOp::create(
          b, loc, cstore.getValue().getType(), calloc);
      cstore.getValueMutable().assign(lload.getResult());
    }
    return writerFunc;
  }
};

class CoreSpecializePass
    : public triton::cpu::impl::CoreSpecializeBase<CoreSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<triton::FuncOp> funcOps;
    for (auto func : m.getOps<triton::FuncOp>()) {
      funcOps.push_back(func);
    }

    for (auto func : funcOps) {
      Specializer(m, func);
    }
    return;
  }
};

} // namespace
