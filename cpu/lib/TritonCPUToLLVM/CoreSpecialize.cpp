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

static constexpr llvm::StringLiteral kAllocIdxAttrName =
    "__core_specialize.alloc_idx";

static bool isLoadLike(Operation *op) {
  return isa<triton::LoadOp, triton::DescriptorLoadOp>(op);
}

static bool isStoreLike(Operation *op) {
  return isa<triton::StoreOp, triton::DescriptorStoreOp>(op);
}

static Type getLoadLikeResultType(Operation *op) {
  if (auto l = dyn_cast<triton::LoadOp>(op))
    return l.getType();
  if (auto l = dyn_cast<triton::DescriptorLoadOp>(op))
    return l.getType();
  llvm_unreachable("not a load-like op");
}

static Value getStoreLikeValue(Operation *op) {
  if (auto s = dyn_cast<triton::StoreOp>(op))
    return s.getValue();
  if (auto s = dyn_cast<triton::DescriptorStoreOp>(op))
    return s.getSrc();
  llvm_unreachable("not a store-like op");
}

static void setStoreLikeValue(Operation *op, Value v) {
  if (auto s = dyn_cast<triton::StoreOp>(op)) {
    s.getValueMutable().assign(v);
    return;
  }
  if (auto s = dyn_cast<triton::DescriptorStoreOp>(op)) {
    s.getSrcMutable().assign(v);
    return;
  }
  llvm_unreachable("not a store-like op");
}

static bool isDependentLoadLike(Operation *loadLike) {
  for (OpResult r : loadLike->getResults()) {
    for (OpOperand &use : r.getUses()) {
      if (isLoadLike(use.getOwner()))
        return true;
    }
  }
  return false;
}

static std::optional<int64_t> getAllocIdx(Operation *op) {
  // alloc_idx is used elsewhere in the Tenstorrent lowering, while
  // kAllocIdxAttrName is specific to this pass and erased afterwards
  llvm::StringRef attrName =
      isa<triton::gpu::LocalAllocOp>(op) ? "alloc_idx" : kAllocIdxAttrName;
  if (auto a = op->getAttrOfType<IntegerAttr>(attrName))
    return a.getInt();
  return std::nullopt;
}

class Specializer {
  triton::FuncOp func;
  SmallVector<triton::gpu::LocalAllocOp> allocs;
  SmallVector<Operation *> loads; // triton::LoadOp or triton::DescriptorLoadOp
  SmallVector<Operation *>
      stores; // triton::StoreOp or triton::DescriptorStoreOp

public:
  Specializer(ModuleOp m, triton::FuncOp _func) : func(_func) {
    // Collect load/store ops and create shared buffers (one per op).
    func.walk([&](Operation *op) {
      if (isStoreLike(op)) {
        int64_t idx = allocs.size();
        createSharedBuffer(op, getStoreLikeValue(op).getType(), idx);
        op->setAttr(
            kAllocIdxAttrName,
            IntegerAttr::get(IntegerType::get(func.getContext(), 64), idx));
        stores.push_back(op);
      } else if (isLoadLike(op)) {
        if (isDependentLoadLike(op))
          return;

        int64_t idx = allocs.size();
        createSharedBuffer(op, getLoadLikeResultType(op), idx);
        op->setAttr(
            kAllocIdxAttrName,
            IntegerAttr::get(IntegerType::get(func.getContext(), 64), idx));
        loads.push_back(op);
      }
    });

    auto readerFunc = makeReader(func);
    auto computeFunc = makeCompute(func);
    auto writerFunc = makeWriter(func);

    // Insert clones before the original for stable ordering.
    readerFunc->moveBefore(func);
    computeFunc->moveBefore(func);
    writerFunc->moveBefore(func);
  }

  ~Specializer() {
    allocs.clear();
    loads.clear();
    stores.clear();
    func.erase();
  }

  // Create a global tensor shared-buffer that is externally visible
  //
  void createSharedBuffer(Operation *op, Type type, int64_t idx) {
    auto rtType = cast<RankedTensorType>(type);
    MLIRContext *ctx = func.getContext();

    Block &entry = func.getBody().front();
    OpBuilder b(&entry, entry.begin());

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
  }

  // build the alloc map on the fly after cloning functions to avoid map lookups
  DenseMap<int64_t, triton::gpu::LocalAllocOp> buildAllocMap(triton::FuncOp f) {
    DenseMap<int64_t, triton::gpu::LocalAllocOp> map;
    f.walk([&](triton::gpu::LocalAllocOp a) {
      if (auto idx = getAllocIdx(a.getOperation()))
        map[*idx] = a;
    });
    return map;
  }

  void rewriteReader(triton::FuncOp f) {
    auto allocIdMap = buildAllocMap(f);

    // Replace all loads with DMA to SRAM
    // TODO: handle address loads for subsequent loads
    // TODO: preload first N tiles (needs address logic..)
    f.walk([&](Operation *op) {
      if (!isLoadLike(op))
        return;
      auto idx = getAllocIdx(op);
      if (!idx)
        return;

      auto it = allocIdMap.find(*idx);
      if (it == allocIdMap.end())
        return;

      auto alloc = it->second;
      OpBuilder b(op);
      Location loc = op->getLoc();

      // replace uses with null value
      assert(op->getNumResults() == 1 &&
             "expected load-like op to have a single result");
      Type ty = op->getResult(0).getType();
      auto nullValue = arith::ConstantOp::create(b, loc, ty, b.getZeroAttr(ty));
      op->getResult(0).replaceAllUsesWith(nullValue.getResult());

      // Use local store until async copy is supported
      b.setInsertionPointAfter(op);
      triton::gpu::LocalStoreOp::create(b, loc, op->getResult(0), alloc);
    });

    // Erase all stores
    f.walk([&](Operation *op) {
      if (isStoreLike(op) && getAllocIdx(op))
        op->erase();
    });
  }

  void rewriteCompute(triton::FuncOp f) {
    auto allocIdMap = buildAllocMap(f);

    // Replace all loads with local loads
    f.walk([&](Operation *op) {
      if (!isLoadLike(op))
        return;

      auto idx = getAllocIdx(op);
      if (!idx)
        return;

      auto it = allocIdMap.find(*idx);
      if (it == allocIdMap.end())
        return;

      auto alloc = it->second;
      OpBuilder b(op);
      Location loc = op->getLoc();

      Type ty = getLoadLikeResultType(op);
      auto lload = triton::gpu::LocalLoadOp::create(b, loc, ty, alloc);
      op->getResult(0).replaceAllUsesWith(lload.getResult());
      op->erase();
    });

    // Erase all stores
    f.walk([&](Operation *op) {
      if (!isStoreLike(op))
        return;

      auto idx = getAllocIdx(op);
      if (!idx)
        return;

      auto it = allocIdMap.find(*idx);
      if (it == allocIdMap.end())
        return;

      auto alloc = it->second;
      OpBuilder b(op);
      Location loc = op->getLoc();

      triton::gpu::LocalStoreOp::create(b, loc, getStoreLikeValue(op), alloc);
      op->erase();
    });
  }

  void rewriteWriter(triton::FuncOp f) {
    auto allocIdMap = buildAllocMap(f);

    // Replace all stores with local stores
    f.walk([&](Operation *op) {
      if (!isStoreLike(op))
        return;

      auto idx = getAllocIdx(op);
      if (!idx)
        return;

      auto it = allocIdMap.find(*idx);
      if (it == allocIdMap.end())
        return;

      auto alloc = it->second;
      OpBuilder b(op);
      Location loc = op->getLoc();

      // Use local load until async store is supported
      auto lload = triton::gpu::LocalLoadOp::create(
          b, loc, getStoreLikeValue(op).getType(), alloc);
      setStoreLikeValue(op, lload.getResult());
    });
  }

  triton::FuncOp makeReader(triton::FuncOp func) {
    IRMapping map;
    auto readerFunc = cast<triton::FuncOp>(func->clone(map));
    readerFunc.setName(func.getName().str() + "__reader");
    rewriteReader(readerFunc);
    return readerFunc;
  }

  triton::FuncOp makeCompute(triton::FuncOp func) {
    IRMapping map;
    auto computeFunc = cast<triton::FuncOp>(func->clone(map));
    computeFunc.setName(func.getName().str() + "__compute");
    rewriteCompute(computeFunc);
    return computeFunc;
  }

  triton::FuncOp makeWriter(triton::FuncOp func) {
    IRMapping map;
    auto writerFunc = cast<triton::FuncOp>(func->clone(map));
    writerFunc.setName(func.getName().str() + "__writer");
    rewriteWriter(writerFunc);
    return writerFunc;
  }
};

} // namespace
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
