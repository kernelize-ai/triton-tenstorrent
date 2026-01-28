#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CORESPECIALIZE
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

namespace {

static Type getLoadLikeResultType(Operation *op) {
  if (auto l = dyn_cast<triton::LoadOp>(op))
    return l.getType();
  if (auto l = dyn_cast<triton::DescriptorLoadOp>(op))
    return l.getType();
  llvm_unreachable("not a load-like op");
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
  auto dialect =
      op->getContext()->getLoadedDialect<tt::TritonTenstorrentDialect>();
  auto allocIndexHelper = dialect->getAllocIndexAttrHelper();
  if (allocIndexHelper.isAttrPresent(op))
    return allocIndexHelper.getAttr(op).getInt();
  return std::nullopt;
}

class Specializer {
  triton::FuncOp func;
  SmallVector<triton::gpu::LocalAllocOp> allocs;

public:
  Specializer(ModuleOp m, triton::FuncOp _func) : func(_func) {
    auto context = func->getContext();
    auto *dialect = context->getLoadedDialect<tt::TritonTenstorrentDialect>();
    auto funcTypeHelper = dialect->getFuncTypeAttrHelper();
    auto initialFuncHelper = dialect->getInitialFuncAttrHelper();
    auto allocIndexHelper = dialect->getAllocIndexAttrHelper();
    auto cbTileSizesHelper = dialect->getCbTileSizesAttrHelper();

    // Collect load/store ops and create shared buffers (one per op).
    SmallVector<int64_t> allocIdxs;
    SmallVector<int32_t> allocSizes;
    auto addSharedBuffer = [&](Operation *op, Type valueType) {
      int64_t idx = allocIdxs.size();
      allocIdxs.push_back(idx);
      allocIndexHelper.setAttr(
          op, IntegerAttr::get(IntegerType::get(func.getContext(), 64), idx));
      createSharedBuffer(op, valueType, idx);
      allocSizes.push_back(getNumTiles(valueType));
    };
    func.walk([&](Operation *op) {
      if (isStoreLike(op)) {
        addSharedBuffer(op, getStoreLikeValue(op).getType());
      } else if (isLoadLike(op)) {
        if (isDependentLoadLike(op))
          return;

        addSharedBuffer(op, getLoadLikeResultType(op));
      }
    });

    // Set the initial func attribute, all clones will inherit this
    auto funcNameAttr = StringAttr::get(func.getContext(), func.getName());
    initialFuncHelper.setAttr(func, funcNameAttr);

    // Set the alloc sizes attribute
    auto cbTileSizesAttr =
        DenseI32ArrayAttr::get(func.getContext(), allocSizes);
    cbTileSizesHelper.setAttr(func, cbTileSizesAttr);

    auto readerFunc = makeReader(func);
    auto funcTypeValueAttr =
        tt::DerivedFuncTypeAttr::get(context, tt::DerivedFuncType::ReaderFunc);
    funcTypeHelper.setAttr(readerFunc, funcTypeValueAttr);

    auto computeFunc = makeCompute(func);
    funcTypeValueAttr =
        tt::DerivedFuncTypeAttr::get(context, tt::DerivedFuncType::ComputeFunc);
    funcTypeHelper.setAttr(computeFunc, funcTypeValueAttr);

    auto writerFunc = makeWriter(func);
    funcTypeValueAttr =
        tt::DerivedFuncTypeAttr::get(context, tt::DerivedFuncType::WriterFunc);
    funcTypeHelper.setAttr(writerFunc, funcTypeValueAttr);

    m.insert(func, readerFunc);
    m.insert(func, computeFunc);
    m.insert(func, writerFunc);
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
    auto *dialect =
        alloc->getContext()->getLoadedDialect<tt::TritonTenstorrentDialect>();
    auto allocIndexHelper = dialect->getAllocIndexAttrHelper();
    allocIndexHelper.setAttr(alloc, b.getI32IntegerAttr(idx));

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

    // Erase all stores and compute ops
    f.walk([&](Operation *op) {
      if (isStoreLike(op) && getAllocIdx(op)) {
        op->erase();
      } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        // Erase the dot op in the reader, forwarding the accumulator value to
        // its users. Because the dot op uses a loop carried accumulator it is
        // not sufficient to just erase the store.
        dotOp.getD().replaceAllUsesWith(dotOp.getC());
      }
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

    f.walk([&](Operation *op) {
      if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        // Erase the dot op in the writer, forwarding the accumulator value to
        // its users. Because the dot op uses a loop carried accumulator it is
        // not sufficient to only change the input to the store.
        dotOp.getD().replaceAllUsesWith(dotOp.getC());
      }
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
    : public npu::impl::CoreSpecializeBase<CoreSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (triton::FuncOp func :
         llvm::make_early_inc_range(m.getOps<triton::FuncOp>())) {
      Specializer spec(m, func);
      func.erase();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
