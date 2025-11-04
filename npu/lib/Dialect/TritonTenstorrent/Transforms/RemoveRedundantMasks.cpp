#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/Passes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTREMOVEREDUNDANTMASKS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-remove-redundant-masks"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Search for tail mask pattern:
// mask = arith.cmpi slt, splat(start) + range, splat(nelems))
bool isTailMask(Value mask, uint32_t blockSize) {
  auto cmp = dyn_cast<arith::CmpIOp>(mask.getDefiningOp());
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();

  // we can't actually identify that the function param is # elements, so we
  // will just look for a constant integer value here
  auto rhsSplat = dyn_cast<SplatOp>(rhs.getDefiningOp());
  if (!rhsSplat || !rhsSplat.getSrc().getType().isInteger(32))
    return false;

  auto lhsAdd = dyn_cast<arith::AddIOp>(lhs.getDefiningOp());
  if (!lhsAdd)
    return false;

  Value a = lhsAdd.getLhs();
  Value b = lhsAdd.getRhs();

  // a: splat(start)
  if (!isa<SplatOp>(a.getDefiningOp()))
    return false;

  SetVector<Operation *> blockStartSlice;
  (void)getBackwardSlice(a, &blockStartSlice);
  LDBG("Retrieving slice for block start op");
  for (Operation *op : blockStartSlice) {
    LDBG("  slice op: " << *op << "\n");
  }

  if (blockStartSlice.empty() ||
      !isa<triton::GetProgramIdOp>(blockStartSlice.front()))
    return false;

  /// b: make_range
  auto makeRange = dyn_cast<triton::MakeRangeOp>(b.getDefiningOp());
  if (!makeRange)
    return false;
  return makeRange.getEnd() - makeRange.getStart() == blockSize;
}

struct ReplaceTailMaskForLoads : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    RankedTensorType maskType = cast<RankedTensorType>(mask.getType());
    if (!isTailMask(mask, maskType.getDimSize(0)))
      return failure();

    LDBG("Found redundant tail mask: " << *mask.getDefiningOp() << "\n");

    // replace with unmasked load
    LoadOp newLoad = rewriter.create<LoadOp>(
        loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(), loadOp.getEvict(),
        loadOp.getIsVolatile());
    rewriter.replaceOp(loadOp, newLoad.getResult());
    return success();
  }
};

struct ReplaceTailMaskForStores : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Value mask = storeOp.getMask();
    if (!mask)
      return failure();

    RankedTensorType maskType = cast<RankedTensorType>(mask.getType());
    if (!isTailMask(mask, maskType.getDimSize(0)))
      return failure();

    LDBG("Found redundant tail mask: " << *mask.getDefiningOp() << "\n");

    // replace with unmasked store
    StoreOp newStore = rewriter.create<StoreOp>(
        storeOp.getLoc(), storeOp.getPtr(), storeOp.getValue(),
        storeOp.getCache(), storeOp.getEvict());
    rewriter.replaceOp(storeOp, newStore);
    return success();
  }
};

} // namespace

class RemoveRedundantMasksPass
    : public triton::npu::impl::TritonTenstorrentRemoveRedundantMasksBase<
          RemoveRedundantMasksPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ReplaceTailMaskForStores>(context);
    patterns.add<ReplaceTailMaskForLoads>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
