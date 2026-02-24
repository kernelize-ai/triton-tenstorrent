#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTMATERIALIZEMULTICASTS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-materialize-multicasts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

class MulticastTiledDotOperandLoad
    : public mlir::OpRewritePattern<triton::DescriptorLoadOp> {
public:
  MulticastTiledDotOperandLoad(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<triton::DescriptorLoadOp>(context, benefit) {}

  LogicalResult matchAndRewrite(triton::DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<npu::tt::MulticastOp>())
      return failure();

    auto opTensorTy = cast<RankedTensorType>(op.getType());
    auto tiledDotOperandEncodingAttr =
        dyn_cast<npu::tt::TiledDotOperandEncodingAttr>(
            opTensorTy.getEncoding());
    if (!tiledDotOperandEncodingAttr ||
        tiledDotOperandEncodingAttr.getOpIdx() == 1)
      return failure();

    gpu::LocalStoreOp store;
    for (auto user : op->getUsers()) {
      if (auto _store = dyn_cast<gpu::LocalStoreOp>(user))
        store = _store;
    }
    if (!store) {
      LDBG("Expected a local store user for descriptor load op: " << op
                                                                  << "\n");
      return failure();
    }

    // we need to track the cb address that we want to multicast to. Because it
    // can be difficult to re-associate the local store with the load during
    // lowering we store the target CB as an operand to the multicast and carry
    // it through the yield. This allows us to easily access the target CB
    // during lowering when we need to emit the loads and cb push ops.
    Value destAddr = store.getDst();

    auto multicast = npu::tt::MulticastOp::create(
        rewriter, op->getLoc(), op->getResultTypes(), destAddr);
    rewriter.createBlock(&multicast.getBody());
    rewriter.setInsertionPointToStart(&multicast.getBody().front());
    auto newLoadOp = rewriter.clone(*op);
    npu::tt::YieldOp::create(rewriter, op->getLoc(),
                             {newLoadOp->getResult(0), destAddr});
    op->replaceAllUsesWith(multicast->getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

class TritonTenstorrentMaterializeMulticastsPass
    : public npu::impl::TritonTenstorrentMaterializeMulticastsBase<
          TritonTenstorrentMaterializeMulticastsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;

    patterns.add<MulticastTiledDotOperandLoad>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
