#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::tt;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_FINALIZECBTRANSACTION
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

namespace {

struct TTKernelCBPopInputs : public OpRewritePattern<ttkernel::CBWaitFrontOp> {
  using OpRewritePattern<ttkernel::CBWaitFrontOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::CBWaitFrontOp op,
                                PatternRewriter &rewriter) const final {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected func::FuncOp as a parent of CBWaitFrontOp");
    if (std::distance(
            funcOp.getBody().getOps<ttkernel::CBPopFrontOp>().begin(),
            funcOp.getBody().getOps<ttkernel::CBPopFrontOp>().end()) ==
        std::distance(
            funcOp.getBody().getOps<ttkernel::CBWaitFrontOp>().begin(),
            funcOp.getBody().getOps<ttkernel::CBWaitFrontOp>().end())) {
      // is this a good heuristic for pass termination? do we always have a pop
      // front for every wait front?
      return failure();
    }

    auto packTileOpItr =
        funcOp.getBody().getOps<ttkernel::PackTileOp>().begin();
    assert(packTileOpItr !=
               funcOp.getBody().getOps<ttkernel::PackTileOp>().end() &&
           "expected at least one PackTileOp in the function body");

    rewriter.setInsertionPointAfter(*packTileOpItr);
    rewriter.create<ttkernel::CBPopFrontOp>(op.getLoc(), op.getCb(),
                                            op.getNumPages());

    return success();
  }
};

struct TTKernelCBPushOutputs : public OpRewritePattern<ttkernel::PackTileOp> {
  using OpRewritePattern<ttkernel::PackTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::PackTileOp packTileOp,
                                PatternRewriter &rewriter) const final {
    auto funcOp = packTileOp->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected func::FuncOp as a parent of CBWaitFrontOp");
    if (std::distance(
            funcOp.getBody().getOps<ttkernel::CBPushBackOp>().begin(),
            funcOp.getBody().getOps<ttkernel::CBPushBackOp>().end()) ==
        std::distance(funcOp.getBody().getOps<ttkernel::PackTileOp>().begin(),
                      funcOp.getBody().getOps<ttkernel::PackTileOp>().end())) {
      // is this a good heuristic for pass termination?
      return failure();
    }

    // get a slice from the packtile backwards so we can find the cb root op
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    (void)getBackwardSlice(packTileOp, &backwardSlice, opt);

    auto cbOpIt = llvm::find_if(backwardSlice, [](Operation *op) {
      return isa<ttkernel::GetCompileArgValOp>(op) &&
             isa<ttkernel::CBType>(op->getResult(0).getType());
    });
    assert(cbOpIt != backwardSlice.end() &&
           "expected to find CB root op in backward slice");
    ttkernel::GetCompileArgValOp cbOp =
        cast<ttkernel::GetCompileArgValOp>(*cbOpIt);

    rewriter.setInsertionPointAfter(packTileOp);
    auto numPages = rewriter.create<arith::ConstantOp>(
        packTileOp.getLoc(), rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    rewriter.create<ttkernel::CBPushBackOp>(packTileOp.getLoc(), cbOp,
                                            numPages);

    return success();
  }
};

} // namespace

struct FinalizeCBTransactionPass
    : public npu::impl::FinalizeCBTransactionBase<FinalizeCBTransactionPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<TTKernelCBPushOutputs>(context);
    patterns.add<TTKernelCBPopInputs>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
