#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTREMOVEDOTLOADLAYOUTCONVERSIONS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-remove-dot-load-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

static void updateOperand(PatternRewriter &rewriter, Location loc,
                          Value operand, Attribute newEncoding,
                          IRMapping &mapping) {
  RankedTensorType operandTensorType =
      dyn_cast<RankedTensorType>(operand.getType());
  if (!operandTensorType)
    return;
  auto cvt = gpu::ConvertLayoutOp::create(
      rewriter, loc, operandTensorType.cloneWithEncoding(newEncoding), operand);
  mapping.map(operand, cvt);
}

struct RemoveLoadOpToDotCvt
    : public mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Evaluate cvtOp for removal: " << cvtOp << "\n");
    Value loadSrc = cvtOp.getSrc();
    auto srcOp = loadSrc.getDefiningOp();
    if (!srcOp) {
      // ignore block arguments
      return failure();
    }

    auto loadOp = dyn_cast<triton::LoadOp>(srcOp);
    auto cvtResultType = cast<RankedTensorType>(cvtOp.getType());
    if (loadOp && isa<npu::tt::TiledDotOperandEncodingAttr>(
                      cvtResultType.getEncoding())) {
      LDBG("Remove cvtOp " << *cvtOp
                           << " and push tiled_dot_op encoding into load");

      IRMapping mapping;
      for (auto operand : loadOp->getOperands()) {
        updateOperand(rewriter, loadOp.getLoc(), operand,
                      cvtResultType.getEncoding(), mapping);
      }

      Operation *newLoadOp = rewriter.clone(*loadOp, mapping);
      newLoadOp->getResult(0).setType(cvtResultType);
      LDBG("Created new LoadOp: " << *newLoadOp);

      rewriter.replaceAllUsesWith(cvtOp, newLoadOp->getResult(0));
      rewriter.eraseOp(cvtOp);
      rewriter.replaceAllUsesWith(loadOp, newLoadOp->getResult(0));
      rewriter.eraseOp(loadOp);

      return success();
    }

    return failure();
  }
};

struct RemoveDescriptorLoadOpToDotCvt
    : public mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Evaluate cvtOp for removal: " << cvtOp << "\n");
    Value descriptorLoad = cvtOp.getSrc();
    auto srcOp = descriptorLoad.getDefiningOp();
    if (!srcOp) {
      // ignore block arguments
      return failure();
    }

    auto descriptorLoadOp = dyn_cast<triton::DescriptorLoadOp>(srcOp);
    auto cvtResultType = cast<RankedTensorType>(cvtOp.getType());
    if (descriptorLoadOp && isa<npu::tt::TiledDotOperandEncodingAttr>(
                                cvtResultType.getEncoding())) {
      LDBG("Remove cvtOp " << *cvtOp
                           << " and push tiled_dot_op encoding into load");

      IRMapping mapping;
      for (auto operand : descriptorLoadOp->getOperands()) {
        updateOperand(rewriter, descriptorLoadOp.getLoc(), operand,
                      cvtResultType.getEncoding(), mapping);
      }

      Operation *newDescriptorLoad = rewriter.clone(*descriptorLoadOp, mapping);
      newDescriptorLoad->getResult(0).setType(cvtResultType);
      LDBG("Created new LoadOp: " << *newDescriptorLoad);

      rewriter.replaceAllUsesWith(cvtOp, newDescriptorLoad->getResult(0));
      rewriter.eraseOp(cvtOp);
      rewriter.replaceAllUsesWith(descriptorLoadOp,
                                  newDescriptorLoad->getResult(0));
      rewriter.eraseOp(descriptorLoadOp);

      return success();
    }

    return failure();
  }
};

} // namespace

class TritonTenstorrentRemoveDotLoadLayoutConversionsPass
    : public npu::impl::TritonTenstorrentRemoveDotLoadLayoutConversionsBase<
          TritonTenstorrentRemoveDotLoadLayoutConversionsPass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;

    patterns.add<RemoveLoadOpToDotCvt>(context, benefitDefault);
    patterns.add<RemoveDescriptorLoadOpToDotCvt>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
