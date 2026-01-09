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

#define GEN_PASS_DEF_TRITONTENSTORRENTACCELERATEMATMUL
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-accelerate-matmul"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

class PropagateTiledEncoding : public mlir::OpRewritePattern<triton::DotOp> {

public:
  PropagateTiledEncoding(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<triton::DotOp>(context, benefit) {}

  LogicalResult matchAndRewrite(triton::DotOp op,
                                PatternRewriter &rewriter) const override {

    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto oldCType = cast<RankedTensorType>(c.getType());
    if (isa<npu::tt::TiledEncodingAttr>(oldAType.getEncoding()) &&
        isa<npu::tt::TiledEncodingAttr>(oldBType.getEncoding()) &&
        isa<npu::tt::TiledEncodingAttr>(oldCType.getEncoding())) {
      return failure();
    }

    auto oldRetType = cast<RankedTensorType>(op.getType());

    int64_t mShape = oldAType.getShape()[0];
    int64_t kShape = oldAType.getShape()[1];
    int64_t nShape = oldBType.getShape()[1];

    auto mTiles = mShape / tileSize[0];
    auto nTiles = nShape / tileSize[1];
    auto kTiles = kShape / tileSize[1];

    auto rowMajorOrder = SmallVector<unsigned>{1, 0};

    auto updateOperandEncoding = [](Value v, int opIdx,
                                    RankedTensorType newRetType,
                                    PatternRewriter &rewriter) {
      auto vType = cast<RankedTensorType>(v.getType());
#if 1
      // TODO: DotOperandEncodingAttr fails on our new encoding. For now only
      // dot ops use the TileEncoding, but we may need our own custom
      // DotOperandEncodingAttr later.
      auto newVEncoding = newRetType.getEncoding();
#else
      auto newVEncoding = gpu::DotOperandEncodingAttr::get(
          v.getContext(), opIdx, newRetType.getEncoding(), v.getType());
#endif
      auto newVType = vType.cloneWithEncoding(newVEncoding);
      return gpu::ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
    };

    auto newAEnc = npu::tt::TiledEncodingAttr::get(
        rewriter.getContext(),
        /*tilesPerCore=*/
        ArrayRef<unsigned>{static_cast<unsigned>(mTiles),
                           static_cast<unsigned>(kTiles)},
        rowMajorOrder,
        /*tileShape=*/ArrayRef<unsigned>{tileSize[0], tileSize[1]});

    auto newBEnc = npu::tt::TiledEncodingAttr::get(
        rewriter.getContext(),
        /*tilesPerCore=*/
        ArrayRef<unsigned>{static_cast<unsigned>(kTiles),
                           static_cast<unsigned>(nTiles)},
        rowMajorOrder,
        /*tileShape=*/ArrayRef<unsigned>{tileSize[0], tileSize[1]});

    auto newCEnc = npu::tt::TiledEncodingAttr::get(
        rewriter.getContext(),
        /*tilesPerCore=*/
        ArrayRef<unsigned>{static_cast<unsigned>(mTiles),
                           static_cast<unsigned>(nTiles)},
        rowMajorOrder,
        /*tileShape=*/ArrayRef<unsigned>{tileSize[0], tileSize[1]});

    auto aCvt = updateOperandEncoding(a, 0, oldAType.cloneWithEncoding(newAEnc),
                                      rewriter);
    auto bCvt = updateOperandEncoding(b, 1, oldBType.cloneWithEncoding(newBEnc),
                                      rewriter);
    auto cCvt = updateOperandEncoding(c, 2, oldCType.cloneWithEncoding(newCEnc),
                                      rewriter);

    auto newDot = triton::DotOp::create(rewriter, op.getLoc(),
                                        oldRetType.cloneWithEncoding(newCEnc),
                                        aCvt, bCvt, cCvt);

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot->getResult(0));
    return success();
  }

protected:
  static constexpr std::array<unsigned, 2> tileSize = {32, 32};
};

} // namespace

class TritonTenstorrentAccelerateMatmulPass
    : public npu::impl::TritonTenstorrentAccelerateMatmulBase<
          TritonTenstorrentAccelerateMatmulPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;

    patterns.add<PropagateTiledEncoding>(context, benefitDefault);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
