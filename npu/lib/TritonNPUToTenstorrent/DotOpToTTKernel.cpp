#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "Utility.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func.getSymName().ends_with("__compute")) {
      // Hack until we can fix CoreSpecialize
      LDBG("Deleting dot op from non-compute kernel");
      rewriter.replaceOp(op, adaptor.getC());
      return success();
    }

    Location loc = op.getLoc();
    auto dialect =
        op->getContext()->getLoadedDialect<tt::TritonTenstorrentDialect>();
    int64_t alloc_offset =
        dialect->getAllocOffsetAttrHelper().getAttr(op).getInt();
    int64_t alloc_size = dialect->getAllocSizeAttrHelper().getAttr(op).getInt();
    LDBG("Dot op allocation offset: " << alloc_offset
                                      << " size: " << alloc_size);

    auto typeConverter = getTypeConverter();

    ttkernel::CBType aCBType = cast<ttkernel::CBType>(adaptor.getA().getType());
    ttkernel::CBType bCBType = cast<ttkernel::CBType>(adaptor.getB().getType());

    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    if (aType.getRank() != 2 || bType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2D dot");

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    int64_t mTiles = (M + kTileDimSize - 1) / kTileDimSize;
    int64_t kTiles = (K + kTileDimSize - 1) / kTileDimSize;
    int64_t nTiles = (N + kTileDimSize - 1) / kTileDimSize;

    assert(mTiles * kTiles == aCBType.getNumTiles() &&
           "a cb num tiles does not match a tensor shape");
    assert(kTiles * nTiles == bCBType.getNumTiles() &&
           "b cb num tiles does not match b tensor shape");
    assert(mTiles * nTiles == alloc_size &&
           "output size does not match output tensor shape");

    Value aNumInputTilesValue = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(aCBType.getNumTiles()));
    Value bNumInputTilesValue = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(bCBType.getNumTiles()));

    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getA(),
                                    aNumInputTilesValue);
    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getB(),
                                    bNumInputTilesValue);

    for (int64_t k = 0; k < kTiles; ++k) {
      // reset DEST index for each K pass
      int64_t crtDestIndex = alloc_offset;

      for (int64_t m = 0; m < mTiles; ++m) {
        for (int64_t n = 0; n < nTiles; ++n) {

          if (crtDestIndex > 7) {
            // see
            // https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#id2
            LDBG("MatmulTilesOp dest register index: " << crtDestIndex << "\n");
            return failure();
          }
          Value destRegIdxVal =
              arith::createIndexConstant(loc, rewriter, crtDestIndex);

          // TODO: read from layouts
          int64_t aTileIndex = m * kTiles + k;
          int64_t bTileIndex = n * kTiles + k;

          Value aTileIndexVal = arith::createConstantI32(
              loc, rewriter, static_cast<int32_t>(aTileIndex));
          Value bTileIndexVal = arith::createConstantI32(
              loc, rewriter, static_cast<int32_t>(bTileIndex));

          ttkernel::MatmulTilesOp::create(rewriter, loc, adaptor.getA(),
                                          adaptor.getB(), aTileIndexVal,
                                          bTileIndexVal, destRegIdxVal);

          crtDestIndex++;
        }
      }
    }

    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getA(),
                                   aNumInputTilesValue);
    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getB(),
                                   bNumInputTilesValue);

    // foward the dot op c operand to the users of the dot op
    rewriter.replaceOp(op, adaptor.getC());
    return success();
  }
};

} // namespace

void populateDotOpConversionPattern(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<ConvertDotOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
