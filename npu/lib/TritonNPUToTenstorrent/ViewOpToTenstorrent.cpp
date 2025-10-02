#include "PatternTritonNPUOpToTenstorrent.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct SplatOpConversion : public OpConversionPattern<SplatOp> {
  explicit SplatOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                             PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // convert to tensor splatop?
    return failure();
  }
};

} // namespace

void mlir::triton::npu::tt::populateViewOpToTenstorrentPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<SplatOpConversion>(typeConverter, patterns.getContext(),
                                  benefit);
}
