#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

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
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tensor::SplatOp>(op, adaptor.getSrc(),
                                                 tensorType.getShape());

    return success();
  }
};

} // namespace

void mlir::triton::npu::tt::populateViewOpToTenstorrentPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<SplatOpConversion>(typeConverter, patterns.getContext(),
                                  benefit);
}
