#include "PatternTritonNPUOpToTenstorrent.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct AddPtrOpConversion : public OpConversionPattern<AddPtrOp> {
  explicit AddPtrOpConversion(TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: ignore for now, maybe we can fold this into the load lowering?
    return failure();
  }
};

} // namespace

void mlir::triton::npu::tt::populateElementwiseOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, patterns.getContext(),
                                   benefit);
}
