#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct MakeRangeOpConversion : public OpConversionPattern<MakeRangeOp> {

  explicit MakeRangeOpConversion(TypeConverter &typeConverter,
                                 MLIRContext *context, PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t start = op.getStartAttr().getInt();
    int32_t end = op.getEndAttr().getInt();
    assert(start < end);

    llvm::SmallVector<int32_t> values;
    values.reserve(end - start);
    for (int32_t i = start; i < end; i++) {
      values.push_back(i);
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resTy, rewriter.getI32VectorAttr(values));

    return success();
  }
};

} // namespace

void mlir::triton::npu::tt::populateMakeRangeOpToTenstorrentPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<MakeRangeOpConversion>(typeConverter, patterns.getContext(),
                                      benefit);
}
