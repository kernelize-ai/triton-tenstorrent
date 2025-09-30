#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetProgramIdOpConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {

  explicit GetProgramIdOpConversion(TypeConverter &typeConverter,
                                    MLIRContext *context,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    Value programId = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);

    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::npu::tt::populateSPMDOpToTenstorrentPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, patterns.getContext(),
                                         targetInfo, benefit);
}
