#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct ReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return success();
  }
};

// TODO: call op

} // namespace

void mlir::triton::npu::tt::populateControlFlowOpToFuncOpPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReturnOpConversion>(typeConverter, patterns.getContext(),
                                   benefit);
}
