#include "PatternTritonNPUOpToTenstorrent.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct AddOpConversion : public OpConversionPattern<arith::AddFOp> {
  explicit AddOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // convert arith addf to ttkernel.add_tiles
    llvm::errs() << "Lowering AddFOp: " << op << "\n";

    // collect inputs
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    llvm::errs() << "lhs: " << lhs << "\n";
    llvm::errs() << "rhs: " << rhs << "\n";

    assert(false);
    return success();
  }
};

} // namespace

void mlir::triton::npu::tt::populateElementwiseOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<AddOpConversion>(typeConverter, patterns.getContext(), benefit);
}
