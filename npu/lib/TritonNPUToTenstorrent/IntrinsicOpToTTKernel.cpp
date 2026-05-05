#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace triton {
namespace npu {

namespace {

struct RemoveLLVMAssume : OpConversionPattern<LLVM::AssumeOp> {
  using OpConversionPattern<LLVM::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateIntrinsicOpConversionPattern(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit) {
  patterns.add<RemoveLLVMAssume>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
