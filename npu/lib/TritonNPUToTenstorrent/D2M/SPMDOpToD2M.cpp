#include "PatternTritonNPUToD2M.h"

#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {
namespace experimental {

namespace {

template <typename OpTy>
class BlockIndexOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

public:
  explicit BlockIndexOpConversion(TypeConverter &typeConverter,
                                  const int funcArgIndexOffset,
                                  MLIRContext *context)
      : OpConversionPattern<OpTy>(typeConverter, context),
        funcArgIndexOffset(funcArgIndexOffset) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->template getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected FuncOp as a parent of BlockIndexOp");

    unsigned numArgs = funcOp.getNumArguments();
    Value blockArg = funcOp.getArgument(numArgs - 1 + funcArgIndexOffset);
    rewriter.replaceOp(op, blockArg);
    return success();
  }

private:
  const int funcArgIndexOffset;
};

struct CurrentBlockConversion
    : public OpConversionPattern<cpu::CurrentBlockOp> {
  using OpConversionPattern<cpu::CurrentBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::CurrentBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

} // namespace

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<BlockIndexOpConversion<mlir::triton::cpu::BlockStartOp>>(
      typeConverter, /*offset=*/-1, patterns.getContext());
  patterns.add<BlockIndexOpConversion<mlir::triton::cpu::BlockEndOp>>(
      typeConverter, /*offset=*/0, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
