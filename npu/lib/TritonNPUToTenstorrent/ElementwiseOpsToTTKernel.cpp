#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "Utility.h"

namespace mlir {
namespace triton {
namespace npu {

namespace {

struct ConvertAddPtrOp : public OpConversionPattern<AddPtrOp> {
  using OpConversionPattern<AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto baseAddr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    if (!isa<IntegerType>(baseAddr.getType()) ||
        !isa<IntegerType>(offset.getType())) {
      rewriter.eraseOp(op);
      return success();
    }

    auto type = cast<triton::PointerType>(op.getPtr().getType());
    auto elemType = type.getPointeeType();
    auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
    Value elemSizeValue = arith::createConstantI32(loc, rewriter, elemSize);
    offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);
    auto newAddPtrOp = arith::AddIOp::create(rewriter, loc, baseAddr, offset);
    rewriter.replaceOp(op, newAddPtrOp.getResult());

    return success();
  }
};

template <typename OpTy>
struct ArithBinaryOpOnTensorsConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getLhs().getType()) &&
        isa<RankedTensorType>(op.getRhs().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // namespace

void populateElementwiseOpConversionPattern(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<ConvertAddPtrOp>(typeConverter, patterns.getContext());

#define POPULATE_ARITH_BINARY_OP_ON_TENSORS(OP)                                \
  patterns.add<ArithBinaryOpOnTensorsConversion<OP>>(typeConverter,            \
                                                     patterns.getContext());

  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::AddIOp);
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::MulIOp);
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::RemSIOp);

  // TODO: can this be handled by RemoveRedundantMasks instead?
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::CmpIOp);
}

} // namespace npu
} // namespace triton
} // namespace mlir
