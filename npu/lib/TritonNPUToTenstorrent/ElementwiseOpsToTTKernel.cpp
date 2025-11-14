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

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (!isa<IntegerType>(baseAddr.getType()) ||
        !isa<IntegerType>(offset.getType())) {
      return failure();
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

} // namespace

void populateElementwiseOpConversionPattern(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<ConvertAddPtrOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
