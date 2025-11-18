#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "Utility.h"

namespace mlir {
namespace triton {
namespace npu {

namespace {

template <typename OpTy>
struct ViewOpEraser : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcOp = op.getSrc().getDefiningOp();
    if (isa<IntegerType>(adaptor.getSrc().getType())) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    // otherwise just erase the splat
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateViewOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ViewOpEraser<triton::SplatOp>>(typeConverter,
                                              patterns.getContext());
  patterns.add<ViewOpEraser<triton::ExpandDimsOp>>(typeConverter,
                                                   patterns.getContext());
  patterns.add<ViewOpEraser<triton::BroadcastOp>>(typeConverter,
                                                  patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
