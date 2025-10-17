#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONTOLINALGGENERIC
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

using namespace mlir;
using namespace triton;

namespace {

struct SplatToLinalgFill : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp,
                    OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = splatOp.getLoc();
    auto resTy = dyn_cast<RankedTensorType>(splatOp.getResult().getType());
    if (!resTy) return failure();

    // Create an empty tensor with the same shape/elt type.
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(resTy.getRank());
    for (auto s : resTy.getShape())
      sizes.push_back(rewriter.getIndexAttr(s));

    Value init = rewriter.create<tensor::EmptyOp>(loc, sizes, resTy.getElementType(), resTy.getEncoding());

    Value val = splatOp.getSrc();
    if (val.getType() != resTy.getElementType()) {
        return rewriter.notifyMatchFailure(splatOp, "element type mismatch between splat value and tensor result");
    }

    auto fill = rewriter.create<linalg::FillOp>(loc, ValueRange{val}, ValueRange{init});
    rewriter.replaceOp(splatOp, fill.getResult(0));

    return success();
  }
};

}

struct ConvertTritonToLinalgGenericPass : public impl::ConvertTritonToLinalgGenericBase<ConvertTritonToLinalgGenericPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalOp<triton::SplatOp>();
    // target.addIllegalOp<triton::MakeRangeOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    patterns.add<SplatToLinalgFill>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

}
}
}
