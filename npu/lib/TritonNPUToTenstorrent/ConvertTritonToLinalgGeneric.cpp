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

struct MakeRangeToLinalgGeneric
    : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp makeRangeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = makeRangeOp.getLoc();
    auto resTy = dyn_cast<RankedTensorType>(makeRangeOp.getResult().getType());
    if (!resTy || resTy.getRank() != 1)
      return failure();

    Type elemTy = resTy.getElementType();
    if (!elemTy.isSignlessInteger(32)) {
      return rewriter.notifyMatchFailure(makeRangeOp,
                                         "expect i32 element type in skeleton");
    }

    // Build empty destination
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(resTy.getDimSize(0))};
    Value init = rewriter.create<tensor::EmptyOp>(loc, sizes, elemTy,
                                                  resTy.getEncoding());

    // linalg.generic: outs-only, 1 parallel loop, map (i)->(i)
    SmallVector<AffineMap> maps = {AffineMap::get(
        /*dimCount=*/1, /*symCount=*/0, rewriter.getAffineDimExpr(0))};
    SmallVector<mlir::utils::IteratorType> iters = {
        mlir::utils::IteratorType::parallel};

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{init},
        /*indexing_maps=*/maps,
        /*iterator_types=*/iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // index i
          Value i = b.create<linalg::IndexOp>(loc, 0);
          // start/end are attributes on tt.make_range
          int64_t start = makeRangeOp.getStart(); // adjust to your accessor
          Value startIdx = b.create<arith::ConstantIndexOp>(loc, start);
          Value sumIdx = b.create<arith::AddIOp>(loc, i, startIdx);
          Value val = b.create<arith::IndexCastOp>(loc, elemTy, sumIdx);
          b.create<linalg::YieldOp>(loc, val);
        });

    rewriter.replaceOp(makeRangeOp, generic.getResult(0));
    return success();
  }
};

struct SplatToLinalgFill : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = splatOp.getLoc();
    auto resTy = dyn_cast<RankedTensorType>(splatOp.getResult().getType());
    if (!resTy)
      return failure();

    // Create an empty tensor with the same shape/elt type.
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(resTy.getRank());
    for (auto s : resTy.getShape())
      sizes.push_back(rewriter.getIndexAttr(s));

    Value init = rewriter.create<tensor::EmptyOp>(
        loc, sizes, resTy.getElementType(), resTy.getEncoding());

    Value val = splatOp.getSrc();
    if (val.getType() != resTy.getElementType()) {
      return rewriter.notifyMatchFailure(
          splatOp,
          "element type mismatch between splat value and tensor result");
    }

    auto fill =
        rewriter.create<linalg::FillOp>(loc, ValueRange{val}, ValueRange{init});
    rewriter.replaceOp(splatOp, fill.getResult(0));

    return success();
  }
};

} // namespace

struct ConvertTritonToLinalgGenericPass
    : public impl::ConvertTritonToLinalgGenericBase<
          ConvertTritonToLinalgGenericPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<triton::SplatOp>();
    target.addIllegalOp<triton::MakeRangeOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    patterns.add<SplatToLinalgFill>(typeConverter, patterns.getContext());
    patterns.add<MakeRangeToLinalgGeneric>(typeConverter,
                                           patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
