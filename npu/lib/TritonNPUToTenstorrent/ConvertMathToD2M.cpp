#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTMATHTOD2M
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

struct ConvertAddOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting AddOp to D2M: " << addOp << "\n";
    auto typeConverter = getTypeConverter();
    Type retType = typeConverter->convertType(addOp.getType());
    rewriter.replaceOpWithNewOp<tt::d2m::TileAddOp>(
        addOp, retType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertMathTOD2MPass
    : public impl::ConvertMathToD2MBase<ConvertMathTOD2MPass> {
  using impl::ConvertMathToD2MBase<ConvertMathTOD2MPass>::ConvertMathToD2MBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addIllegalOp<arith::AddFOp>();
    target.addLegalDialect<tt::d2m::D2MDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) {
      return tt::ttcore::TileType::get(type.getElementType());
    });

    RewritePatternSet patterns(context);
    patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
