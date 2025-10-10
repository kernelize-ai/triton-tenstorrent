#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONFUNCTOFUNC
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

struct ConvertFuncOp : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(triton::isKernel(funcOp) && "only kernel functions are supported");

    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(funcOp);

    mlir::FunctionType funcTy = funcOp.getFunctionType();
    assert(funcTy.getResults().empty() &&
           "expected triton kernel to return void");
    llvm::errs() << "new funcTy = " << funcTy << "\n";

    func::FuncOp newFuncOp =
        rewriter.create<func::FuncOp>(loc, funcOp.getName(), funcTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFuncOp->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.eraseOp(funcOp); // old symbol is gone; newFunc replaces it
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp retOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, adaptor.getOperands());
    return success();
  }
};

struct ConvertTritonFuncToFuncPass
    : public impl::ConvertTritonFuncToFuncBase<ConvertTritonFuncToFuncPass> {
  using impl::ConvertTritonFuncToFuncBase<
      ConvertTritonFuncToFuncPass>::ConvertTritonFuncToFuncBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addIllegalOp<triton::FuncOp>();
    target.addIllegalOp<triton::ReturnOp>();
    target.addIllegalOp<triton::CallOp>(); // should be removed by inliner
    target.addLegalDialect<func::FuncDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    patterns.add<ConvertFuncOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertReturnOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
