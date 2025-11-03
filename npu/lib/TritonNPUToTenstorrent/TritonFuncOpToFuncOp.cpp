#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Analysis/SliceAnalysis.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONFUNCTOFUNC
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

namespace {

// TODO: can we remove the ::tt suffix?
inline ::tt::ttkernel::ThreadType
getThreadTypeFromFunctionName(StringRef funcName) {
  if (funcName.ends_with("__compute"))
    return ::tt::ttkernel::ThreadType::Compute;
  else if (funcName.ends_with("__reader") || funcName.ends_with("__writer"))
    return ::tt::ttkernel::ThreadType::Noc;

  assert(false && "unexpected function name suffix");
}

struct ConvertTritonFunc : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isKernel(funcOp)) {
      return rewriter.notifyMatchFailure(
          funcOp, "non-kernel functions are not yet supported");
    }

    llvm::errs() << "Converting triton.func " << funcOp.getName() << "\n";

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(funcOp);

    // create a new funcop copy over (or re-use) regions/blocks
    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    mlir::FunctionType newTy = mlir::FunctionType::get(
        context, /*inputs=*/tritonTy.getInputs(), /*results=*/TypeRange{});
    // skip triton attributes

    auto newFunc =
        rewriter.create<mlir::func::FuncOp>(loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    newFunc->setAttr(::tt::ttkernel::ThreadTypeAttr::name,
                     rewriter.getAttr<::tt::ttkernel::ThreadTypeAttr>(
                         getThreadTypeFromFunctionName(funcOp.getName())));

    // build the arg spec using the function ops - tt.ptr ops become cb ports,
    // others are ignored
    SmallVector<ttkernel::ArgAttr> ctArgs;
    for (auto argType : tritonTy.getInputs()) {
      if (isa<PointerType>(argType)) {
        ctArgs.push_back(rewriter.getAttr<ttkernel::ArgAttr>(
            ttkernel::ArgType::CBPort, ctArgs.size()));
      }
    }
    SmallVector<ttkernel::ArgAttr> rtArgs;

    ttkernel::ArgSpecAttr::setArgSpec(
        newFunc, rewriter.getAttr<ttkernel::ArgSpecAttr>(rtArgs, ctArgs));

    // copy the body
    rewriter.inlineRegionBefore(funcOp.getBody(), newFunc.getBody(),
                                newFunc.end());

    // update the arguments 
    rewriter.setInsertionPointToStart(&newFunc.getBody().front());
    for (int i = newFunc.getNumArguments() - 1; i >= 0; --i) {
      llvm::errs() << "processing arg " << i << "\n";
      llvm::errs() << "arg = " << newFunc.getArgument(i) << "\n";
      Type argType = newFunc.getArgument(i).getType();
      if (isa<PointerType>(argType)) {
        // compile time arg
        
        SetVector<Operation *> forwardSlice;
        Value argVal = newFunc.getArgument(i);
        ForwardSliceOptions opt;
        // filter for TileEncodingAttr results
#if 0
        opt.filter = [&] (Operation* op) {
          llvm::errs() << "Processing op " << *op << "\n";
          return llvm::any_of(op->getResults(), [&](Value result) {
            if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
              return isa<npu::tt::TileEncodingAttr>(tensorTy.getEncoding());
            }
            return false; 
          });
        };
#endif 
        Type newArgType;
        getForwardSlice(argVal, &forwardSlice, opt);
        // auto forwardSlice = llvm::to_vector(newFunc.getArgument(i).getUsers());
        // getForwardSlice(newFunc.getArgument(i), &forwardSlice);
        llvm::errs() << "slice for arg " << i << ":\n";
        for (Operation *op : forwardSlice) {
          auto resultItr = llvm::find_if (op->getResults(), [&](Value result) {
            if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
              return isa<npu::tt::TileEncodingAttr>(tensorTy.getEncoding());
            }
            return false; 
          });
          if (resultItr != op->getResults().end()) {
            newArgType = (*resultItr).getType();
          }
        }

        if (!newArgType) {
          assert(newFunc.getArgument(i).use_empty() && "expected unused argument");
          continue;
        }
        
        // need to get the memref type 
        rewriter.replaceAllUsesWith(
            newFunc.getArgument(i),
            rewriter.create<ttkernel::GetCompileArgValOp>(
                newFunc.getLoc(), getTypeConverter()->convertType(newArgType), i));

      } else {
        // runtime arg. 
        Value c_index = rewriter.create<arith::ConstantOp>(
            newFunc.getLoc(), rewriter.getIndexType(),
            rewriter.getIntegerAttr(rewriter.getIndexType(), i));
        rewriter.replaceAllUsesWith(
            newFunc.getArgument(i),
            rewriter.create<ttkernel::GetArgValOp>(
                newFunc.getLoc(), argType, c_index));
      }

#if 0
      assert(newFunc.getArgument(i).use_empty() && "expected unused argument");
      (void)newFunc.eraseArgument(i);
#endif
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

} // namespace

struct ConvertTritonFuncOpToFuncOpPass
    : public impl::ConvertTritonFuncToFuncBase<
          ConvertTritonFuncOpToFuncOpPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<triton::FuncOp>();
    target.addIllegalOp<triton::ReturnOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertTritonFunc>(typeConverter, patterns.getContext());
    patterns.add<ConvertReturnOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ConvertTritonFunc>(typeConverter, patterns.getContext(),
                                  benefit);
  patterns.add<ConvertReturnOp>(typeConverter, patterns.getContext(), benefit);
}

} // namespace npu
} // namespace triton
} // namespace mlir
