#include "PatternTritonNPUOpToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct TritonFuncOpConversion : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!triton::isKernel(funcOp)) {
      return rewriter.notifyMatchFailure(
          funcOp, "non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *ctx = funcOp.getContext();

    Block *entry = &funcOp.front();
    rewriter.setInsertionPointToStart(entry);

    // ttKernel lowering expects a function with no arguments and no return in
    // the signature. Arguments will retrieved with get_compile_time_arg_val op.
    // move function parameters into attributes and change usage of the
    // arguments to custom ops
    SmallVector<Value> newArgVals(entry->getNumArguments());
    for (unsigned i = 0; i < entry->getNumArguments(); ++i) {
      Type argTy = entry->getArgument(i).getType();

      auto val = rewriter.create<npu::tt::KernelArgOp>(
          loc, argTy, rewriter.getIndexAttr(i));

      // TODO: val or getResult?
      newArgVals[i] = val.getResult();
    }

    // Replace all uses then erase the block args.
    for (unsigned i = 0, e = entry->getNumArguments(); i < e; ++i)
      entry->getArgument(i).replaceAllUsesWith(newArgVals[i]);
    if (entry->getNumArguments() != 0)
      entry->eraseArguments(0, entry->getNumArguments());

    OpBuilder::InsertionGuard g(rewriter);
    // ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointAfter(funcOp);
    // create a new funcop void -> void and copy over (or re-use) regions/blocks
    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    // now that the entry block is empty, create the new function and copy over
    // the body
    mlir::FunctionType newTy = mlir::FunctionType::get(
        ctx, /*inputs=*/TypeRange{}, /*results=*/TypeRange{});

    // TODO: should we copy the name or use `ttkernel_compute`?
    auto newFunc =
        rewriter.create<mlir::func::FuncOp>(loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    // TODO: set tenstorrent specific function attributes
    // Skip copying function attributes from the old function for now.

    // copy the body
    rewriter.inlineRegionBefore(funcOp.getBody(), newFunc.getBody(),
                                newFunc.end());

    rewriter.eraseOp(funcOp); // old symbol is gone; newFunc replaces it
    return success();
  }
};

} // namespace

void mlir::triton::npu::tt::populateFuncOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonFuncOpConversion>(typeConverter, context, benefit);
}
