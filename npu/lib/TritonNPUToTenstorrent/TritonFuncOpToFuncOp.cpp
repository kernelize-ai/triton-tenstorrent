#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "PatternTritonNPUToTenstorrent.h"
#include "Utility.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline tt::ttkernel::ThreadType
getThreadTypeFromFunctionName(StringRef funcName) {
  if (funcName.ends_with("__compute"))
    return tt::ttkernel::ThreadType::Compute;
  else if (funcName.ends_with("__reader") || funcName.ends_with("__writer"))
    return tt::ttkernel::ThreadType::Noc;

  llvm_unreachable("unexpected function name suffix");
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

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();
    auto typeConverter = getTypeConverter();

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    // build the arg spec using the function ops - tt.ptr ops become cb ports,
    // others are ignored
    // TODO: this holds true for the compute kernel - what about reader/writer?
    SmallVector<ttkernel::ArgAttr> ctArgs;
    for (auto argType : tritonTy.getInputs()) {
      if (isa<PointerType>(argType)) {
        ctArgs.push_back(rewriter.getAttr<ttkernel::ArgAttr>(
            ttkernel::ArgType::CBPort, ctArgs.size()));
      }
    }

    Block &oldEntry = funcOp.getBody().front();
    const unsigned numArgs = oldEntry.getNumArguments();

    // create a new function with no arguments
    mlir::FunctionType newTy = mlir::FunctionType::get(
        context, /*inputs=*/TypeRange{}, /*results=*/TypeRange{});
    // skip triton attributes

    auto newFunc =
        mlir::func::FuncOp::create(rewriter, loc, funcOp.getName(), newTy);
    if (auto vis = funcOp.getOperation()->getAttr(
            SymbolTable::getVisibilityAttrName()))
      newFunc->setAttr(SymbolTable::getVisibilityAttrName(), vis);

    newFunc->setAttr(tt::ttkernel::ThreadTypeAttr::name,
                     rewriter.getAttr<tt::ttkernel::ThreadTypeAttr>(
                         getThreadTypeFromFunctionName(funcOp.getName())));

    SmallVector<ttkernel::ArgAttr> rtArgs;

    ttkernel::ArgSpecAttr::setArgSpec(
        newFunc, rewriter.getAttr<ttkernel::ArgSpecAttr>(rtArgs, ctArgs));

    // copy the body
    Region &newRegion = newFunc.getBody();

    auto *newEntry = rewriter.createBlock(&newRegion);
    // copy all the blocks first, then we'll fix up the arguments and merge the
    // fix in
    rewriter.inlineRegionBefore(funcOp.getBody(), newRegion, newRegion.end());

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(newEntry);

      SmallVector<Value> argReplacements;
      argReplacements.reserve(oldEntry.getNumArguments());
      for (auto [idx, arg] : llvm::enumerate(oldEntry.getArguments())) {
        Type oldType = arg.getType();
        Type newType = typeConverter->convertType(oldType);

        LDBG("Replacing arg " << idx << " of type " << oldType << " with type "
                              << newType);

        Value indexVal = arith::createIndexConstant(loc, rewriter, idx);
        Value getArgVal =
            ttkernel::GetArgValOp::create(rewriter, loc, newType, indexVal);

        argReplacements.push_back(getArgVal);
      }
      rewriter.mergeBlocks(&oldEntry, newEntry, argReplacements);
    }

    // Number of user args
    // TODO: add launch params (grid size, block size, shared memory size, etc)
    newFunc->setAttr("tt.num_args", rewriter.getI32IntegerAttr(numArgs));

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
