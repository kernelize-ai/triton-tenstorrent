#include "PatternTritonNPUToD2M.h"

#include "ArgConversionHelper.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "npu/include/Analysis/Utility.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "SPMDArgs.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertTritonFunc : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isKernel(funcOp)) {
      return funcOp.emitError("non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();
    auto typeConverter = getTypeConverter();

    ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    // read the triton defined grid attribute from the module and use it to set
    // tenstorrent grid parameters
    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);

    auto grid = ttcore::GridAttr::get(context, gridAttr.getShape());

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    ArgConversionHelper helper;
    // 1. Convert function arguments and add tenstorrent specific args (block
    // start/end)
    if (failed(helper.convertFunctionArguments(funcOp, rewriter, grid,
                                               getTypeConverter()))) {
      return funcOp.emitError("failed to convert function arguments");
    }

    // 2. Generate the new function with converted signature and a single entry
    // block.
    func::FuncOp newFunc = helper.generateNewFunction(funcOp, rewriter);
    Block *newEntry = &newFunc.getBody().front();

    // 3. populate the new function body with a d2m.generic op
    rewriter.setInsertionPointToStart(newEntry);

    // Build the GenericOp in explicit data-movement form (all three of
    // indexing_maps / block_factors / iterator_types are empty).
    auto threadsAttr = rewriter.getArrayAttr(
        rewriter.getAttr<d2m::ThreadAttr>(d2m::ThreadType::Unified));
    auto emptyAttr = rewriter.getArrayAttr({});

    // generate casts from the converted function args to the types expected by
    // the d2m.generic op
    SmallVector<Value> inputArgs = helper.generateInputArgs(newFunc, rewriter);
    SmallVector<Value> outputArgs =
        helper.generateOutputArgs(newFunc, rewriter);

    auto genericOp =
        d2m::GenericOp::create(rewriter, loc,
                               /*results=*/TypeRange{},
                               /*inputs=*/inputArgs,
                               /*outputs=*/outputArgs,
                               /*additionalArgs=*/helper.getScalarArgs(newFunc),
                               /*grid=*/grid,
                               /*block_factors=*/emptyAttr,
                               /*indexing_maps=*/emptyAttr,
                               /*iterator_types=*/emptyAttr,
                               /*threads=*/threadsAttr,
                               /*scratch_inputs=*/nullptr,
                               /*regionsCount=*/1);

    assert(helper.outputTensorMap.size() == 1 &&
           "currently only support one output tensor argument");
    auto resultTensor =
        newFunc.getArgument(helper.outputTensorMap.begin()->first);
    func::ReturnOp::create(rewriter, loc, resultTensor);

    // 4. Populate the generic's region with the old triton body.
    //
    // For each old block argument we construct a replacement value: a 1:1
    // identity mapping reuses the converted function arg directly; a 1:N
    // expansion (e.g. TensorDescType -> memref + i32s + i1) is reassembled
    // via an UnrealizedConversionCastOp back to the original type so that the
    // triton ops inside the generic continue to work unchanged.
    Region &genericRegion = genericOp.getRegion(0);
    Block *genericEntry = rewriter.createBlock(&genericRegion);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(genericEntry);

      SmallVector<Value> argReplacements;
      unsigned convertedIdx = 0;
      unsigned crtInputIndex = 0, crtOutputIndex = 0;
      Block &oldEntry = funcOp.getBody().front();
      for (BlockArgument oldArg : oldEntry.getArguments()) {
        Type oldType = oldArg.getType();
        SmallVector<Type> convertedTypes;
        (void)typeConverter->convertType(oldType, convertedTypes);

        if (convertedTypes.size() == 1 && convertedTypes[0] == oldType) {
          // Identity conversion: use the converted arg directly.
          argReplacements.push_back(newEntry->getArgument(convertedIdx));
        } else {
          // Reconstruct the original type from the expanded converted values.
          SmallVector<Value> convertedVals(
              newEntry->getArguments().begin() + convertedIdx,
              newEntry->getArguments().begin() + convertedIdx +
                  convertedTypes.size());
          // overwrite the converted values from the function arguments with the
          // appropriate layout casts generated by the helper which will be
          // forwarded through the generic
          if (helper.isInputTensorArg(convertedIdx))
            convertedVals[0] = inputArgs[crtInputIndex++];
          else if (helper.isOutputTensorArg(convertedIdx))
            convertedVals[0] = outputArgs[crtOutputIndex++];

          Value materialized = typeConverter->materializeSourceConversion(
              rewriter, loc, oldType, convertedVals);
          assert(materialized && "expected source materialization to succeed");
          argReplacements.push_back(materialized);
        }

        convertedIdx += convertedTypes.size();
      }

      // Move the old triton body into the generic's region and merge the
      // (empty) genericEntry with the old entry block, splicing the arg
      // replacements in for the original block arguments.
      rewriter.inlineRegionBefore(funcOp.getBody(), genericRegion,
                                  genericRegion.end());
      rewriter.mergeBlocks(&oldEntry, genericEntry, argReplacements);
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
    rewriter.eraseOp(op);
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

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
