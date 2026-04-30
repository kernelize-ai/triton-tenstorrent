#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

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
      return rewriter.notifyMatchFailure(
          funcOp, "non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();
    auto typeConverter = getTypeConverter();

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    Block &oldEntry = funcOp.getBody().front();

    SmallVector<Type> convertedArgTypes;
    if (failed(typeConverter->convertTypes(tritonTy.getInputs(),
                                           convertedArgTypes)))
      return rewriter.notifyMatchFailure(funcOp, "failed to convert arg types");

    // add block start/block end args
    convertedArgTypes.push_back(rewriter.getI32Type()); // block start
    convertedArgTypes.push_back(rewriter.getI32Type()); // block end

    auto newFuncType =
        rewriter.getFunctionType(convertedArgTypes, /*results=*/{});
    auto newFunc =
        rewriter.create<func::FuncOp>(loc, funcOp.getName(), newFuncType);
    ttmlir::utils::setFunctionType(newFunc,
                                   ttmlir::utils::FunctionType::ForwardDevice);

    // Create the function entry block with converted arg types as block args.
    Region &newRegion = newFunc.getBody();
    SmallVector<Location> argLocs(convertedArgTypes.size(), loc);
    Block *newEntry = rewriter.createBlock(&newRegion, newRegion.end(),
                                           convertedArgTypes, argLocs);
    SmallVector<Value> newFuncArgs(newEntry->args_begin(),
                                   newEntry->args_end());

    SmallVector<Value> tensorArgs =
        llvm::to_vector(llvm::make_filter_range(newFuncArgs, [](Value v) {
          if (auto memRefType = dyn_cast<MemRefType>(v.getType())) {
            return isa<tt::ttcore::TileType>(memRefType.getElementType());
          }
          return false;
        }));

    rewriter.setInsertionPointToStart(newEntry);

    // Build the GenericOp in explicit data-movement form (all three of
    // indexing_maps / block_factors / iterator_types are empty).  All
    // converted function arguments are forwarded as additionalArgs so that
    // the triton ops inside can capture them.  They will be lowered in later
    // passes.
    auto threadsAttr = rewriter.getArrayAttr(
        rewriter.getAttr<d2m::ThreadAttr>(d2m::ThreadType::Unified));
    // TODO: populate correct grid size
    auto grid = ttcore::GridAttr::get(context, {1, 1});
    auto emptyAttr = rewriter.getArrayAttr({});

    auto genericOp = rewriter.create<d2m::GenericOp>(
        loc,
        /*results=*/TypeRange{},
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{tensorArgs[0]}, // TODO: d2m.generic verifier
                                               // requires one output
        /*additionalArgs=*/newFuncArgs,
        /*grid=*/grid,
        /*block_factors=*/emptyAttr,
        /*indexing_maps=*/emptyAttr,
        /*iterator_types=*/emptyAttr,
        /*threads=*/threadsAttr,
        /*scratch_inputs=*/nullptr,
        /*regionsCount=*/1);

    // The kernel returns void.
    rewriter.create<func::ReturnOp>(loc);

    // Populate the generic's region with the old triton body.
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
      for (BlockArgument oldArg : oldEntry.getArguments()) {
        Type oldType = oldArg.getType();
        SmallVector<Type> convertedTypes;
        (void)typeConverter->convertType(oldType, convertedTypes);

        SmallVector<Value> convertedVals(newFuncArgs.begin() + convertedIdx,
                                         newFuncArgs.begin() + convertedIdx +
                                             convertedTypes.size());
        convertedIdx += convertedTypes.size();

        if (convertedTypes.size() == 1 && convertedTypes[0] == oldType) {
          // Identity conversion: use the converted arg directly.
          argReplacements.push_back(convertedVals[0]);
        } else {
          // Reconstruct the original type from the expanded converted values.
          Value materialized = typeConverter->materializeSourceConversion(
              rewriter, loc, oldType, convertedVals);
          assert(materialized && "expected source materialization to succeed");
          argReplacements.push_back(materialized);
        }
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
