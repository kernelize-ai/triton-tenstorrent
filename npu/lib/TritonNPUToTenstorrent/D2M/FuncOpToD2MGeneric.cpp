#include "PatternTritonNPUToD2M.h"

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

    // TODO: populate correct grid size
    auto grid = ttcore::GridAttr::get(context, {1, 1});

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    Block &oldEntry = funcOp.getBody().front();

    SmallVector<Type> convertedArgTypes;
    SmallVector<Location> argLocs;
    DenseMap<unsigned, Type>
        tensorArgsMap; // maps new argument index to the original triton type

    for (auto [argType, oldArg] :
         llvm::zip(tritonTy.getInputs(), oldEntry.getArguments())) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(argType)) {
        if (isa<PointerType>(tensorTy.getElementType())) {
          // TODO: need to handle the pointer path... don't forget
          assert(false && "pointer types not yet supported");
        }
      }
      if (auto tensorDescTy = dyn_cast<triton::TensorDescType>(argType)) {
        auto blockTensorTy = tensorDescTy.getBlockType();
        auto makeDynamicTensorTy = [](RankedTensorType tensorTy,
                                      Attribute encoding) {
          SmallVector<int64_t> dynShape(tensorTy.getRank(),
                                        ShapedType::kDynamic);
          return RankedTensorType::get(dynShape, tensorTy.getElementType(),
                                       encoding);
        };
        tensorArgsMap[convertedArgTypes.size()] = tensorDescTy;

        SmallVector<Type> expandedTypes;
        if (failed(typeConverter->convertType(argType, expandedTypes))) {
          return rewriter.notifyMatchFailure(
              funcOp, "failed to convert tensor desc arg type");
        }
        assert(!expandedTypes.empty() &&
               isa<MemRefType>(expandedTypes.front()) &&
               "expected first expanded tensor desc type to be memref");
        // drop the memref, but populate the rest of the expanded args
        auto perCoreMemRef = cast<MemRefType>(expandedTypes.front());
        auto memSpaceAttr =
            cast<ttcore::MemorySpaceAttr>(perCoreMemRef.getMemorySpace());
        // TODO: is there ever a case where this would be L1?
        ttnn::BufferType bufferType =
            memSpaceAttr.getValue() == ttcore::MemorySpace::DeviceL1
                ? ttnn::BufferType::L1
                : ttnn::BufferType::DRAM;

        auto ttnnLayout =
            ttnn::TTNNLayoutAttr::Builder(context, blockTensorTy.getShape(),
                                          perCoreMemRef.getElementType())
                .setBufferType(bufferType)
                .setMemoryLayout(
                    ttnn::TensorMemoryLayout::Interleaved) // or Sharded?
                .build();

        convertedArgTypes.push_back(
            makeDynamicTensorTy(blockTensorTy, ttnnLayout));
        argLocs.append(expandedTypes.size(), oldArg.getLoc());

        convertedArgTypes.append(expandedTypes.begin() + 1,
                                 expandedTypes.end());
        continue;
      }
      auto convertedType = typeConverter->convertType(argType);
      convertedArgTypes.push_back(convertedType);
      argLocs.push_back(oldArg.getLoc());
    }

    // Add block start/block end arguments with appropriate NameLocs:
    // %block_start, %block_end.
    convertedArgTypes.push_back(rewriter.getI32Type()); // block start
    auto blockStartLoc =
        NameLoc::get(StringAttr::get(context, "block_start"),
                     FileLineColLoc::get(context, __FILE__, __LINE__, 0));
    argLocs.push_back(blockStartLoc);
    auto blockEndLoc =
        NameLoc::get(StringAttr::get(context, "block_end"),
                     FileLineColLoc::get(context, __FILE__, __LINE__, 0));
    convertedArgTypes.push_back(rewriter.getI32Type()); // block end
    argLocs.push_back(blockEndLoc);

    auto newFuncType =
        rewriter.getFunctionType(convertedArgTypes, /*results=*/{});
    auto newFunc =
        rewriter.create<func::FuncOp>(loc, funcOp.getName(), newFuncType);
    ttmlir::utils::setFunctionType(newFunc,
                                   ttmlir::utils::FunctionType::ForwardDevice);

    // Create the function entry block with converted arg types as block args.
    Region &newRegion = newFunc.getBody();
    Block *newEntry = rewriter.createBlock(&newRegion, newRegion.end(),
                                           convertedArgTypes, argLocs);

    rewriter.setInsertionPointToStart(newEntry);
    SmallVector<Value> newFuncArgs(
        newEntry->getNumArguments()); // for remapping
    SmallVector<Value> tensorArgs, scalarArgs;
    for (auto [i, arg] : llvm::enumerate(newEntry->getArguments())) {
      auto it = tensorArgsMap.find(i);
      if (it != tensorArgsMap.end()) {
        assert(isa<RankedTensorType>(arg.getType()) &&
               "expected converted tensor arg to be ranked tensor (i.e. not a "
               "memref)");

        SmallVector<Type> outputTypes;
        if (failed(typeConverter->convertType(it->second, outputTypes))) {
          return rewriter.notifyMatchFailure(
              funcOp,
              "failed to convert back original tensor desc type for d2m.view");
        }
        assert(!outputTypes.empty() && isa<MemRefType>(outputTypes.front()) &&
               "expected at least one output type and for the first output "
               "type to be a memref");

        auto layoutCast = ttir::TTNNMetalLayoutCastOp::create(
            rewriter, arg.getLoc(), outputTypes[0], arg);
        Value result = layoutCast.getResult();
        tensorArgs.push_back(result);
        newFuncArgs[i] = result;
        continue;
      }
      assert(!(isa<RankedTensorType>(arg.getType()) ||
               isa<MemRefType>(arg.getType()) ||
               isa<TensorDescType>(arg.getType())) &&
             "expected non-tensor arg to not be ranked tensor");
      scalarArgs.push_back(arg);
      newFuncArgs[i] = arg;
    }

    // Build the GenericOp in explicit data-movement form (all three of
    // indexing_maps / block_factors / iterator_types are empty).
    auto threadsAttr = rewriter.getArrayAttr(
        rewriter.getAttr<d2m::ThreadAttr>(d2m::ThreadType::Unified));
    auto emptyAttr = rewriter.getArrayAttr({});

    auto genericOp = rewriter.create<d2m::GenericOp>(
        loc,
        /*results=*/TypeRange{},
        /*inputs=*/tensorArgs,
        /*outputs=*/ValueRange{tensorArgs[0]}, // TODO: d2m.generic verifier
                                               // requires one output
        /*additionalArgs=*/scalarArgs,
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
