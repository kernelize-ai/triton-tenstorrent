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
    DenseMap<unsigned, Type>
        tensorArgsMap; // maps new argument index to the original triton type
    // TODO: can probably drop enumate?
    for (auto [i, argType] : llvm::enumerate(tritonTy.getInputs())) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(argType)) {
        if (isa<PointerType>(tensorTy.getElementType())) {
          // TODO: need to handle the pointer path... don't forget
          assert(false && "pointer types not yet supported");
        }
      }
      if (auto tensorDescTy = dyn_cast<triton::TensorDescType>(argType)) {
#if 1
        auto blockTensorTy = tensorDescTy.getBlockType();
        auto makeDynamicTensorTy = [](RankedTensorType tensorTy,
                                      Attribute encoding) {
          // TODO: do we need a layout here?
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
        llvm::errs() << "orig memref type: " << expandedTypes.front() << "\n";
        auto perCoreMemRef = cast<MemRefType>(expandedTypes.front());
#if 0
        auto memSpaceAttr =
    cast<ttcore::MemorySpaceAttr>(perCoreMemRef.getMemorySpace());
    auto metalLayout = ttcore::MetalLayoutAttr::get(
    context,
    /*logicalShape=*/blockTensorTy.getShape(),
    /*memorySpace=*/memSpaceAttr.getValue(),
    /*memoryLayout=*/ttcore::TensorMemoryLayout::Sharded);
    auto ttnnLayout = metalLayout; // TODO: remove
#else
        auto memSpaceAttr =
            cast<ttcore::MemorySpaceAttr>(perCoreMemRef.getMemorySpace());
        // TODO: is there ever a case where this would be L1?
        ttnn::BufferType bufferType =
            memSpaceAttr.getValue() == ttcore::MemorySpace::DeviceL1
                ? ttnn::BufferType::L1
                : ttnn::BufferType::DRAM;

        auto ttnnLayout =
            ttnn::TTNNLayoutAttr::Builder(context, blockTensorTy.getShape(),
                                          blockTensorTy.getElementType())
                .setBufferType(bufferType)
                .setMemoryLayout(
                    ttnn::TensorMemoryLayout::Interleaved) // or Sharded?
                .build();

#if 0
        auto linearMap = AffineMap::getMultiDimIdentityMap(perCoreMemRef.getRank(), context);
        auto ttnnLayout = ttnn::TTNNLayoutAttr::get(context, linearMap, grid.getShape(), perCoreMemRef, /*mem_layout=*/ttnn::TensorMemoryLayoutAttr{},
    /*tensor_mesh=*/ttcore::TensorMeshAttr{},
    /*ignorePhysicalLayout=*/false,
    /*core_range_set=*/ttnn::CoreRangeSetAttr{});
#endif
#endif
        convertedArgTypes.push_back(
            makeDynamicTensorTy(blockTensorTy, ttnnLayout));

        convertedArgTypes.append(expandedTypes.begin() + 1,
                                 expandedTypes.end());
        continue;
#else
        SmallVector<Type> expandedTypes;
        if (failed(typeConverter->convertType(argType, expandedTypes))) {
          return rewriter.notifyMatchFailure(
              funcOp, "failed to convert tensor desc arg type");
        }
        llvm::errs() << "tensor desc ty = " << tensorDescTy << "\n";
        llvm::errs() << "expanded types front: " << expandedTypes.front()
                     << "\n";
        assert(false && "TODO");
        convertedArgTypes.push_back(argType);
        continue;
#endif
      }
      auto convertedType = typeConverter->convertType(argType);
      convertedArgTypes.push_back(convertedType);
    }

    // add block start/block end args
    convertedArgTypes.push_back(rewriter.getI32Type()); // block start
    convertedArgTypes.push_back(rewriter.getI32Type()); // block end

    for (auto type : convertedArgTypes) {
      llvm::errs() << "converted arg type: " << type << "\n";
    }

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

    rewriter.setInsertionPointToStart(newEntry);
#if 1
    SmallVector<Value> newFuncArgs(
        newEntry->getNumArguments()); // for remapping
    SmallVector<Value> tensorArgs, scalarArgs;
    for (auto [i, arg] : llvm::enumerate(newEntry->getArguments())) {
      auto it = tensorArgsMap.find(i);
      if (it != tensorArgsMap.end()) {
        assert(isa<RankedTensorType>(arg.getType()) &&
               "expected converted tensor arg to be ranked tensor (i.e. not a "
               "memref)");

        // TODO: we found a tensor of points or tensor desc type - need to
        // convert the original type again so we can construct the expected
        // d2m.generic input type, and run that through a d2m.view op

        SmallVector<Type> outputTypes;
        if (failed(typeConverter->convertType(it->second, outputTypes))) {
          return rewriter.notifyMatchFailure(
              funcOp,
              "failed to convert back original tensor desc type for d2m.view");
        }
        assert(!outputTypes.empty() && isa<MemRefType>(outputTypes.front()) &&
               "expected at least one output type and for the first output "
               "type to be a memref");

        llvm::errs() << "create view with memref type: " << outputTypes.front()
                     << "\n";

        // TODO: replace this with ttir metal layout cast op (see sanity.mlir)
#if 1
        auto layoutCast = ttir::TTNNMetalLayoutCastOp::create(
            rewriter, arg.getLoc(), outputTypes[0], arg);
        Value result = layoutCast.getResult();
#else
        AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
            cast<RankedTensorType>(arg.getType()).getRank(), arg.getContext());
        auto viewOp = d2m::ViewLayoutOp::create(
            rewriter, arg.getLoc(), outputTypes[0], arg, identityMap,
            /*reinterpretLayout=*/false);
        Value result = viewOp.getResult();
#endif
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
#else
    SmallVector<Value> newFuncArgs(newEntry->args_begin(),
                                   newEntry->args_end());

    auto isTileMemref = [](Value v) {
      auto memRefType = dyn_cast<MemRefType>(v.getType());
      return memRefType &&
             isa<tt::ttcore::TileType>(memRefType.getElementType());
    };
    SmallVector<Value> tensorArgs, scalarArgs;
    std::partition_copy(newFuncArgs.begin(), newFuncArgs.end(),
                        std::back_inserter(tensorArgs),
                        std::back_inserter(scalarArgs), isTileMemref);
#endif
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

    // assert(false && "TODO");
    // SmallVector<Value> newFuncArgs; // TODO: broken
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
