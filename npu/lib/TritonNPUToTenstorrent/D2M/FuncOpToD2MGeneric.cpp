#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/StrUtil.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ArgConversionHelper {
  // map the converted arg index to the original memref type for tensor
  // arguments so that we can create the appropriate layout cast to pass the
  // function argument to the d2m.genericop
  DenseMap<unsigned, MemRefType> inputTensorMap;
  DenseMap<unsigned, MemRefType> outputTensorMap;
  SmallVector<Type> convertedArgTypes;
  SmallVector<Location> argLocs;

  ArgConversionHelper() = default;

  LogicalResult convertFunctionArguments(triton::FuncOp funcOp,
                                         ConversionPatternRewriter &rewriter,
                                         const TypeConverter *typeConverter);

  func::FuncOp generateNewFunction(triton::FuncOp origFunc,
                                   ConversionPatternRewriter &rewriter);

  SmallVector<Value> getScalarArgs(func::FuncOp newFunc) const {
    auto filtered = llvm::make_filter_range(
        llvm::enumerate(newFunc.getArguments()), [&](auto indexedArg) {
          auto [index, arg] = indexedArg;
          return inputTensorMap.count(index) == 0 &&
                 outputTensorMap.count(index) == 0;
        });
    return llvm::to_vector(llvm::map_range(
        filtered, [](auto indexedArg) -> Value { return indexedArg.value(); }));
  }

  SmallVector<Value>
  generateInputArgs(func::FuncOp newFunc,
                    ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> inputArgs;
    for (auto [index, memrefType] : inputTensorMap) {
      auto arg = newFunc.getArgument(index);
      auto layoutCast = ttir::TTNNMetalLayoutCastOp::create(
          rewriter, arg.getLoc(), memrefType, arg);
      inputArgs.push_back(layoutCast.getResult());
    }
    return inputArgs;
  }

  SmallVector<Value>
  generateOutputArgs(func::FuncOp newFunc,
                     ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> outputArgs;
    for (auto [index, memrefType] : outputTensorMap) {
      auto arg = newFunc.getArgument(index);
      auto layoutCast = ttir::TTNNMetalLayoutCastOp::create(
          rewriter, arg.getLoc(), memrefType, arg);
      outputArgs.push_back(layoutCast.getResult());
    }
    return outputArgs;
  }

  bool isInputTensorArg(unsigned index) const {
    return inputTensorMap.count(index) > 0;
  }
  bool isOutputTensorArg(unsigned index) const {
    return outputTensorMap.count(index) > 0;
  }
};

LogicalResult ArgConversionHelper::convertFunctionArguments(
    triton::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    const TypeConverter *typeConverter) {
  MLIRContext *context = rewriter.getContext();

  auto makeDynamicTensorTy = [](RankedTensorType tensorTy, Attribute encoding) {
    SmallVector<int64_t> dynShape(tensorTy.getRank(), ShapedType::kDynamic);
    return RankedTensorType::get(dynShape, tensorTy.getElementType(), encoding);
  };

  Block &oldEntry = funcOp.getBody().front();
  for (auto [idx, oldArg] : llvm::enumerate(oldEntry.getArguments())) {
    Type argType = oldArg.getType();
    if (auto tensorTy = dyn_cast<RankedTensorType>(argType)) {
      if (isa<PointerType>(tensorTy.getElementType())) {
        return emitError(oldArg.getLoc(), "pointer types not yet supported");
      }
    }
    if (auto tensorDescTy = dyn_cast<triton::TensorDescType>(argType)) {
      auto blockTensorTy = tensorDescTy.getBlockType();

      SmallVector<Type> expandedTypes;
      if (failed(typeConverter->convertType(argType, expandedTypes))) {
        return emitError(oldArg.getLoc(),
                         "failed to convert tensor desc arg type");
      }
      assert(!expandedTypes.empty() && isa<MemRefType>(expandedTypes.front()) &&
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
          ttnn::TTNNLayoutAttr::Builder(rewriter.getContext(),
                                        blockTensorTy.getShape(),
                                        perCoreMemRef.getElementType())
              .setBufferType(bufferType)
              .setMemoryLayout(
                  ttnn::TensorMemoryLayout::Interleaved) // or Sharded?
              .build();

      // TODO: read attribute from funcOp, put in appropraite input/output list
      // TODO: define attribute name in dialect
      auto ioTypeAttr = dyn_cast_or_null<tt::IOTypeAttr>(
          funcOp.getArgAttr(idx, "triton_tenstorrent.io_type"));
      if (!ioTypeAttr) {
        return rewriter.notifyMatchFailure(
            funcOp, "missing IOType attribute on tensor argument");
      }
      if (ioTypeAttr.getValue() == tt::IOType::INPUT) {
        inputTensorMap.insert({convertedArgTypes.size(), perCoreMemRef});
      } else if (ioTypeAttr.getValue() == tt::IOType::OUTPUT) {
        outputTensorMap.insert({convertedArgTypes.size(), perCoreMemRef});
      } else {
        return rewriter.notifyMatchFailure(
            funcOp, "unexpected IOType value on tensor argument");
      }

      convertedArgTypes.push_back(
          makeDynamicTensorTy(blockTensorTy, ttnnLayout));
      argLocs.append(expandedTypes.size(), oldArg.getLoc());

      convertedArgTypes.append(expandedTypes.begin() + 1, expandedTypes.end());
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

  if (outputTensorMap.size() != 1) {
    return emitError(funcOp.getLoc(),
                     "currently only support one output tensor argument");
  }

  return success();
}

func::FuncOp
ArgConversionHelper::generateNewFunction(triton::FuncOp origFunc,
                                         ConversionPatternRewriter &rewriter) {
  assert(outputTensorMap.size() == 1 &&
         "currently only support one output tensor argument");
  // the new function returns a tensor, not a memref
  Type returnType = convertedArgTypes[outputTensorMap.begin()->first];

  auto newFuncType =
      rewriter.getFunctionType(convertedArgTypes, /*results=*/{returnType});

  LDBG("Converting function " << origFunc.getFunctionType()
                              << " to new function type: " << newFuncType);

  auto newFunc = func::FuncOp::create(rewriter, origFunc.getLoc(),
                                      origFunc.getName(), newFuncType);
  ttmlir::utils::setFunctionType(newFunc,
                                 ttmlir::utils::FunctionType::ForwardDevice);

  Region &newRegion = newFunc.getBody();
  Block *newEntry = rewriter.createBlock(&newRegion, newRegion.end(),
                                         convertedArgTypes, argLocs);
  return newFunc;
}

struct ConvertTritonFunc : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  // Convert function arguments - for tensor arguments, we convert to dynamic
  // shape tensors with ttnn layouts. For tensor descriptor arguments, we
  // convert the tensor descriptor to dynamic shape tensors with ttnn layouts
  // and add the expanded values to the function signature to match the triton
  // runtime. For scalar args, just use the type converter.
  LogicalResult convertTritonFunctionArgs(
      Block &oldEntry, SmallVector<Type> &convertedArgTypes,
      SmallVector<Location> &argLocs, Type &returnType,
      DenseMap<unsigned, Type> &tensorArgsMap,
      DenseMap<unsigned, MemRefType> &tensorIndexToMemrefType,
      ConversionPatternRewriter &rewriter) const {
    auto makeDynamicTensorTy = [](RankedTensorType tensorTy,
                                  Attribute encoding) {
      SmallVector<int64_t> dynShape(tensorTy.getRank(), ShapedType::kDynamic);
      return RankedTensorType::get(dynShape, tensorTy.getElementType(),
                                   encoding);
    };
    MLIRContext *context = rewriter.getContext();

    for (auto oldArg : oldEntry.getArguments()) {
      Type argType = oldArg.getType();
      if (auto tensorTy = dyn_cast<RankedTensorType>(argType)) {
        if (isa<PointerType>(tensorTy.getElementType())) {
          return emitError(oldArg.getLoc(), "pointer types not yet supported");
        }
      }
      if (auto tensorDescTy = dyn_cast<triton::TensorDescType>(argType)) {
        auto blockTensorTy = tensorDescTy.getBlockType();

        SmallVector<Type> expandedTypes;
        if (failed(typeConverter->convertType(argType, expandedTypes))) {
          return emitError(oldArg.getLoc(),
                           "failed to convert tensor desc arg type");
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

        tensorIndexToMemrefType.insert(
            {convertedArgTypes.size(), perCoreMemRef});
        convertedArgTypes.push_back(
            makeDynamicTensorTy(blockTensorTy, ttnnLayout));
        argLocs.append(expandedTypes.size(), oldArg.getLoc());

        if (tensorArgsMap.size() == 3)
          returnType = convertedArgTypes.back();

        convertedArgTypes.append(expandedTypes.begin() + 1,
                                 expandedTypes.end());
        continue;
      }
      auto convertedType = typeConverter->convertType(argType);
      convertedArgTypes.push_back(convertedType);
      argLocs.push_back(oldArg.getLoc());
    }
    return success();
  }

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
    DenseMap<unsigned, MemRefType> tensorIndexToMemrefType; // delete
    Type returnType;                                        // delete

    ArgConversionHelper helper;
    // 1. Convert function arguments and add tenstorrent specific args (block
    // start/end)
#if 1
    if (failed(helper.convertFunctionArguments(funcOp, rewriter,
                                               getTypeConverter()))) {
      return rewriter.notifyMatchFailure(
          funcOp, "failed to convert function arguments");
    }
#else

    Type returnType;
    // maps converted args to the original tensor type for creating the cast
    // from function argument (which is a tensor type) to d2m generic (which is
    // a memref type).
    DenseMap<unsigned, MemRefType> tensorIndexToMemrefType;
    if (failed(convertTritonFunctionArgs(oldEntry, convertedArgTypes, argLocs,
                                         returnType, tensorArgsMap,
                                         tensorIndexToMemrefType, rewriter))) {
      return rewriter.notifyMatchFailure(
          funcOp, "failed to convert function arguments");
    }
#endif

    // LDBG("Converted triton function "
    //  << tritonTy " arguments to Tenstorrent compatible function signature: "
    //  << convertedArgTypes);
#if 1
    func::FuncOp newFunc = helper.generateNewFunction(funcOp, rewriter);
    Block *newEntry = &newFunc.getBody().front();
#else
    assert(false && "TODO");

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

    // 2. Create the new function with converted signature and a single entry
    // block.
    auto newFuncType =
        rewriter.getFunctionType(convertedArgTypes, /*results=*/{returnType});
    auto newFunc =
        rewriter.create<func::FuncOp>(loc, funcOp.getName(), newFuncType);
    ttmlir::utils::setFunctionType(newFunc,
                                   ttmlir::utils::FunctionType::ForwardDevice);

    Region &newRegion = newFunc.getBody();
    Block *newEntry = rewriter.createBlock(&newRegion, newRegion.end(),
                                           convertedArgTypes, argLocs);
#endif
    // 3. populate the new function body with a d2m.generic op
    rewriter.setInsertionPointToStart(newEntry);
    // SmallVector<Value> newFuncArgs(
    //     newEntry->getNumArguments()); // for remapping
    SmallVector<Value> tensorArgs, scalarArgs;
    SmallVector<unsigned> tensorArgIndices;
#if 1
#else
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
        tensorArgIndices.push_back(i);
        continue;
      }
      assert(!(isa<RankedTensorType>(arg.getType()) ||
               isa<MemRefType>(arg.getType()) ||
               isa<TensorDescType>(arg.getType())) &&
             "expected non-tensor arg to not be ranked tensor");
      scalarArgs.push_back(arg);
      newFuncArgs[i] = arg;
    }
#endif
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
