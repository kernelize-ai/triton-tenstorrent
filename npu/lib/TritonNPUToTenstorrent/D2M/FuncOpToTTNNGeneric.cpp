// Pass: --convert-triton-npu-to-ttnn-generic
//
// Rewrites a Triton NPU kernel function directly into a func.func containing
// a single `ttnn.generic` op, bypassing the D2M/TTKernel/EmitC/D2MToTTNN
// pipeline entirely. The generic's kernel descriptors carry an inline
// `source` string (via `#ttnn.source_read_kernel` /
// `#ttnn.source_write_kernel` / `#ttnn.source_compute_kernel`) instead of a
// `symbol_ref` to a kernel func.func -- no kernel bodies are created. This
// establishes the shape of the final `ttnn.generic` op ahead of real kernel
// codegen filling in the `source` strings.

#include "PatternTritonNPUToD2M.h"

#include "ArgConversionHelper.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "../TypeConverter.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOTTNNGENERIC
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

namespace {

// Builds a single-rectangle core range set spanning the whole grid, i.e. the
// same fallback shape `D2MToTTNN.cpp`'s `coreRangeSetFromGeneric` uses when a
// generic has no virt-to-physical map (grid built via
// `ttcore::GridAttr::get(ctx, shape)` never carries one).
ttnn::CoreRangeSetAttr fullGridCoreRangeSet(MLIRContext *ctx,
                                            ttcore::GridAttr grid) {
  ArrayRef<int64_t> gridShape = grid.getShape();
  return ttnn::CoreRangeSetAttr::get(
      ctx,
      ttnn::CoreRangeAttr::get(
          ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
          ttnn::CoreCoordAttr::get(ctx, gridShape[1] - 1, gridShape[0] - 1)));
}

// Builds a double-buffered CB for one tensor input/output: sized to hold
// 2x the number of 32x32 tiles the kernel's single load/store on that
// tensor moves (`tensorMemRef`'s shape is already in tile units -- see
// ArgConversionHelper::convertFunctionArguments), so one buffer's worth of
// tiles can be in flight while the other is being produced/consumed.
ttnn::KernelCBAttr doubleBufferedCB(MLIRContext *ctx, MemRefType tensorMemRef,
                                    int64_t numStages,
                                    ttnn::CoreRangeSetAttr coreRanges,
                                    uint32_t cbIndex) {
  auto tile = cast<ttcore::TileType>(tensorMemRef.getElementType());
  int64_t numTiles = 1;
  for (int64_t dim : tensorMemRef.getShape())
    numTiles *= dim;
  uint32_t pageSize = static_cast<uint32_t>(tile.getSizeBytes());
  uint32_t totalSize = pageSize * static_cast<uint32_t>(numStages * numTiles);
  auto format =
      ttnn::KernelCBFormatAttr::get(ctx, cbIndex, tile.getDataType(), pageSize);
  return ttnn::KernelCBAttr::get(
      ctx, totalSize, coreRanges, {format},
      /*buffer=*/ttnn::KernelCBGlobalBufferAddressOfTensorAttr());
}

// TODO: derive one CoreRuntimeArgsAttr per physical core in `grid`, assigning
// each a linear work-item id (see ttnn_generic_kernel_contract.md's rt_args
// section: tile_start/tile_end baked per physical core). Not implemented
// yet -- this pass only establishes the ttnn.generic's shape, so no per-core
// rt_args are emitted.
SmallVector<mlir::tt::ttnn::CoreRuntimeArgsAttr>
perCoreWorkRangeArgs(MLIRContext *ctx, ttcore::GridAttr grid) {
  return {};
}

// Builds the #ttnn.program's three kernel descriptors, each with an inline
// `source` string that includes the kernel header file.
SmallVector<mlir::Attribute>
kernelDescriptors(MLIRContext *ctx, const std::string &kernelName,
                  ttnn::CoreRangeSetAttr coreRanges,
                  llvm::ArrayRef<mlir::Attribute> commonRtArgs,
                  llvm::ArrayRef<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs,
                  llvm::ArrayRef<mlir::Attribute> ctArgs) {
  auto readerSource = mlir::StringAttr::get(
      ctx, "#define READER_KERNEL\n#include \"" + kernelName + ".h\"");
  auto readKernel = ttnn::SourceReadKernelAttr::get(
      ctx, readerSource, coreRanges, commonRtArgs, rtArgs, ctArgs);

  auto writerSource = mlir::StringAttr::get(
      ctx, "#define WRITER_KERNEL\n#include \"" + kernelName + ".h\"");
  auto writeKernel = ttnn::SourceWriteKernelAttr::get(
      ctx, writerSource, coreRanges, commonRtArgs, rtArgs, ctArgs);
  auto computeSource = mlir::StringAttr::get(
      ctx, "#define COMPUTE_KERNEL\n#include \"" + kernelName + ".h\"");
  auto computeKernel = ttnn::SourceComputeKernelAttr::get(
      ctx, computeSource, coreRanges, ttnn::ComputeKernelMathFidelity::HiFi4,
      /*fp32DestAccEn=*/false, /*dstFullSyncEn=*/false,
      /*unpackToDestModes=*/{}, /*bfp8PackPrecise=*/false,
      /*mathApproxMode=*/false, commonRtArgs, rtArgs, ctArgs);
  return {readKernel, writeKernel, computeKernel};
}

struct ConvertTritonFuncToTTNNGeneric
    : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isKernel(funcOp)) {
      return funcOp.emitError("non-kernel functions are not yet supported");
    }

    Location loc = funcOp.getLoc();
    MLIRContext *context = funcOp.getContext();
    std::string kernelName = funcOp.getName().str();

    ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    auto grid = ttcore::GridAttr::get(context, gridAttr.getShape());

    mlir::FunctionType tritonTy = funcOp.getFunctionType();
    assert(tritonTy.getResults().empty() &&
           "expected triton kernel to return void");

    experimental::ArgConversionHelper helper;
    if (failed(helper.convertFunctionArguments(funcOp, rewriter, grid,
                                               getTypeConverter()))) {
      return funcOp.emitError("failed to convert function arguments");
    }
    if (helper.outputTensorMap.size() != 1) {
      return funcOp.emitError(
          "currently only support one output tensor argument");
    }

    func::FuncOp newFunc = helper.generateNewFunction(funcOp, rewriter);
    Block *newEntry = &newFunc.getBody().front();
    rewriter.setInsertionPointToStart(newEntry);

    llvm::SmallVector<mlir::Attribute> commonRtArgs;
    llvm::SmallVector<mlir::tt::ttnn::CoreRuntimeArgsAttr> rtArgs;
    llvm::SmallVector<mlir::Attribute> ctArgs;

    ttnn::CoreRangeSetAttr coreRanges = fullGridCoreRangeSet(context, grid);
    rtArgs = perCoreWorkRangeArgs(context, grid);

    // ttnn.generic's io tensors are the function's tensor arguments
    // themselves (already carrying a TTNN layout) -- unlike d2m.generic,
    // there is no memref layout cast: inputs first, then the output. Each
    // tensor also gets a double-buffered CB (see doubleBufferedCB).
    //
    // commonRtArgs indices are positions in the runtime's combined
    // io_tensors-then-additionalArgs operand list (see generic_op.cpp's
    // `run()`, which builds `argRefs` as io_tensors followed by
    // additional_args): both KernelArgAddressOfTensorAttr and
    // KernelArgScalarAttr index into that single flat list, not into two
    // separate namespaces. So scalar indices must be offset by the total
    // io tensor count (numIoTensors, i.e. inputs + the one output).
    SmallVector<Value> ios;
    SmallVector<ttnn::KernelCBAttr> cbs;
    size_t tensorIndex = 0;
    // The single output always lands last in `ios` (see the second loop
    // below), regardless of where it falls in the function's raw argument
    // order, so its io_tensors index is fixed at inputTensorMap.size().
    size_t outputTensorIndex = helper.inputTensorMap.size();
    size_t numIoTensors = outputTensorIndex + 1;
    size_t scalarIndex = numIoTensors;
    // Get the number of pipeline stages from the function attribute
    int64_t numStages = 2;
    const char *pipelineStagesAttrName = "tt.num_stages";
    auto pipelineStagesAttr =
        funcOp->getAttrOfType<IntegerAttr>(pipelineStagesAttrName);
    if (pipelineStagesAttr) {
      numStages = pipelineStagesAttr.getInt();
    }
    for (auto [index, arg] : llvm::enumerate(newEntry->getArguments())) {
      if (helper.isInputTensorArg(index)) {
        ios.push_back(arg);
        commonRtArgs.push_back(
            ttnn::KernelArgAddressOfTensorAttr::get(context, tensorIndex));
        cbs.push_back(doubleBufferedCB(
            context, helper.inputTensorMap.lookup(index), numStages, coreRanges,
            static_cast<uint32_t>(tensorIndex)));
        ctArgs.push_back(
            ttnn::KernelArgCBBufferIndexAttr::get(context, cbs.size() - 1));
        tensorIndex++;
      } else if (helper.isOutputTensorArg(index)) {
        commonRtArgs.push_back(ttnn::KernelArgAddressOfTensorAttr::get(
            context, outputTensorIndex));
        cbs.push_back(doubleBufferedCB(
            context, helper.outputTensorMap.lookup(index), numStages,
            coreRanges, static_cast<uint32_t>(outputTensorIndex)));
        ctArgs.push_back(
            ttnn::KernelArgCBBufferIndexAttr::get(context, cbs.size() - 1));
        outputTensorIndex++;
      } else {
        commonRtArgs.push_back(
            ttnn::KernelArgScalarAttr::get(context, scalarIndex++));
      }
    }
    for (auto [index, arg] : llvm::enumerate(newEntry->getArguments())) {
      if (helper.isOutputTensorArg(index)) {
        ios.push_back(arg);
      }
    }

    // One TensorAccessorArgs marker per io tensor follows the CB indices.
    for (size_t i = 0; i < numIoTensors; ++i) {
      ctArgs.push_back(ttnn::KernelArgTensorAccessorArgsAttr::get(context, i));
    }

    // TODO: add semaphores

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        context,
        kernelDescriptors(context, kernelName, coreRanges, commonRtArgs, rtArgs,
                          ctArgs),
        cbs, /*semaphores=*/{});

    ttnn::GenericOp::create(rewriter, loc, ios, helper.getScalarArgs(newFunc),
                            program);

    auto resultTensor =
        newFunc.getArgument(helper.outputTensorMap.begin()->first);
    func::ReturnOp::create(rewriter, loc, resultTensor);

    rewriter.eraseOp(funcOp);
    return success();
  }
};

} // namespace

struct ConvertTritonNPUToTTNNGenericPass
    : public impl::ConvertTritonNPUToTTNNGenericBase<
          ConvertTritonNPUToTTNNGenericPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> gridShape = llvm::to_vector(gridAttr.getShape());

    TritonNPUToTenstorrentTypeConverter typeConverter(context);
    experimental::populateTritonNPUTypeConversions(typeConverter, gridShape);

    mlir::ConversionTarget target(*context);
    target.addIllegalOp<triton::FuncOp>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<mlir::tt::ttnn::TTNNDialect>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertTritonFuncToTTNNGeneric>(typeConverter, context,
                                                 PatternBenefit(1));

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
