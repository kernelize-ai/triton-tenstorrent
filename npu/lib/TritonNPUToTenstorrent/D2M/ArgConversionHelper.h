#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_D2M_ARGCONVERSIONHELPER_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_D2M_ARGCONVERSIONHELPER_H

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "npu/include/Analysis/Utility.h"

namespace mlir {
namespace triton {
namespace npu {
namespace experimental {

// Shared by the triton-func-to-d2m.generic and triton-func-to-ttnn.generic
// conversion patterns: converts a triton::FuncOp's arguments into the
// signature expected by the tenstorrent lowering (tensors get a TTNN
// layout, tensor descriptors get expanded into memref + shape/stride/padding
// values, scalars pass through the type converter), and builds the new
// func.func with that signature.
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
                                         mlir::tt::ttcore::GridAttr grid,
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
      auto layoutCast = mlir::tt::ttir::TTNNMetalLayoutCastOp::create(
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
      auto layoutCast = mlir::tt::ttir::TTNNMetalLayoutCastOp::create(
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

  static mlir::tt::ttnn::TTNNLayoutAttr
  getTTNNLayoutForMemRef(MemRefType perCoreMemRef,
                         ArrayRef<int64_t> scalarShape,
                         ArrayRef<int64_t> gridShape) {
    namespace ttcore = mlir::tt::ttcore;
    namespace ttnn = mlir::tt::ttnn;
    MLIRContext *context = perCoreMemRef.getContext();
    auto memSpaceAttr =
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceDRAM);
    // TODO: is there ever a case where this would be L1?
    ttnn::BufferType bufferType =
        memSpaceAttr.getValue() == ttcore::MemorySpace::DeviceL1
            ? ttnn::BufferType::L1
            : ttnn::BufferType::DRAM;

    return ttnn::TTNNLayoutAttr::Builder(context, scalarShape,
                                         perCoreMemRef.getElementType())
        .setBufferType(bufferType)
        .setMemoryLayout(
            ttnn::TensorMemoryLayout::Interleaved) // support sharded?
        .build();
  }
};

template <typename Op>
static Op findLoadStoreOpForTensorArg(BlockArgument arg,
                                      triton::FuncOp funcOp) {
  Op ret;
  funcOp.walk([&](Op op) {
    BlockArgument funcArg =
        traceToBlock(op.getPtr(), &funcOp.getBody().front());
    if (funcArg && funcArg == arg) {
      // TODO: do we care if there are multiple loads for the same ptr?
      // probably...
      ret = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ret;
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir

#endif
