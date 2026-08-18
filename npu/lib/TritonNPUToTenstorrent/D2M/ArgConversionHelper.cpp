#include "ArgConversionHelper.h"

#include "llvm/Support/Debug.h"

#include "ttmlir/FunctionTypes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "SPMDArgs.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "triton-npu-arg-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Convert function arguments - for tensor arguments, we convert to dynamic
// shape tensors with ttnn layouts. For tensor descriptor arguments, we
// convert the tensor descriptor to dynamic shape tensors with ttnn layouts
// and add the expanded values to the function signature to match the triton
// runtime. For scalar args, just use the type converter.
LogicalResult ArgConversionHelper::convertFunctionArguments(
    triton::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    ttcore::GridAttr grid, const TypeConverter *typeConverter) {
  MLIRContext *context = rewriter.getContext();

  auto makeDynamicTensorTy = [](RankedTensorType tensorTy, Attribute encoding) {
    SmallVector<int64_t> dynShape(tensorTy.getRank(), ShapedType::kDynamic);
    return RankedTensorType::get(dynShape, tensorTy.getElementType(), encoding);
  };

  Block &oldEntry = funcOp.getBody().front();
  for (auto [idx, oldArg] : llvm::enumerate(oldEntry.getArguments())) {
    Type argType = oldArg.getType();

    if (isa<PointerType>(argType)) {
      auto ioTypeAttr = dyn_cast_or_null<tt::IOTypeAttr>(
          funcOp.getArgAttr(idx, kIOTypeAttrName));
      if (!ioTypeAttr) {
        // TagInputOutputs only tags pointers reachable from an ordinary
        // tt.load/tt.store (or descriptor load/store). A pointer used only
        // by an atomic op (no ordinary load/store on it) is never tagged --
        // treat it like a plain scalar argument instead of erroring; the
        // atomic-op lowering resolves its address independently via
        // PointerInfoAnalysis-style tracing, not through the tensor/memref
        // machinery below.
        auto convertedType = typeConverter->convertType(argType);
        convertedArgTypes.push_back(convertedType);
        argLocs.push_back(oldArg.getLoc());
        continue;
      }

      RankedTensorType tritonType;

      if (ioTypeAttr.getValue() == tt::IOType::INPUT) {
        auto loadOp =
            findLoadStoreOpForTensorArg<triton::LoadOp>(oldArg, funcOp);
        assert(
            loadOp &&
            "expected to find dependent load for INPUT type function argument");
        tritonType = cast<RankedTensorType>(loadOp.getType());
      } else if (ioTypeAttr.getValue() == tt::IOType::OUTPUT) {
        auto storeOp =
            findLoadStoreOpForTensorArg<triton::StoreOp>(oldArg, funcOp);
        assert(storeOp && "expected to find dependent store for OUTPUT type "
                          "function argument");
        // storeOp.getPtr() is the tensor-of-pointers being stored through
        // (element type !tt.ptr<T>); the stored value's own type already
        // has the correct scalar element type T, matching how the INPUT
        // branch above reads it off the load's result rather than its ptr.
        tritonType = cast<RankedTensorType>(storeOp.getValue().getType());
      } else {
        llvm_unreachable("unexpected IOTypeAttr for function argument");
      }
      assert(tritonType &&
             "failed to set tensor block shape information from ptr");

      // use the type converter to get the tiled type from the triton tensor
      auto perCoreMemRef =
          cast<MemRefType>(typeConverter->convertType(tritonType));

      // convert the tiled memref shape to a scalar shape to build the ttnn
      // layout
      SmallVector<int64_t> tiledShape =
          llvm::to_vector(perCoreMemRef.getShape());
      if (tiledShape.size() == 1)
        tiledShape.push_back(
            1); // tenstorrent tiled tensors must be at least rank 2
      auto tile = cast<ttcore::TileType>(perCoreMemRef.getElementType());
      auto scalarShape = tile.getScalarShape(tiledShape);

      auto ttnnLayout =
          getTTNNLayoutForMemRef(perCoreMemRef, scalarShape, grid.getShape());

      // The triton type converter doesn't have access to the grid shape. CB
      // memrefs don't need the grid shape, but function argument ("ptr")
      // memrefs do need the grid shape as remote load/store ops expect grid
      // shapes on the memref arguments. Add the grid shape to the perCoreMemref
      // here
      // TODO: push this into type converter?
      SmallVector<int64_t> argShape(grid.getShape().size(),
                                    1); // interleaved layout requires unit grid
      argShape.append(tiledShape);

      MemRefType functionArgMemRef = MemRefType::get(
          argShape, tile, ttcore::InterleavedLayoutAttr::get(tiledShape, tile),
          ttcore::MemorySpaceAttr::get(context,
                                       ttcore::MemorySpace::DeviceDRAM));

      if (ioTypeAttr.getValue() == tt::IOType::INPUT)
        inputTensorMap.insert({convertedArgTypes.size(), functionArgMemRef});
      else
        outputTensorMap.insert({convertedArgTypes.size(), functionArgMemRef});

      // use the tiled shape in the function arguments so the tensor rank
      // matches the memrefs
      convertedArgTypes.push_back(makeDynamicTensorTy(
          RankedTensorType::get(tiledShape, tritonType.getElementType()),
          ttnnLayout));
      argLocs.push_back(oldArg.getLoc());
      continue;
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
      auto ttnnLayout = getTTNNLayoutForMemRef(
          perCoreMemRef, blockTensorTy.getShape(), grid.getShape());

      auto ioTypeAttr = dyn_cast_or_null<tt::IOTypeAttr>(
          funcOp.getArgAttr(idx, kIOTypeAttrName));
      if (!ioTypeAttr) {
        return funcOp.emitError("missing IOType attribute on tensor argument");
      }
      if (ioTypeAttr.getValue() == tt::IOType::INPUT) {
        inputTensorMap.insert({convertedArgTypes.size(), perCoreMemRef});
      } else if (ioTypeAttr.getValue() == tt::IOType::OUTPUT) {
        outputTensorMap.insert({convertedArgTypes.size(), perCoreMemRef});
      } else {
        return funcOp.emitError("unexpected IOType value on tensor argument");
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

  for (unsigned i = 0; i < (unsigned)SpmdArg::Count; i++) {
    convertedArgTypes.push_back(rewriter.getI32Type()); // all SPMD args are i32
    auto argName = spmdArgName((SpmdArg)i);
    auto argLoc = NameLoc::get(StringAttr::get(context, argName));
    argLocs.push_back(argLoc);
  }

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
  rewriter.createBlock(&newRegion, newRegion.end(), convertedArgTypes,
                       argLocs);
  return newFunc;
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
