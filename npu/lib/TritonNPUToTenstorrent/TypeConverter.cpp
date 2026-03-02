#include "TypeConverter.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

// TODO: move getTileDimSize() to Dialect
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::triton;

namespace {

SmallVector<int64_t, 2> convertShapeToTileShape(ArrayRef<int64_t> shape) {
  if (shape.size() == 1) {
    return SmallVector<int64_t, 2>{
        shape[0] / (npu::getTileDimSize() * npu::getTileDimSize())};
  }
  SmallVector<int64_t, 2> tileShape;
  tileShape.reserve(shape.size());
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) {
      tileShape.push_back(1);
      continue;
    }
    assert(shape[i] % npu::getTileDimSize() == 0 &&
           "expecting shape dimensions to be multiple of tile dimension size");
    tileShape.push_back(shape[i] / npu::getTileDimSize());
  }
  return tileShape;
}

Type convertTypeToCBType(Type type) {
  assert(isa<RankedTensorType>(type) && "expected ranked tensor type");
  auto rankedType = cast<RankedTensorType>(type);
  auto etype = rankedType.getElementType();
  auto shape = convertShapeToTileShape(rankedType.getShape());
  auto ttcoreTileType = ttcore::TileType::get(
      type.getContext(), ttcore::TileType::getDefaultShape(),
      ttcore::elementTypeToDataType(etype));
  MemRefType cbMemRefType =
      MemRefType::get(shape, ttcoreTileType, MemRefLayoutAttrInterface{},
                      ttcore::MemorySpaceAttr::get(
                          type.getContext(), ttcore::MemorySpace::DeviceL1));
  return ttkernel::CBType::get(cbMemRefType);
}

} // namespace

namespace mlir {
namespace triton {
namespace npu {

TritonNPUToTenstorrentTypeConverter::TritonNPUToTenstorrentTypeConverter(
    MLIRContext *)
    : TypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([](triton::gpu::MemDescType memdesc) {
    // convert memdesc to memref
    auto shape = convertShapeToTileShape(memdesc.getShape());
    auto ttcoreTileType = ttcore::TileType::get(
        memdesc.getContext(), ttcore::TileType::getDefaultShape(),
        ttcore::elementTypeToDataType(memdesc.getElementType()));

    MemRefType cbMemRefType = MemRefType::get(
        shape, ttcoreTileType, MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(memdesc.getContext(),
                                     ttcore::MemorySpace::DeviceL1));
    return ttkernel::CBType::get(cbMemRefType);
  });
  addConversion([](RankedTensorType type) -> Type {
    auto etype = type.getElementType();
    if (isa<triton::PointerType>(etype)) {
      return IntegerType::get(type.getContext(), 32);
    }
    if (!type.getEncoding()) {
      return type;
    }
    if (isa<npu::tt::TiledDotOperandEncodingAttr>(type.getEncoding()) ||
        isa<gpu::DotOperandEncodingAttr>(type.getEncoding())) {
      // dot operands read directly from cbs, so convert to cb type
      assert(type.getShape().size() == 2 &&
             "expecting rank 2 tensor for dot operand");
      return convertTypeToCBType(type);
    }
    return type;
  });
  addConversion([](triton::PointerType type) -> Type {
    // convert pointer to i32
    return IntegerType::get(type.getContext(), 32);
  });
  addSourceMaterialization([](OpBuilder &builder, PointerType type,
                              ValueRange inputs, Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
  addConversion([](mlir::triton::TensorDescType t,
                   llvm::SmallVectorImpl<mlir::Type> &out) {
    // We convert a tensor descriptor into an pointer, and a shape and stride
    // for each dimension, and padding option. i.e., we create 1+2*rank+1
    // values. Note that tensor descriptors may be signed/unsigned integers
    // whereas pointers should always be signless.
    auto tensorType = t.getSignlessBlockType();
    out.push_back(triton::getPointerType(tensorType.getElementType()));
    out.insert(out.end(), 2 * tensorType.getRank(),
               mlir::IntegerType::get(t.getContext(), 32));
    out.push_back(mlir::IntegerType::get(t.getContext(), 1));
    return mlir::success();
  });
  addSourceMaterialization([](OpBuilder &builder,
                              mlir::triton::TensorDescType type,
                              ValueRange inputs, Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
}

} // namespace npu
} // namespace triton
} // namespace mlir
