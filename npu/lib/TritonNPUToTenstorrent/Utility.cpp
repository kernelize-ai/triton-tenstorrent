#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/MLIRTypes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

// Tenstorrent TTKernel includes
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;

namespace mlir::arith {

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v) {
  auto i1Ty = rewriter.getIntegerType(1);
  return ConstantOp::create(rewriter, loc, i1Ty, IntegerAttr::get(i1Ty, v));
}

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32Ty = rewriter.getIntegerType(32);
  return ConstantOp::create(rewriter, loc, i32Ty, IntegerAttr::get(i32Ty, v));
}

Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v) {
  auto i64Ty = rewriter.getIntegerType(64);
  return ConstantOp::create(rewriter, loc, i64Ty, IntegerAttr::get(i64Ty, v));
}

Value createConstantF16(Location loc, OpBuilder &rewriter, float v) {
  auto f16Ty = triton::type::f16Ty(rewriter.getContext());
  return ConstantOp::create(rewriter, loc, f16Ty, rewriter.getF16FloatAttr(v));
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto f32Ty = triton::type::f32Ty(rewriter.getContext());
  return ConstantOp::create(rewriter, loc, f32Ty, rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto f64Ty = triton::type::f64Ty(rewriter.getContext());
  return ConstantOp::create(rewriter, loc, f64Ty, rewriter.getF64FloatAttr(v));
}

Value createIndexConstant(Location loc, OpBuilder &builder, int64_t value) {
  auto indexTy = builder.getIndexType();
  return ConstantOp::create(builder, loc, indexTy,
                            IntegerAttr::get(indexTy, value));
}

} // namespace mlir::arith

using namespace mlir::tt;

namespace mlir::triton::npu {

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

} // namespace mlir::triton::npu
