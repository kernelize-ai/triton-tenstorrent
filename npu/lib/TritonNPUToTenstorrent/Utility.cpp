#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Conversion/MLIRTypes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

// Tenstorrent TTKernel includes
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::tt;

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

namespace mlir::triton::npu {

Value createNocId(ConversionPatternRewriter &rewriter, Location loc,
                  int64_t nocIndex) {
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getI8IntegerAttr(nocIndex));
}

int64_t getKernelNocIndex(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  assert(funcOp && "expected data-movement op inside a kernel func");
  return funcOp.getSymName().ends_with("__writer") ? kWriterNocIndex
                                                   : kReaderNocIndex;
}

std::optional<int64_t> getBaseCommonArgIndex(Value baseAddr) {
  auto getArg = baseAddr.getDefiningOp<ttkernel::GetCommonArgValOp>();
  if (!getArg)
    return std::nullopt;
  auto cst = getArg.getArgIndex().getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
    return intAttr.getInt();
  return std::nullopt;
}

Value findAccessorArgsForBuffer(Operation *op, int64_t baseArgIndex) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return nullptr;
  Value result;
  funcOp.walk([&](ttkernel::TensorAccessorArgsOp argsOp) {
    auto attr = argsOp->getAttrOfType<IntegerAttr>(kAccessorBaseArgIndexAttr);
    if (attr && attr.getInt() == baseArgIndex) {
      result = argsOp.getResult();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

Value createTensorAccessor(ConversionPatternRewriter &rewriter, Location loc,
                           Operation *op, Value baseAddr, Value pageSize) {
  std::optional<int64_t> argIdx = getBaseCommonArgIndex(baseAddr);
  assert(argIdx &&
         "expected buffer base address to come from get_common_arg_val");
  Value accessorArgs = findAccessorArgsForBuffer(op, *argIdx);
  assert(
      accessorArgs &&
      "no TensorAccessorArgs found for buffer; expected the func-entry chain "
      "built in TritonFuncOpToFuncOp");
  return ttkernel::TensorAccessorOp::create(
      rewriter, loc, ttkernel::TensorAccessorType::get(rewriter.getContext()),
      accessorArgs, baseAddr, pageSize);
}

} // namespace mlir::triton::npu
