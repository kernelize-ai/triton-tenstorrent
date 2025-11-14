#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace mlir::arith {

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v);
Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);

Value createIndexConstant(Location loc, OpBuilder &builder, int64_t value);

} // namespace mlir::arith

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H
