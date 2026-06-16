#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

namespace triton {
namespace npu {
static constexpr llvm::StringLiteral kTTNumCommonArgsAttr =
    "tt.num_common_args";
static constexpr llvm::StringLiteral kTTNumPerCoreArgsAttr =
    "tt.num_per_core_args";

namespace GridArgffsets {
constexpr int kStrideX = 0;
constexpr int kStrideY = 1;
constexpr int kGridX = 2;
constexpr int kGridY = 3;
} // namespace GridArgffsets

namespace PerCoreArgOffsets {
constexpr int kMulticastGroup = 0; // TTKernel only
} // namespace PerCoreArgOffsets

} // namespace npu
} // namespace triton

namespace arith {

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v);
Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);

Value createIndexConstant(Location loc, OpBuilder &builder, int64_t value);

} // namespace arith
} // namespace mlir

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H
