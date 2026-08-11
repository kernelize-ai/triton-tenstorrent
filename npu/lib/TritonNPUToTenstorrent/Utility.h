#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace triton {
namespace npu {
static constexpr llvm::StringLiteral kTTNumCommonArgsAttr =
    "tt.num_common_args";
static constexpr llvm::StringLiteral kTTNumPerCoreArgsAttr =
    "tt.num_per_core_args";
static constexpr llvm::StringLiteral kAccessorBaseArgIndexAttr =
    "tt.accessor_base_arg_index";

namespace PerCoreArgOffsets {
constexpr int kBlockStart = 0;
constexpr int kBlockEnd = 1;
constexpr int kNumBlocks = 2;
constexpr int kThreadId = 3;
} // namespace PerCoreArgOffsets

// Every NoC op in a kernel runs on *that kernel's* NoC: the reader kernel
// runs on RISCV_1 (NOC 1) and the writer on RISCV_0 (NOC 0), regardless of
// whether the op itself is a read or a write. See getKernelNocIndex.
constexpr int64_t kReaderNocIndex = 1;
constexpr int64_t kWriterNocIndex = 0;

// Build a materialized i8 NoC index constant for use as the optional `noc`
// operand on NoC-family TTKernel ops.
Value createNocId(ConversionPatternRewriter &rewriter, Location loc,
                  int64_t nocIndex);

// Resolve the NoC index for a data-movement op from its enclosing kernel
// func (reader kernels use NOC 1, writer/compute use NOC 0).
int64_t getKernelNocIndex(Operation *op);

// A buffer's base address is materialized by the func conversion as a
// `ttkernel.get_common_arg_val(idx)`; recover `idx` so it can be matched to
// the `TensorAccessorArgs` built for it at function entry.
std::optional<int64_t> getBaseCommonArgIndex(Value baseAddr);

// Find the chained `ttkernel.TensorAccessorArgs` that the func conversion
// built at the enclosing kernel's entry for the buffer whose base address
// has the given common-arg index (tagged with kAccessorBaseArgIndexAttr).
// See TritonFuncOpToFuncOp.cpp.
Value findAccessorArgsForBuffer(Operation *op, int64_t baseArgIndex);

// Build a TensorAccessor for a buffer access by pairing the buffer's runtime
// base address + page size with the chained `TensorAccessorArgs` reserved
// for that buffer at function entry.
Value createTensorAccessor(ConversionPatternRewriter &rewriter, Location loc,
                           Operation *op, Value baseAddr, Value pageSize);

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
