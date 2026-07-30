#ifndef TRITON_NPU_ANALYSIS_UTILITY_H
#define TRITON_NPU_ANALYSIS_UTILITY_H

#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

BlockArgument traceToBlock(Value v, Block *parentBlock);

}
} // namespace triton
} // namespace mlir

#endif
