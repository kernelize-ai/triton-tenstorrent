#ifndef TRITON_NPU_ANALYSIS_UTILITY_H
#define TRITON_NPU_ANALYSIS_UTILITY_H

#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

BlockArgument traceToFuncArg(Value v, triton::FuncOp funcOp);

}
} // namespace triton
} // namespace mlir

#endif
