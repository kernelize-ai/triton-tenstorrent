#ifndef TRITONNPU_CONVERSION_TRITONNPUTOD2M_PASSES_H
#define TRITONNPU_CONVERSION_TRITONNPUTOD2M_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace npu {

#define GEN_PASS_DECL
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

} // namespace npu
} // namespace triton

} // namespace mlir

#endif
