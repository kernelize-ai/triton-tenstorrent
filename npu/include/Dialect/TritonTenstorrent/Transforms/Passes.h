#ifndef TRITONTENSTORRENT_TRANSFORMS_PASSES_H
#define TRITONTENSTORRENT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace npu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
