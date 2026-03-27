#include "npu/include/TritonNPUToD2M/Passes.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOD2M
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct ConvertTritonNPUToD2MPass
    : public impl::ConvertTritonNPUToD2MBase<ConvertTritonNPUToD2MPass> {
  void runOnOperation() override { assert(false && "TODO"); }
};

} // namespace npu
} // namespace triton
} // namespace mlir
