#include "npu/include/TritonNPUToD2M/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_HOISTCBALLOCS
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "hoist-cb-allocs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {}

struct HoistCBAllocsPass : public impl::HoistCBAllocsBase<HoistCBAllocsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    assert(false && "TODO");
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
