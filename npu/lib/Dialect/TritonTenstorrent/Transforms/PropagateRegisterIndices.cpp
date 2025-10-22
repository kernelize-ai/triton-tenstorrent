#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTPROPAGATEREGISTERINDICES
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-propagate-register-indices"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class TritonTenstorrentPropagateRegisterIndicesPass
    : public triton::npu::impl::TritonTenstorrentPropagateRegisterIndicesBase<
          TritonTenstorrentPropagateRegisterIndicesPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // TODO

  }
};


}
}
}
