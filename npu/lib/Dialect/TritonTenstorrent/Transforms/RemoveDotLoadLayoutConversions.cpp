#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTREMOVEDOTLOADLAYOUTCONVERSIONS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-remove-dot-load-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class TritonTenstorrentRemoveDotLoadLayoutConversionsPass
    : public npu::impl::TritonTenstorrentRemoveDotLoadLayoutConversionsBase<
          TritonTenstorrentRemoveDotLoadLayoutConversionsPass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m.walk([](triton::gpu::ConvertLayoutOp cvtOp) {
      LDBG("Evaluate cvtOp for removal: " << cvtOp << "\n");
      Value loadSrc = cvtOp.getSrc();
      auto srcOp = loadSrc.getDefiningOp();
      if (!srcOp) {
        // ignore block arguments
        return;
      }

      auto loadOp = dyn_cast<triton::LoadOp>(srcOp);
      auto cvtResultType = cast<RankedTensorType>(cvtOp.getType());

      if (loadOp && isa<npu::tt::TiledDotOperandEncodingAttr>(
                        cvtResultType.getEncoding())) {
        LDBG("Remove cvtOp " << *cvtOp
                             << " and push tiled_dot_op encoding into load");

        OpBuilder builder(loadOp);

#if 0
        auto newLoad = builder.clone(*loadOp);
        newLoad->getResult(0).setType(cvtResultType);
        cvtOp.replaceAllUsesWith(newLoad->getResult(0));
        cvtOp.erase();
        loadOp.replaceAllUsesWith(newLoad->getResult(0));
        loadOp.erase();
#else

        IRMapping mapping;
        for (auto operand : loadOp->getOperands()) {
          RankedTensorType operandTensorType =
              dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTensorType)
            continue;
          auto cvt = gpu::ConvertLayoutOp::create(
              builder, loadOp.getLoc(),
              operandTensorType.cloneWithEncoding(cvtResultType.getEncoding()),
              operand);
          mapping.map(operand, cvt);
        }

        builder.setInsertionPointAfter(loadOp);
        Operation *newLoadOp = builder.clone(*loadOp, mapping);
        newLoadOp->getResult(0).setType(cvtResultType);
        LDBG("Created new LoadOp: " << *newLoadOp);

        cvtOp.replaceAllUsesWith(newLoadOp->getResult(0));
        cvtOp.erase();
        loadOp.replaceAllUsesWith(newLoadOp->getResult(0));
        loadOp.erase();
#endif
      }
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
