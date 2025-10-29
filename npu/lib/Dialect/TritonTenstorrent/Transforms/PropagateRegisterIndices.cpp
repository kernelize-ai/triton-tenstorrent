#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTPROPAGATEREGISTERINDICES
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-propagate-register-indices"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

class RegisterIndexPropagation {
public:
  RegisterIndexPropagation(FuncOp funcOp) : funcOp(funcOp) {}

  // Find the compute ops and associate their register indices to operand values
  void initComputeRegisterIndices();

  // Propagate register indices to local loads
  void propagateToLocalLoads();

  // override the inputs of the compute op with the new layouts
  void updateComputeOpInputs();

private:
  // map from value to its index in the tile register buffer
  llvm::MapVector<Operation *, unsigned> layouts;
  DenseSet<Operation *> computeOps;
  FuncOp funcOp;
};

void RegisterIndexPropagation::initComputeRegisterIndices() {
  auto storeRegisterIndex = [&](Operation *op, unsigned index) {
    LDBG("Mapping " << op << " to register index " << index);
    layouts.insert({op, index});
  };

  funcOp.walk([&](Operation *op) {
    if (isa<npu::tt::BinaryComputeOp>(op)) {
      computeOps.insert(op);
      storeRegisterIndex(op->getOperand(0).getDefiningOp(), 0);
      storeRegisterIndex(op->getOperand(1).getDefiningOp(), 1);
      for (Operation *user : op->getResult(0).getUsers()) {
        if (isa<StoreOp>(user)) {
          storeRegisterIndex(user, 2);
        }
      }
    }
  });
}

void RegisterIndexPropagation::propagateToLocalLoads() {
  for (auto &[op, index] : layouts) {
    auto propagateToLoads = [](LoadOp loadOp, unsigned index) {
      auto loadTensorType = dyn_cast<RankedTensorType>(loadOp.getType());
      if (!loadTensorType) {
        LDBG("LoadOp " << loadOp
                       << " does not have RankedTensorType, skipping");
        return;
      }
      auto loadEncoding = loadTensorType.getEncoding();
      auto newLoadEncoding = npu::tt::TileEncodingAttr::get(
          loadOp.getContext(), index,
          cast<gpu::DistributedEncodingTrait>(loadEncoding));
      LDBG("Propagating new encoding " << newLoadEncoding << " to LoadOp "
                                       << loadOp);
      auto newLoadType = loadTensorType.cloneWithEncoding(newLoadEncoding);

      OpBuilder rewriter(loadOp.getContext());
      rewriter.setInsertionPoint(loadOp);

      IRMapping mapping;
      for (auto operand : loadOp->getOperands()) {
        RankedTensorType operandTensorType =
            dyn_cast<RankedTensorType>(operand.getType());
        if (!operandTensorType)
          continue;
        auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
            loadOp.getLoc(),
            operandTensorType.cloneWithEncoding(newLoadEncoding), operand);
        mapping.map(operand, cvt);
      }

      rewriter.setInsertionPointAfter(loadOp);
      Operation *newLoadOp = rewriter.clone(*loadOp, mapping);
      newLoadOp->getResult(0).setType(newLoadType);
      LDBG("Created new LoadOp: " << *newLoadOp);

      auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
          loadOp.getLoc(), loadTensorType, newLoadOp->getResult(0));
      loadOp.replaceAllUsesWith(cvt.getResult());
      loadOp.erase();
    };

    auto propagateToStores = [](StoreOp storeOp, unsigned index) {
      auto storeTensorType =
          dyn_cast<RankedTensorType>(storeOp.getValue().getType());
      if (!storeTensorType) {
        LDBG("StoreOp " << storeOp
                        << " does not have RankedTensorType, skipping");
        return;
      }
      auto storeEncoding = storeTensorType.getEncoding();
      auto newStoreEncoding = npu::tt::TileEncodingAttr::get(
          storeOp.getContext(), index,
          cast<gpu::DistributedEncodingTrait>(storeEncoding));
      LDBG("Propagating new encoding " << newStoreEncoding << " to StoreOp "
                                       << storeOp);
      auto newStoreType = storeTensorType.cloneWithEncoding(newStoreEncoding);

      OpBuilder rewriter(storeOp.getContext());
      rewriter.setInsertionPoint(storeOp);

      IRMapping mapping;
      for (auto operand : storeOp->getOperands()) {
        RankedTensorType operandTensorType =
            dyn_cast<RankedTensorType>(operand.getType());
        if (!operandTensorType)
          continue;
        if (isa<npu::tt::BinaryComputeOp>(operand.getDefiningOp())) {
          operand.getDefiningOp()->getResult(0).setType(
              operandTensorType.cloneWithEncoding(newStoreEncoding));
        } else {
          auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
              storeOp.getLoc(),
              operandTensorType.cloneWithEncoding(newStoreEncoding), operand);
          mapping.map(operand, cvt);
        }
      }

      rewriter.setInsertionPointAfter(storeOp);
      Operation *newStoreOp = rewriter.clone(*storeOp, mapping);
      LDBG("Created new StoreOp: " << *newStoreOp);

      storeOp.erase();
    };

    LDBG("Propagating register index " << index << " for operation " << op);

    LoadOp loadOp = dyn_cast<LoadOp>(op);
    if (loadOp) {
      propagateToLoads(loadOp, index);
      continue;
    }
    StoreOp storeOp = dyn_cast<StoreOp>(op);
    if (storeOp) {
      propagateToStores(storeOp, index);
      continue;
    }
    LDBG("Operation " << op << " is neither LoadOp nor StoreOp, skipping");
  }
}

void RegisterIndexPropagation::updateComputeOpInputs() {
  for (auto *op : computeOps) {
    OpBuilder rewriter(op->getContext());
    rewriter.setInsertionPoint(op);

    IRMapping mapping;
    for (auto operand : op->getOperands()) {
      gpu::ConvertLayoutOp cvt =
          dyn_cast<gpu::ConvertLayoutOp>(operand.getDefiningOp());
      if (!cvt)
        continue;

      // TODO: relax this condition
      if (!isa<LoadOp>(cvt.getSrc().getDefiningOp()))
        continue;

      mapping.map(operand, cvt.getSrc());
    }

    Operation *newOp = rewriter.clone(*op, mapping);
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();
  }
}

} // namespace

class TritonTenstorrentPropagateRegisterIndicesPass
    : public triton::npu::impl::TritonTenstorrentPropagateRegisterIndicesBase<
          TritonTenstorrentPropagateRegisterIndicesPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m.walk([](FuncOp funcOp) {
      StringRef funcName = funcOp.getSymName();
      if (false && !funcName.ends_with("__compute"))
        return;

      RegisterIndexPropagation propagation(funcOp);
      propagation.initComputeRegisterIndices();
      propagation.propagateToLocalLoads();
      propagation.updateComputeOpInputs();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
