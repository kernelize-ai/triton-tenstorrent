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
  llvm::MapVector<Value, unsigned> layouts;
  DenseSet<Operation *> computeOps;
  FuncOp funcOp;
};

void RegisterIndexPropagation::initComputeRegisterIndices() {
  auto storeRegisterIndex = [&](Value operand, unsigned index) {
    LDBG("Mapping " << operand << " to register index " << index);
    layouts.insert({operand, index});
  };

  funcOp.walk([&](Operation *op) {
    if (isa<npu::tt::BinaryComputeOp>(op)) {
      computeOps.insert(op);
      storeRegisterIndex(op->getOperand(0), 0);
      storeRegisterIndex(op->getOperand(1), 1);
    }
  });
}

void RegisterIndexPropagation::propagateToLocalLoads() {
  for (auto &[value, index] : layouts) {
    LoadOp loadOp = dyn_cast<LoadOp>(value.getDefiningOp());
    if (!loadOp) {
      LDBG("Value " << value << " is not defined by a LoadOp, skipping");
      continue;
    }

#if 0
    loadOp->setAttr("ttts.register_index",
                   IntegerAttr::get(IntegerType::get(loadOp.getContext(), 32),
                                    index));
#else
    auto loadTensorType = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!loadTensorType) {
      LDBG("LoadOp " << loadOp << " does not have RankedTensorType, skipping");
      continue;
    }
    auto loadEncoding = loadTensorType.getEncoding();
    auto newLoadEncoding = npu::tt::TileEncodingAttr::get(
        loadOp.getContext(), index,
        cast<gpu::DistributedEncodingTrait>(loadEncoding));
    LDBG("Propagating new encoding " << newLoadEncoding << " to LoadOp "
                                     << loadOp);
    auto newLoadType = loadTensorType.cloneWithEncoding(newLoadEncoding);

    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPoint(loadOp);

    IRMapping mapping;
    for (auto operand : loadOp->getOperands()) {
      RankedTensorType operandTensorType =
          dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTensorType)
        continue;
      auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
          loadOp.getLoc(), operandTensorType.cloneWithEncoding(newLoadEncoding),
          operand);
      mapping.map(operand, cvt);
    }

    rewriter.setInsertionPointAfter(loadOp);
    Operation *newLoadOp = rewriter.clone(*loadOp, mapping);
    newLoadOp->getResult(0).setType(newLoadType);
    llvm::errs() << "new load op: " << *newLoadOp << "\n";

    auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
        loadOp.getLoc(), loadTensorType, newLoadOp->getResult(0));
    loadOp.replaceAllUsesWith(cvt.getResult());
    loadOp.erase();
#endif
  }
}

void RegisterIndexPropagation::updateComputeOpInputs() {
  for (auto *op : computeOps) {
    OpBuilder rewriter(op->getContext());
    rewriter.setInsertionPoint(op);

    IRMapping mapping;
    for (auto operand : op->getOperands()) {
#if 1
      gpu::ConvertLayoutOp cvt =
          dyn_cast<gpu::ConvertLayoutOp>(operand.getDefiningOp());
      if (!cvt)
        continue;

      // TODO: relax this condition
      if (!isa<LoadOp>(cvt.getSrc().getDefiningOp()))
        continue;

      mapping.map(operand, cvt.getSrc());
#else
      auto it = layouts.find(operand);
      if (it == layouts.end())
        continue;
      unsigned index = it->second;

      RankedTensorType operandTensorType =
          dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTensorType)
        continue;
      auto operandEncoding = operandTensorType.getEncoding();
      auto newOperandEncoding = npu::tt::TileEncodingAttr::get(
          op->getContext(), index,
          cast<gpu::DistributedEncodingTrait>(operandEncoding));

      auto cvt = rewriter.create<gpu::ConvertLayoutOp>(
          op->getLoc(), operandTensorType.cloneWithEncoding(newOperandEncoding),
          operand);
      mapping.map(operand, cvt);
#endif
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
