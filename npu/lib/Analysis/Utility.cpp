#include "npu/include/Analysis/Utility.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace triton {
namespace npu {

BlockArgument traceToFuncArg(Value v, triton::FuncOp funcOp) {
  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(v))
      return blockArg.getOwner() == &funcOp.getBody().front() ? blockArg
                                                              : nullptr;

    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;

    // traverse the addptr -> splat -> broadcast chain to find the original
    // ptr argument
    v = llvm::TypeSwitch<Operation *, Value>(def)
            .Case<triton::AddPtrOp>(
                [](auto addPtrOp) { return addPtrOp.getPtr(); })
            .Case<triton::SplatOp>(
                [](auto splatOp) { return splatOp.getSrc(); })
            .Case<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
                  triton::BitcastOp>([](auto o) { return o->getOperand(0); })
            .Default([](Operation *) { return Value(); });

    if (!v)
      return nullptr;
  }
}

} // namespace npu
} // namespace triton
} // namespace mlir
