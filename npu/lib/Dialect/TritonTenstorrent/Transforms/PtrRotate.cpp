#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <utility>

#define DEBUG_TYPE "tritontenstorrent-ptr-rotate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTPTRROTATE
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

/// The pass structure/action is roughly:
///
/// 1. Perform an approximate sparse dataflow analysis to find all transitive
/// uses for `tt.func` args that are `tt.ptr`s; legalize only these ops;
/// 2. Rewrite all operations' `use`s and `result`s to be `(%baseptr,
/// %offsetptr)` using `ConversionPattern`s that takes the new
/// `OneToNOpAdaptor`, which automatically forwards both `%baseptr` and
/// `%offsetptr` through `adaptor.getOperands()`[^3];
/// 3. Clean up remaining `unrealized_casts` (currently only handling one
/// category of such remaining casts but can be extended to handle all; see
/// bullet 1 in TODOs).
class TritonTenstorrentPtrRotate
    : public impl::TritonTenstorrentPtrRotateBase<TritonTenstorrentPtrRotate> {
  using Base::Base;

public:
  void runOnOperation() override;

private:
  Value rotateArith(Value basePtr, OpOperand &offset);
};

// rotate the scalar offset to basePtr
Value TritonTenstorrentPtrRotate::rotateArith(Value basePtr,
                                              OpOperand &offset) {
  auto defOp = offset.get().getDefiningOp();
  if (defOp) {
    OpBuilder builder(defOp);
    if (isa<triton::SplatOp>(defOp)) {
      // scalar splat, return the source
      auto splatOp = cast<triton::SplatOp>(defOp);
      return splatOp.getSrc();
    } else if (isa<arith::AddIOp>(defOp)) {
      auto addOp = cast<arith::AddIOp>(defOp);
      if (isa<RankedTensorType>(addOp.getType())) {
        // tensor add, rotate operand to basePtr
        for (auto [idx, operand] : llvm::enumerate(addOp->getOpOperands())) {
          Value newOperand = rotateArith(basePtr, operand);
          if (newOperand != basePtr) {
            basePtr = triton::AddPtrOp::create(builder, basePtr.getLoc(),
                                               basePtr.getType(), basePtr,
                                               newOperand);
            // replace offset with the other operand
            offset.set(addOp->getOperand(idx ^ 1));
            return basePtr;
          }
        }
      }
    }
    // TODO: handle other arithmetic ops (MulI)
  }
  return basePtr;
}

void TritonTenstorrentPtrRotate::runOnOperation() {
  getOperation()->walk([&](triton::AddPtrOp op) {
    auto srcOp = op.getPtr().getDefiningOp();
    if (!srcOp || !isa<triton::SplatOp>(srcOp)) {
      op.emitRemark("skipping addptr due to not a splat op");
      return WalkResult::advance();
    }
    Value basePtr = srcOp->getOperand(0);
    Value newBasePtr = rotateArith(basePtr, op->getOpOperand(1));
    srcOp->setOperand(0, newBasePtr);
    return WalkResult::advance();
  });
}

} // namespace npu
} // namespace triton
} // namespace mlir
