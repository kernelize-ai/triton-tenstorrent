#include "PointerInfoAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/Support/Debug.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "triton-npu-pointer-info-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline Value traceToBaseAddress(Value ptr) {
  if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
    LDBG("Tracing load op pointer from block arg: " << blockArg);

    auto parentOp = blockArg.getParentBlock()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      Value sourceVal = forOp.getInitArgs()[blockArg.getArgNumber() -
                                            forOp.getNumInductionVars()];
      return traceToBaseAddress(sourceVal);
    }
    llvm_unreachable("expected block-arg ptr value to be child of scf::ForOp");
  }

  SetVector<Operation *> baseAddrSlice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = [](Operation *op) { return !isa<ttkernel::GetArgValOp>(op); };
  (void)getBackwardSlice(ptr, &baseAddrSlice, opt);
  LLVM_DEBUG(for (Operation *op : baseAddrSlice) {
    DBGS() << "backward slice op: " << *op << "\n";
  });

  Value baseAddr;
  for (auto op : baseAddrSlice) {
    if (op->getNumOperands() == 1 &&
        isa<IntegerType>(op->getOperand(0).getType())) {
      baseAddr = op->getOperand(0);
      break;
    }
  }

  assert(baseAddr && "could not find base address in backward slice");
  LDBG("Found base address: " << baseAddr << ", for ptr: " << ptr);
  return baseAddr;
}

} // namespace

PointerInfoAnalysis::PointerInfoAnalysis(mlir::Operation *root) {
  root->walk([&](triton::LoadOp loadOp) {
    LDBG("pointer analysis for load op : " << *loadOp << "\n");
    Value baseAddr = traceToBaseAddress(loadOp.getPtr());
    assert(ptrInfo.try_emplace(loadOp, PointerInfo{baseAddr}).second &&
           "expected unique load op in pointer info analysis");
  });
  root->walk([&](triton::StoreOp storeOp) {
    LDBG("pointer analysis for store op : " << *storeOp << "\n");
    Value baseAddr = traceToBaseAddress(storeOp.getPtr());
    assert(ptrInfo.try_emplace(storeOp, PointerInfo{baseAddr}).second &&
           "expected unique store op in pointer info analysis");
    LDBG("store op base address: " << baseAddr << "\n");
  });
}

} // namespace npu
} // namespace triton
} // namespace mlir
