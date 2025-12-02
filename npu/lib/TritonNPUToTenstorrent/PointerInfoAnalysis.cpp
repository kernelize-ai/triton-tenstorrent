#include "PointerInfoAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/Support/Debug.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

// maybe this should be analysis specific
#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline Value traceToBaseAddress(Value ptr) {
  SetVector<Operation *> baseAddrSlice;
  mlir::BackwardSliceOptions opt;
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
    assert(loadInfo.try_emplace(loadOp, PointerInfo{baseAddr}).second &&
           "expected unique load op in pointer info analysis");
  });
}

} // namespace npu
} // namespace triton
} // namespace mlir
