#include "npu/include/Dialect/TritonTenstorrent/Transforms/RegAlias.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/LLVM.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tt-reg-alias"
#define LDBG(x) LLVM_DEBUG(llvm::dbgs() << x << "\n")

namespace mlir {
namespace triton {
namespace npu {

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

LogicalResult RegAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
  LDBG("RegAliasAnalysis::visitOperation: " << op->getName());
  if (op->getNumResults() == 0)
    return success();
  auto result = op->getResult(0);
  AliasInfo aliasInfo;
  if (isa<triton::DotOp>(op)) {
    aliasInfo.insert(result);
  } else if (isa<ub::PoisonOp>(op)) {
    aliasInfo = AliasInfo();
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));

  return success();
}

} // namespace npu
} // namespace triton
} // namespace mlir
