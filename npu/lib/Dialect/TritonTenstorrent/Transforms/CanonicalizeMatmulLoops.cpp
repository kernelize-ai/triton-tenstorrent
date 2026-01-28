#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTCANONICALIZEMATMULLOOPS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-canonicalize-matmul-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct RemoveUnusedPointerArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    unsigned ivCount = forOp.getNumInductionVars();

    Block &body = forOp.getRegion().front();
    unsigned numIterArgs = body.getNumArguments() - ivCount;
    if (numIterArgs == 0)
      return failure();

    SetVector<unsigned> indicesToRemove;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      Value result = forOp.getResult(i);
      Value iterArg = body.getArgument(ivCount + i);

      LDBG("Checking iter arg " << i << ", " << iterArg);

      auto tensorType = dyn_cast<RankedTensorType>(iterArg.getType());
      if (tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
        LDBG("Arg is tensor type, evaluating for removal");

        bool usersArePure = true;
        for (Operation *user : iterArg.getUsers()) {
          if (!isa<scf::YieldOp>(user) && !isPure(user)) {
            LDBG("Found user with side effects, unable to remove arg. User: "
                 << *user);
            usersArePure = false;
            break;
          }
        }

        if (result.use_empty() && usersArePure)
          indicesToRemove.insert(i);
      }
    }

    LDBG("Removing " << indicesToRemove.size() << " unused pointer args");
    if (indicesToRemove.empty())
      return failure();

    auto oldInitArgs = forOp.getInitArgs();
    SmallVector<Value> newInitArgs;
    SmallVector<Type> newResultTypes;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      if (!indicesToRemove.contains(i)) {
        newInitArgs.push_back(oldInitArgs[i]);
        newResultTypes.push_back(forOp.getResult(i).getType());
      }
    }

    rewriter.setInsertionPoint(forOp);
    auto newFor =
        scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);

    Block *newBody = newFor.getBody();
    IRMapping mapping;
    for (unsigned i = 0; i < ivCount; ++i)
      mapping.map(body.getArgument(i), newBody->getArgument(i));

    SmallVector<int> iterArgsOldToNewMap(numIterArgs, -1);
    unsigned newIterArgPos = 0;
    for (unsigned i = 0; i < numIterArgs; i++) {
      Value oldIterArg = body.getArgument(ivCount + i);

      if (indicesToRemove.contains(i)) {
        mapping.map(oldIterArg, oldInitArgs[i]);
      } else {
        Value newIterArg = newBody->getArgument(ivCount + newIterArgPos);
        mapping.map(oldIterArg, newIterArg);
        iterArgsOldToNewMap[i] = newIterArgPos;
        ++newIterArgPos;
      }
    }
    rewriter.setInsertionPointToEnd(newBody);
    for (Operation &op : body.without_terminator())
      rewriter.clone(op, mapping);

    auto yield = cast<scf::YieldOp>(body.getTerminator());
    // Build the new yield with only kept operands.
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < numIterArgs; i++) {
      if (indicesToRemove.contains(i))
        continue;
      newYieldOperands.push_back(mapping.lookupOrDefault(yield.getOperand(i)));
    }

    scf::YieldOp::create(rewriter, newFor.getLoc(), newYieldOperands);

    for (unsigned i = 0; i < numIterArgs; i++) {
      if (indicesToRemove.contains(i)) {
        assert(forOp.getResult(i).use_empty() &&
               "expected removed result to have no uses");
      } else {
        assert(iterArgsOldToNewMap[i] != -1 &&
               "expected valid mapping for kept iter arg");
        rewriter.replaceAllUsesWith(forOp.getResult(i),
                                    newFor.getResult(iterArgsOldToNewMap[i]));
      }
    }

    rewriter.eraseOp(forOp);
    return success();
  }
};

} // namespace

class TritonTenstorrentCanonicalizeMatmulLoops
    : public impl::TritonTenstorrentCanonicalizeMatmulLoopsBase<
          TritonTenstorrentCanonicalizeMatmulLoops> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    auto computeKernel = findComputeKernel(mod);
    if (!computeKernel) {
      return;
    }
    RewritePatternSet patterns(context);
    patterns.add<RemoveUnusedPointerArgs>(patterns.getContext());
    if (failed(applyPatternsGreedily(computeKernel, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
