#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTCONVERTCOMPUTEOPS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-convert-compute-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

template <typename OpType>
struct RewriteBinaryComputeOp : OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() < 1)
      return failure();

    if (!llvm::all_of(op->getOperands(), [](Value val) {
          // TODO: is isa<LoadOp> a strong enough condition by itself?
          // probably not if we have convert layout ops
          return isa<RankedTensorType>(val.getType());
        }))
      return failure();

    auto opcode = rewriter.getStringAttr(op->getName().getStringRef());
    LDBG("Rewriting binary compute op: " << opcode);

    SmallVector<Type> resultTys(op->getResultTypes().begin(),
                                op->getResultTypes().end());

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    auto newOp = npu::tt::BinaryComputeOp::create(rewriter, op.getLoc(),
                                                  resultTys, lhs, rhs, opcode);
    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

} // namespace

class TritonTenstorrentConvertComputeOpsPass
    : public npu::impl::TritonTenstorrentConvertComputeOpsBase<
          TritonTenstorrentConvertComputeOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<RewriteBinaryComputeOp<arith::AddFOp>>(context);
    patterns.add<RewriteBinaryComputeOp<arith::SubFOp>>(context);
    patterns.add<RewriteBinaryComputeOp<arith::MulFOp>>(context);
    patterns.add<RewriteBinaryComputeOp<arith::DivFOp>>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
