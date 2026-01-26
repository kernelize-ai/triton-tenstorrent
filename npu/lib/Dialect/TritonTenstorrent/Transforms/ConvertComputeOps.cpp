#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
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
          return isa<RankedTensorType>(val.getType());
        }))
      return failure();

    auto opcode = rewriter.getStringAttr(op->getName().getStringRef());
    LDBG("Rewriting binary compute op: " << opcode);

    // TODO: create a TiledEncodingAttr builder and unify with AccelerateMatmul
    auto rowMajorOrder = SmallVector<unsigned>{1, 0};
    static constexpr std::array<unsigned, 2> tileSize = {32, 32};

    // if any of the tiled operands are loaded from a DescriptorLoadOp, we can
    // use tiled encoding for all operands
    const bool canUseTiledEncoding =
        llvm::any_of(op->getOperands(), [](Value val) {
          auto definingOp = val.getDefiningOp();
          return definingOp && isa<triton::DescriptorLoadOp>(definingOp);
        });

    auto updateOperandEncoding = [&](Value v) -> Value {
      if (!canUseTiledEncoding)
        return v;

      auto vType = cast<RankedTensorType>(v.getType());
      auto vShape = vType.getShape();
      assert(vShape.size() <= 2 &&
             "expected <= rank 2 tensor for binary compute op conversion");

      SmallVector<unsigned> tiledShape(vShape.size());
      for (auto [idx, dim] : llvm::enumerate(vShape)) {
        tiledShape[idx] = static_cast<unsigned>(dim / tileSize[idx]);
      }

      auto tiledEncoding = npu::tt::TiledEncodingAttr::get(
          v.getContext(), tiledShape, rowMajorOrder,
          /*tileShape=*/ArrayRef<unsigned>{tileSize[0], tileSize[1]});
      auto newVType = vType.cloneWithEncoding(tiledEncoding);
      return gpu::ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v)
          .getResult();
    };

    Value lhs = updateOperandEncoding(op->getOperand(0));
    Value rhs = updateOperandEncoding(op->getOperand(1));

    auto newOp = npu::tt::BinaryComputeOp::create(
        rewriter, op.getLoc(), lhs.getType(), lhs, rhs, opcode);
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
