#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTCONVERTTENSORDESCTOLOADSTORE
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-convert-tensor-desc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}
struct RewriteLoadPattern : OpRewritePattern<triton::DescriptorLoadOp> {
  using OpRewritePattern<triton::DescriptorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = TensorDescriptorUnpacked(descTy, op.getDesc());
    auto offsets = op.getIndices();

    auto blockTy = descTy.getSignlessBlockType();
    auto attr = rewriter.getZeroAttr(blockTy);
    auto other = arith::ConstantOp::create(rewriter, loc, attr);

    Value mask = desc.generateMask(rewriter, loc, blockShape);

    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, desc.generatePtr(rewriter, loc, blockShape, offsets), mask, other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return success();
  }
};

struct RewriteStorePattern : OpRewritePattern<triton::DescriptorStoreOp> {
  using OpRewritePattern<triton::DescriptorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = TensorDescriptorUnpacked(descTy, op.getDesc());
    auto offsets = op.getIndices();

    Value mask = desc.generateMask(rewriter, loc, blockShape);

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, desc.generatePtr(rewriter, loc, blockShape, offsets), op.getSrc(),
        mask, triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return success();
  }
};

} // namespace

class TritonTenstorrentConvertTensorDescPass
    : public npu::impl::TritonTenstorrentConvertTensorDescToLoadStoreBase<
          TritonTenstorrentConvertTensorDescPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<RewriteLoadPattern, RewriteStorePattern>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(mod, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
