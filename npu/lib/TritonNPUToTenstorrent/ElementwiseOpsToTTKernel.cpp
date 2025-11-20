#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "Utility.h"

namespace mlir {
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertAddPtrOp : public OpConversionPattern<AddPtrOp> {
  using OpConversionPattern<AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Take a backward slice from the offset operand up to find the integer
    // offset value for the block. The last op in the slice should be the offset
    // value. Intermediate ops convert the offset to an appropriate tensor
    // representation.
    SetVector<Operation *> slice;
    (void)getBackwardSlice(op.getOffset(), &slice);
    LLVM_DEBUG(for (Operation *op : slice) {
      DBGS() << "backward slice op: " << *op << "\n";
    });

    auto it = std::find_if(slice.rbegin(), slice.rend(), [](Operation *op) {
      return isa<IntegerType>(op->getResult(0).getType());
    });
    if (it == slice.rend()) {
      return rewriter.notifyMatchFailure(
          op, "could not find integer offset in backward slice");
    }

    Value offset = (*it)->getResult(0);

    LDBG("Converting AddPtrOp offset: " << offset);

    // Drop the base addr and just return the offset in bytes
    auto tensorType = cast<RankedTensorType>(op.getPtr().getType());
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
    Value elemSizeValue = arith::createConstantI32(loc, rewriter, elemSize);
    offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);
    rewriter.replaceOp(op, offset);

    return success();
  }
};

template <typename OpTy>
struct ArithBinaryOpOnTensorsConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if constexpr (!std::is_same<OpTy, arith::CmpIOp>::value) {
      if (isa<IntegerType>(adaptor.getLhs().getType()) &&
          isa<IntegerType>(adaptor.getRhs().getType())) {
        // probably can use replaceOpWithNewOp now that we've dropped CmpIOp
        auto newOp = OpTy::create(rewriter, op.getLoc(), adaptor.getLhs(),
                                  adaptor.getRhs());
        rewriter.replaceOp(op, newOp);
        return success();
      }
    }
    if (isa<RankedTensorType>(op.getLhs().getType()) &&
        isa<RankedTensorType>(op.getRhs().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

template <typename OpTy>
struct ArithUnaryOpOnTensorsConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getOperand().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // namespace

void populateElementwiseOpConversionPattern(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<ConvertAddPtrOp>(typeConverter, patterns.getContext());

#define POPULATE_ARITH_BINARY_OP_ON_TENSORS(OP)                                \
  patterns.add<ArithBinaryOpOnTensorsConversion<OP>>(typeConverter,            \
                                                     patterns.getContext());

  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::AddIOp);
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::MulIOp);
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::RemSIOp);

  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::AndIOp);

  // TODO: can this be handled by RemoveRedundantMasks instead?
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::CmpIOp);

#define POPULATE_ARITH_UNARY_OP_ON_TENSORS(OP)                                 \
  patterns.add<ArithUnaryOpOnTensorsConversion<OP>>(typeConverter,             \
                                                    patterns.getContext());

  POPULATE_ARITH_UNARY_OP_ON_TENSORS(arith::TruncFOp);
}

} // namespace npu
} // namespace triton
} // namespace mlir
