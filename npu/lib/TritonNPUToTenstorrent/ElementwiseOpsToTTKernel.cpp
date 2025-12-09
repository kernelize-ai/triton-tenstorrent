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

    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.filter = [](Operation *op) {
      return !isa<IntegerType>(op->getResult(0).getType());
    };
    (void)getBackwardSlice(adaptor.getOffset(), &slice, opt);
    slice.insert(adaptor.getOffset().getDefiningOp());

    DenseMap<Value, Value> tensorToScalar;
    SetVector<Value> rewrittenOffsetOps;
    for (Operation *op : slice) {
      LDBG("Visiting " << *op);

      if (auto broadcastOp = dyn_cast<BroadcastOp>(op)) {
        Value src = tensorToScalar.count(broadcastOp.getSrc())
                        ? tensorToScalar[broadcastOp.getSrc()]
                        : broadcastOp.getSrc();
        tensorToScalar[broadcastOp.getResult()] = src;
      } else if (auto expandDimsOp = dyn_cast<ExpandDimsOp>(op)) {
        Value src = tensorToScalar.count(expandDimsOp.getSrc())
                        ? tensorToScalar[expandDimsOp.getSrc()]
                        : expandDimsOp.getSrc();
        tensorToScalar[expandDimsOp.getResult()] = src;
      } else if (auto makeRangeOp = dyn_cast<MakeRangeOp>(op)) {
        rewriter.setInsertionPointAfter(makeRangeOp);
        uint32_t start = makeRangeOp.getStart();
        Value c = arith::createConstantI32(loc, rewriter, start);
        tensorToScalar[makeRangeOp.getResult()] = c;
        rewrittenOffsetOps.insert(c);
      } else if (auto splatOp = dyn_cast<SplatOp>(op)) {
        tensorToScalar[splatOp.getResult()] = splatOp.getSrc();
        rewrittenOffsetOps.insert(splatOp.getSrc());
      } else if (auto unrealizedConversionCast =
                     dyn_cast<UnrealizedConversionCastOp>(op)) {
        // TODO: this seems a little suspect?
        Value src = tensorToScalar.count(unrealizedConversionCast.getOperand(0))
                        ? tensorToScalar[unrealizedConversionCast.getOperand(0)]
                        : unrealizedConversionCast.getOperand(0);
        tensorToScalar[unrealizedConversionCast.getResult(0)] = src;
        rewrittenOffsetOps.insert(src);
      } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        auto value = constOp.getValue();
        auto dense = dyn_cast<SplatElementsAttr>(value);
        if (dense) {
          APInt v = dense.getSplatValue<APInt>();
          Value c = arith::createConstantI32(loc, rewriter, v.getSExtValue());
          tensorToScalar[constOp.getResult()] = c;
          rewrittenOffsetOps.insert(c);
        }
      } else {
        Value result = op->getResult(0);

        IRMapping mapping;
        for (auto operand : op->getOperands()) {
          auto replacementItr = tensorToScalar.find(operand);
          if (replacementItr == tensorToScalar.end()) {
            LDBG("No replacement found for operand: " << operand);
            assert(!isa<RankedTensorType>(operand.getType()) &&
                   "expected to find replacement for tensor operand");
            continue;
          }
          LDBG("replace " << operand << " with " << replacementItr->second);
          mapping.map(operand, replacementItr->second);
        }
        rewriter.setInsertionPointAfter(op);
        auto newOp = rewriter.clone(*op, mapping);
        Value newVal = newOp->getResult(0);
        auto tensorType = dyn_cast<RankedTensorType>(newVal.getType());
        if (tensorType)
          newVal.setType(tensorType.getElementType());
        LDBG("Rewritten op: " << *newOp);
        tensorToScalar[result] = newVal;
        rewrittenOffsetOps.insert(newVal);
      }
    }
    Value offset = rewrittenOffsetOps.back();
    LDBG("Computed scalar offset: " << offset);

    // Drop the base addr and just return the offset converted to bytes
    auto tensorType = cast<RankedTensorType>(op.getPtr().getType());
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
    Value elemSizeValue = arith::createConstantI32(loc, rewriter, elemSize);
    offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);
    LDBG("Replace addptr with offset in bytes: " << offset);

    // for addptr ops with a loop carried ptr operand we want to keep the add
    // TODO: we need a better way of handling this generically - differentiating
    // between when the addptr operation happens within a TensorAccessor-style
    // object vs when we need to increment an integer value
    const bool userRequiresAdd = isa<BlockArgument>(adaptor.getPtr());
    if (userRequiresAdd) {
      // replace op with add
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getPtr(), offset);
    } else {
      // replace op with offset
      rewriter.replaceOp(op, offset);
    }

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
  POPULATE_ARITH_BINARY_OP_ON_TENSORS(arith::DivSIOp);

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
