#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
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

struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
};

Descriptor unpackDescriptor(TensorDescType type, ValueRange pack) {
  int rank = type.getBlockType().getRank();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 1 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by a padding option value.");

  Descriptor res;
  res.base = pack[0];
  res.shape = pack.slice(1, rank);
  res.strides = pack.slice(1 + rank, rank);
  res.paddingOption = pack[1 + 2 * rank];
  return res;
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
#if 1
  // tenstorrent address space is 32-bits
  return values;
#else
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
#endif
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        triton::ExpandDimsOp::create(builder, loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, indexRowType, offset);
  Value range = triton::MakeRangeOp::create(builder, loc, indexI32RowType, 0,
                                            blockShape[dim]);
  // Value i64Range = arith::ExtSIOp::create(builder, loc, indexRowType, range);

  Value offsets = arith::AddIOp::create(builder, loc, splatOffset, range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI32Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = triton::SplatOp::create(
        builder, loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, offsets[i], splatStride);
    Value broadcasted = triton::BroadcastOp::create(
        builder, loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        triton::AddPtrOp::create(builder, loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

// TODO: generate masks if required - currently we assume the tensors are
// tileized and appropriated padded so the mask is always true
Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<int64_t> blockShape) {
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  auto attr = builder.getIntegerAttr(builder.getI1Type(), 1);
  auto maskVal = SplatElementsAttr::get(maskTensorType, attr);
  Value mask = arith::ConstantOp::create(builder, loc, maskVal);
  return mask;
}

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    auto blockTy = descTy.getSignlessBlockType();
    auto attr = rewriter.getZeroAttr(blockTy);
    auto other = arith::ConstantOp::create(rewriter, loc, attr);

    Value mask = generateMask(rewriter, loc, blockShape);

    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), mask, other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return success();
  }
};

struct RewriteStorePattern : OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    Value mask = generateMask(rewriter, loc, blockShape);

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
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

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::triton::TritonDialect>(
        [](mlir::Operation *op) {
          return !hasATensorDescriptorType(op->getOperandTypes()) &&
                 !hasATensorDescriptorType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into an pointer, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 32));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    mlir::RewritePatternSet patterns(context);

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns.add<RewriteLoadPattern, RewriteStorePattern>(converter, context);

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            mod, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
