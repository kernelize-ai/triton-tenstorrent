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
  TensorDescType type; // TODO remove and get from base value instead
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
  res.type = type;
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
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, indexRowType, offset);
  Value range = triton::MakeRangeOp::create(builder, loc, indexRowType, 0,
                                            blockShape[dim]);
  Value offsets = arith::AddIOp::create(builder, loc, splatOffset, range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}
#if 0
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
#endif

Value computeTileId(OpBuilder &builder, const Location &loc,
                    ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                    ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto i32Ty = builder.getIntegerType(32);

  SmallVector<Value, 4> blockShapeValues;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    blockShapeValues.push_back(arith::ConstantOp::create(
        builder, loc, i32Ty,
        IntegerAttr::get(i32Ty, static_cast<int32_t>(blockShape[i]))));
  }

  SmallVector<Value, 4> tileCoord;
  // tileCoord[i] = offset[i] / blockShape[i]
  for (unsigned i = 0; i < offsets.size(); ++i) {
    tileCoord.push_back(
        arith::DivSIOp::create(builder, loc, offsets[i], blockShapeValues[i]));
  }

  // tilesPerDim[i] = ceil(desc.shape[i] / blockShape[i])
  SmallVector<Value, 4> tilesPerDim;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    tilesPerDim.push_back(arith::CeilDivSIOp::create(
        builder, loc, desc.shape[i], blockShapeValues[i]));
  }

  // linearize the tileId
  // TODO: copy from Utility.h/linearize for LLVM
  Value tileId = tileCoord[0];
  for (unsigned i = 1; i < tileCoord.size(); i++) {
    tileId = arith::MulIOp::create(builder, loc, tileId, tilesPerDim[i]);
    tileId = arith::AddIOp::create(builder, loc, tileId, tileCoord[i]);
  }

#if 0
  int32_t numElems =
  static_cast<int32_t>(std::accumulate(blockShape.begin(), blockShape.end(), 1LL, std::multiplies<int64_t>()));
  Type elementType = cast<RankedTensorType>(desc.type.getBlockType()).getElementType();
  int32_t numBytes = elementType.getIntOrFloatBitWidth() / 8;
  Value offset = arith::MulIOp::create(builder, loc, tileId, arith::ConstantOp::create(builder, loc, i32Ty, IntegerAttr::get(i32Ty, numElems * numBytes)));
#endif
  return tileId;
}

// TODO: this isn't quite right because it doesn't take faces into account, but
// it should otherwise be fine for now
Value buildIntraTileLinearOffsets(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  ValueRange offsetRanges) {
  assert(blockShape.size() == offsetRanges.size());

  auto i32Ty = builder.getI32Type();
  auto fullTy = RankedTensorType::get(blockShape, i32Ty);

  Value lin = nullptr;

  for (unsigned d = 0; d < blockShape.size(); ++d) {
    Value idxFull =
        triton::BroadcastOp::create(builder, loc, fullTy, offsetRanges[d]);

    // Compute intra-tile coordinate: idx % blockShape[d]
    Value dimC = arith::ConstantOp::create(
        builder, loc, i32Ty, builder.getI32IntegerAttr(blockShape[d]));
    Value dimSplat = triton::SplatOp::create(builder, loc, fullTy, dimC);
    Value intra = arith::RemUIOp::create(builder, loc, idxFull, dimSplat);

    // Row-major fold: lin = lin * blockShape[d] + intra
    if (d == 0) {
      lin = intra;
    } else {
      Value scaleC = arith::ConstantOp::create(
          builder, loc, i32Ty, builder.getI32IntegerAttr(blockShape[d]));
      Value scaleSplat = triton::SplatOp::create(builder, loc, fullTy, scaleC);
      lin = arith::MulIOp::create(builder, loc, lin, scaleSplat);
      lin = arith::AddIOp::create(builder, loc, lin, intra);
    }
  }

  return lin;
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange tileBaseOffsets,
                                  ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());
  assert(blockShape.size() == tileBaseOffsets.size());

  auto i32Ty = builder.getI32Type();

  SmallVector<Value, 4> blockShapeValues;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    blockShapeValues.push_back(arith::ConstantOp::create(
        builder, loc, i32Ty,
        IntegerAttr::get(i32Ty, static_cast<int32_t>(blockShape[i]))));
  }

  // tileCoord[i] = tileBaseOffset[i] / blockShape[i]
  SmallVector<Value, 4> tileCoord;
  tileCoord.reserve(blockShape.size());
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    tileCoord.push_back(arith::DivSIOp::create(builder, loc, tileBaseOffsets[i],
                                               blockShapeValues[i]));
  }

  // tilesPerDim[i] = ceil(desc.shape[i] / blockShape[i])
  SmallVector<Value, 4> tilesPerDim;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    tilesPerDim.push_back(arith::CeilDivSIOp::create(
        builder, loc, desc.shape[i], blockShapeValues[i]));
  }

  // linearize the tileId
  // TODO: copy from Utility.h/linearize for LLVM
  Value tileId = tileCoord[0];
  for (unsigned i = 1; i < tileCoord.size(); i++) {
    tileId = arith::MulIOp::create(builder, loc, tileId, tilesPerDim[i]);
    tileId = arith::AddIOp::create(builder, loc, tileId, tileCoord[i]);
  }

  // tileElemOffset = tileId * elemsPerTile
  int32_t numElems = static_cast<int32_t>(std::accumulate(
      blockShape.begin(), blockShape.end(), 1LL, std::multiplies<int64_t>()));
  Value tileElemOffset = arith::MulIOp::create(
      builder, loc, tileId,
      arith::ConstantOp::create(builder, loc, i32Ty,
                                IntegerAttr::get(i32Ty, numElems)));

  // build intra-tile linear offsets tensors
  Value intraTileOffsets =
      buildIntraTileLinearOffsets(builder, loc, blockShape, offsetRanges);

  // compute final elem offset and ptr offset
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI32Type());
  Value tileElemOffsetSplat =
      triton::SplatOp::create(builder, loc, indexTensorType, tileElemOffset);
  Value elemOffsets = arith::AddIOp::create(builder, loc, tileElemOffsetSplat,
                                            intraTileOffsets);

  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);
  Value basePtrSplat =
      triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  Value ptrs = triton::AddPtrOp::create(builder, loc, ptrTensorType,
                                        basePtrSplat, elemOffsets);
  return ptrs;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
#if 1
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc, offsets,
                                     offsetRanges);
#else

#if 1
  Value offset =
      generatedTiledByteOffset(builder, loc, blockShape, desc, offsets);

  // splat the offset to the first dim then broadcast for remaining dims
  auto firstDimTensorType =
      RankedTensorType::get({blockShape[0]}, offset.getType());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, firstDimTensorType, offset);
  Value offsetTensor = expandOffsets(builder, loc, blockShape, splatOffset, 0);

  // TODO: we should really add individual offsets within the broader offsets
  // tensor, but for now we know we're going to drop the tensor in favor of the
  // initial offset value during later lowering. Also, we should respect the
  // tile-ized nfaces convention which is somewhat complicated

  // 2. splat + expand the base ptr
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);
  Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);

  // not right since we're missing broadcast and expand

  // TODO: directionally correct but we do need to return tensor here, so the
  // load input type (ptr type) matches the output type (tensor type)
  return triton::AddPtrOp::create(builder, loc, ptrType, ptr, offsetTensor);

#else
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
#endif
#endif
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
