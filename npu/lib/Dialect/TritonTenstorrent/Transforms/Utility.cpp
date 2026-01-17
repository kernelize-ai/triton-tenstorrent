#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace npu {

Value getStoreLikeValue(Operation *op) {
  if (auto s = dyn_cast<triton::StoreOp>(op))
    return s.getValue();
  if (auto s = dyn_cast<triton::DescriptorStoreOp>(op))
    return s.getSrc();
  llvm_unreachable("not a store-like op");
}

void setStoreLikeValue(Operation *op, Value v) {
  if (auto s = dyn_cast<triton::StoreOp>(op)) {
    s.getValueMutable().assign(v);
    return;
  }
  if (auto s = dyn_cast<triton::DescriptorStoreOp>(op)) {
    s.getSrcMutable().assign(v);
    return;
  }
  llvm_unreachable("not a store-like op");
}

namespace {

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

} // namespace

Value TensorDescriptorUnpacked::generateBaseBlockOffset(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> blockShape,
    ValueRange tileBaseOffsets) {
  assert(blockShape.size() == shape.size());
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

  // tilesPerDim[i] = ceil(shape[i] / blockShape[i])
  SmallVector<Value, 4> tilesPerDim;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    tilesPerDim.push_back(arith::CeilDivSIOp::create(builder, loc, shape[i],
                                                     blockShapeValues[i]));
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

  return tileElemOffset;
}

Value TensorDescriptorUnpacked::generateOffsetsFromOffsetRanges(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> blockShape,
    ValueRange tileBaseOffsets, ValueRange offsetRanges) {
  assert(blockShape.size() == shape.size());
  assert(blockShape.size() == offsetRanges.size());
  assert(blockShape.size() == tileBaseOffsets.size());

  auto i32Ty = builder.getI32Type();

  Value tileElemOffset =
      generateBaseBlockOffset(builder, loc, blockShape, tileBaseOffsets);

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
  return elemOffsets;
}

Value TensorDescriptorUnpacked::generateBasePtr(OpBuilder &builder,
                                                const Location &loc,
                                                ArrayRef<int64_t> blockShape) {
  auto ptrType = cast<triton::PointerType>(base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);
  Value basePtrSplat =
      triton::SplatOp::create(builder, loc, ptrTensorType, base);
  return basePtrSplat;
}

Value TensorDescriptorUnpacked::generateOffsets(OpBuilder &builder,
                                                const Location &loc,
                                                ArrayRef<int64_t> blockShape,
                                                ValueRange offsets) {
  assert(blockShape.size() == shape.size());
  assert(blockShape.size() == offsets.size());

  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  Value elemOffsets = generateOffsetsFromOffsetRanges(builder, loc, blockShape,
                                                      offsets, offsetRanges);
  return elemOffsets;
}

Value TensorDescriptorUnpacked::generatePtr(OpBuilder &builder,
                                            const Location &loc,
                                            ArrayRef<int64_t> blockShape,
                                            ValueRange offsets) {

  Value elemOffsets = generateOffsets(builder, loc, blockShape, offsets);
  Value basePtrSplat = generateBasePtr(builder, loc, blockShape);

  Value ptrs = triton::AddPtrOp::create(builder, loc, basePtrSplat.getType(),
                                        basePtrSplat, elemOffsets);
  return ptrs;
}

Value TensorDescriptorUnpacked::generateMask(OpBuilder &builder,
                                             const Location &loc,
                                             ArrayRef<int64_t> blockShape) {
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  auto attr = builder.getIntegerAttr(builder.getI1Type(), 1);
  auto maskVal = SplatElementsAttr::get(maskTensorType, attr);
  Value mask = arith::ConstantOp::create(builder, loc, maskVal);
  return mask;
}

TensorDescriptorUnpacked::TensorDescriptorUnpacked(TensorDescType type,
                                                   ValueRange pack) {
  int rank = type.getBlockType().getRank();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 1 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by a padding option value.");
  base = pack[0];
  shape = pack.slice(1, rank);
  strides = pack.slice(1 + rank, rank);
  paddingOption = pack[1 + 2 * rank];
}

} // namespace npu
} // namespace triton
} // namespace mlir
