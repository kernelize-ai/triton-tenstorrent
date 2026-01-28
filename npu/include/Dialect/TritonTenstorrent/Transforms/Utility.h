#ifndef TRITON_TENSTORRENT_UTILITY_H
#define TRITON_TENSTORRENT_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

inline bool isLoadLike(Operation *op) {
  return isa<triton::LoadOp, triton::DescriptorLoadOp>(op);
}

inline bool isStoreLike(Operation *op) {
  return isa<triton::StoreOp, triton::DescriptorStoreOp>(op);
}

Value getStoreLikeValue(Operation *op);

void setStoreLikeValue(Operation *op, Value v);

class TensorDescriptorUnpacked {
public:
  TensorDescriptorUnpacked(TensorDescType type, ValueRange pack);

  Value generatePtr(OpBuilder &builder, const Location &loc,
                    ArrayRef<int64_t> blockShape, ValueRange offsets);

  Value generateMask(OpBuilder &builder, const Location &loc,
                     ArrayRef<int64_t> blockShape);

  Value generateBasePtr(OpBuilder &builder, const Location &loc,
                        ArrayRef<int64_t> blockShape);

  Value generateBaseBlockOffset(OpBuilder &builder, Location loc,
                                ArrayRef<int64_t> blockShape,
                                ValueRange tileBaseOffsets);

  Value generateOffsets(OpBuilder &builder, const Location &loc,
                        ArrayRef<int64_t> blockShape, ValueRange offsets);

  ValueRange getShape() const { return shape; }

  Value getPtr() const { return base; }

protected:
  Value generateOffsetsFromOffsetRanges(OpBuilder &builder, Location loc,
                                        ArrayRef<int64_t> blockShape,
                                        ValueRange tileBaseOffsets,
                                        ValueRange offsetRanges);

  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
};

FunctionOpInterface findComputeKernel(ModuleOp mod);

bool isComputeKernel(FunctionOpInterface func);

/// Walk up the stack to find the offset of the register buffer for the given
/// value.
int64_t lookupRegisterIndex(Value v);

/// Tile dim size
constexpr int64_t kTileDimSize = 32;

/// Calculate the number of tiles to cover a type shape
int64_t getNumTiles(Type type);

/// Convert a elements shape to a tiles shape
SmallVector<int64_t, 2> convertShapeToTileShape(ArrayRef<int64_t> shape);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
