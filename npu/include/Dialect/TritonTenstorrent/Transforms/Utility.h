#ifndef TRITON_TENSTORRENT_UTILITY_H
#define TRITON_TENSTORRENT_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

class TensorDescriptorUnpacked {
public:
  TensorDescriptorUnpacked(TensorDescType type, ValueRange pack);

  Value generatePtr(OpBuilder &builder, const Location &loc,
                    ArrayRef<int64_t> blockShape, ValueRange offsets);

  Value generateMask(OpBuilder &builder, const Location &loc,
                     ArrayRef<int64_t> blockShape);

  Value getPtr() const { return base; }

  ValueRange getShape() const { return shape; }

protected:
  Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                    ArrayRef<int64_t> blockShape,
                                    ValueRange tileBaseOffsets,
                                    ValueRange offsetRanges);

  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
