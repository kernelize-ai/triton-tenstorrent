#include <vector>

#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

namespace {

#define S(v) StringAttr::get(ctx, (v))

// TODO: de-dupe with upstream
/// Function to generate lane and warp layout for dot operands.
static LinearLayout broadcastedDotOperandLayout(MLIRContext *ctx,
                                                ArrayRef<unsigned> shape,
                                                ArrayRef<unsigned> order,
                                                unsigned kDim,
                                                StringAttr inDimName) {
  // Let warpsPerCTAMma = {2, 2}, then
  // warpsPerCTA = {2, 1} for opA and warpsPerCTA = {1, 2} for opB
  // assume warpOrder = {1, 0}
  // Assume that C is tiled by 2x2 tiles. Since warpOrder={1, 0}, we have that
  // the C is owned as per the following layout:
  // C: 0 | 1
  //    - | -
  //    2 | 3
  // In order to be able to compute C, we need the following warp tiling of
  // A and B:
  // A: 0 1 | 0 1    B: 0 2 | 1 3
  //    - - | - -       - - | - -
  //    2 3 | 2 3       0 2 | 1 3
  // In other words, we need to broadcast along K
  auto rank = shape.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout layout = LinearLayout::empty();

  // We have to broadcast along the inner dimension
  // For A, when moving along M we go from 0 to 2.
  // For B, when moving along N we go from 0 to 1.
  // As such, choosing the order of A {1, 0}, gives us the correct broadcasting
  // Same happens if the warpOrder is {0, 1}, like in Hopper
  for (auto d : order) {
    if (d == kDim) {
      layout *= LinearLayout::zeros1D(shape[d], inDimName, dimNames[d]);
    } else {
      layout *= LinearLayout::identity1D(shape[d], inDimName, dimNames[d]);
    }
  }
  return layout;
}

// TODO Have order be a mandatory argument of standardOutDimNames.
SmallVector<StringAttr> permuteDimNames(const SmallVector<StringAttr> &names,
                                        const SmallVector<unsigned> &order) {
  assert(names.size() == order.size());
  SmallVector<StringAttr> ret;
  for (unsigned i : order) {
    ret.push_back(names[i]);
  }
  return ret;
}

// Tenstorrent-specific helpers
LinearLayout unitDimGpuStandard(MLIRContext *ctx, ArrayRef<unsigned> order) {
  SmallVector<unsigned> unitShape(order.size(), 1);
  return identityStandardND(S("lane"), unitShape, order) *
         identityStandardND(S("warp"), unitShape, order) *
         identityStandardND(S("block"), unitShape, order);
}

} // namespace

LinearLayout
tt::TiledEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto order = getOrder();
  assert(shape.size() == order.size());
  auto rank = shape.size();
  // (block, tile, register) -> (x, y)

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  // TODO: use nfaces for accurate register mapping inside a tile (for now we
  // only operate on register granularity so this is fine)
  LinearLayout registerLayout =
      identityStandardND(S("register"), getTileShape(), order);
  LinearLayout tileLayout =
      identityStandardND(S("tile"), getTilesPerCore(), order);

  // currently not splitting blocks
  LinearLayout ret =
      registerLayout * tileLayout * unitDimGpuStandard(ctx, order);

  return ret.transposeOuts(outDimNames);
}

LinearLayout
tt::TiledDotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();

  auto tiled = cast<tt::TiledEncodingAttr>(getParent());
  auto order = tiled.getOrder();
  auto dotOrder = gpu::getOrderForDotOperand(getOpIdx(), shape.size(),
                                             /*kContig=*/true);
  assert(order.size() == dotOrder.size());
  auto rank = order.size();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout registerLayout =
      identityStandardND(S("register"), tiled.getTileShape(), order);
  LinearLayout tileLayout =
      identityStandardND(S("tile"), tiled.getTilesPerCore(), order);

  // currently not splitting blocks
  LinearLayout ret =
      registerLayout * tileLayout * unitDimGpuStandard(ctx, order);

  return ret.transposeOuts(outDimNames);
}

} // namespace npu
} // namespace triton
} // namespace mlir
