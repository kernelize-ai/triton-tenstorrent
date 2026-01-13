#include <vector>

#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"

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

}

LinearLayout tt::TiledEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto order = getOrder();
  assert(shape.size() == order.size());
  auto rank = shape.size();
  // (block, tile, register) -> (x, y) 

#if 1
    // TODO: use nfaces for accurate register mapping inside a tile (for now we only operate on register granularity so this is fine)
    SmallVector<unsigned> normalizedShape(shape.begin(), shape.end());
#if 1
    for (int i = 0; i < shape.size(); i++) {
        normalizedShape[i] = normalizedShape[i] / getTilesPerCore()[i];
    }
#endif 
    llvm::errs() << "normalizedShape = ";
    for (auto v : normalizedShape) {
        llvm::errs() << v << ", ";
    }
    llvm::errs() << "\n";
    SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
    
#if 1
    LinearLayout ret = identityStandardND(S("register"), getTileShape(), order);
    llvm::errs() << "reg layout: " << ret << "\n";
#if 1
    LinearLayout tile = identityStandardND(S("tile"), getTilesPerCore(), order);
#else
    LinearLayout tile = LinearLayout::empty();
    assert(rank == getTilesPerCore().size());
    assert(rank <= getTileShape().size()); // should be ok as long as order is valid
    for (unsigned i = 0; i < rank; i++) {
        auto dim = order[i];
        tile *= LinearLayout::strided1D(getTilesPerCore()[dim], getTileShape()[dim], S("tile"), outDimNames[dim]);
    }
#endif
    llvm::errs() << "tile layout: " << tile << "\n";

    // currently not splitting blocks 
    SmallVector<unsigned> blockShape(rank, 1);
    ret *= tile * identityStandardND(S("block"), blockShape, order);
#else
    SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

    LinearLayout reg = LinearLayout::empty();
    reg *= LinearLayout::identity1D(getTileShape()[order[0]], S("register"), outDimNames[order[0]]);
    llvm::errs() << "reg checkpoint 1: " << reg << "\n";
    reg *= LinearLayout::identity1D(getTileShape()[order[1]], S("register"), outDimNames[order[1]]);
    llvm::errs() << "reg checkpoint 2: " << reg << "\n";

    LinearLayout ret = reg *
                             identityStandardND(S("tile"), getTilesPerCore(), order) *
                             identityStandardND(S("block"), normalizedShape, order);
#endif 
    //llvm::errs() << "TiledEncodingAttr toLinearLayout: " << ret << "\n";
#else

  assert(shape.size() == order.size());
  auto rank = shape.size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ctaLayout = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    // TODO: use nfaces for accurate register mapping inside a tile (for now we only operate on register granularity so this is fine)
    ctaLayout *= LinearLayout::identity1D(getRegistersPerTile()[dim], S("register"),
                                          outDimNames[dim]);
    ctaLayout *= LinearLayout::identity1D(getTilesPerCore()[dim], S("tile"),
                                          outDimNames[dim]);
    ctaLayout *= LinearLayout::identity1D(shape[dim], S("block"), outDimNames[dim]);
  }

  llvm::errs() << "TiledEncodingAttr toLinearLayout: " << ctaLayout << "\n";
#endif
  return ret.transposeOuts(outDimNames);
}

LinearLayout tt::TiledDotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
    int rank = shape.size();
    auto tiled = cast<tt::TiledEncodingAttr>(getParent());
    MLIRContext *ctx = getContext();

    auto kDimIdx = getOpIdx() == 0 ? rank - 1 : rank - 2;

    auto tileSizes = llvm::to_vector(tiled.getTilesPerCore());
    tileSizes[kDimIdx] = shape[kDimIdx] / tiled.getTileShape()[kDimIdx];
    
    auto order = llvm::to_vector(tiled.getOrder());
    SmallVector<StringAttr> repDimNames =
      permuteDimNames(standardOutDimNames(ctx, rank), order);

    auto registersLayout = identityStandardND(S("register"), tiled.getTileShape(), order)
                                 * identityStandardND(S("tile"), tileSizes, order);

    // TODO: definitely wrong
    return registersLayout; 
}
    
}
}
}
