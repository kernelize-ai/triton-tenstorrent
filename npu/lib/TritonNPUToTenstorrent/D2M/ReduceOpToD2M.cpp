#include "PatternTritonNPUToD2M.h"

#include <optional>

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Maps a ReduceOp's single combiner to the corresponding D2M tile reduce kind.
enum class ReduceKind { SumFloat, MaxFloat, SumInt, MaxInt };

/// Classify the given `ReduceOp` by the single combiner operation in its body.
/// Returns a `ReduceKind` if the operation is supported by Tenstorrent
/// `d2m.tile_reduce_*` operations, or `std::nullopt` otherwise.
static std::optional<ReduceKind> classifyReduceOp(triton::ReduceOp op,
                                                  Type elemType) {
  Operation *combiner = op.getSingleCombiner();
  if (!combiner)
    return std::nullopt;
  if (isa<FloatType>(elemType)) {
    if (isa<arith::AddFOp>(combiner))
      return ReduceKind::SumFloat;
    if (isa<arith::MaxNumFOp, arith::MaximumFOp>(combiner))
      return ReduceKind::MaxFloat;
  } else {
    if (isa<arith::AddIOp>(combiner))
      return ReduceKind::SumInt;
    if (isa<arith::MaxSIOp, arith::MaxUIOp>(combiner))
      return ReduceKind::MaxInt;
  }
  return std::nullopt;
}

/// Emit a constant value to initialize the output tile of a reduction, based on
/// the `ReduceKind`:
/// - `SumFloat` and `SumInt`: zero
/// - `MaxFloat`: negative infinity
/// - `MaxInt`: minimum representable integer
static Value initValue(OpBuilder &b, Location loc, ReduceKind reduceKind) {
  switch (reduceKind) {
  case ReduceKind::SumFloat:
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(b.getF32Type()))
        .getResult();
  case ReduceKind::SumInt:
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(b.getI32Type()))
        .getResult();
  case ReduceKind::MaxFloat:
    return b
        .create<arith::ConstantOp>(
            loc, b.getFloatAttr(b.getF32Type(),
                                -std::numeric_limits<float>::infinity()))
        .getResult();
  case ReduceKind::MaxInt:
    return b
        .create<arith::ConstantOp>(
            loc, b.getIntegerAttr(b.getI32Type(),
                                  std::numeric_limits<int32_t>::min()))
        .getResult();
  }
  llvm_unreachable("invalid ReduceKind");
}

/// Emit operations to fill a D2M tile with the value `1.0`.
static Value fillOnes(OpBuilder &b, Location loc, Type tileType) {
  assert(isa<FloatType>(cast<ttcore::TileType>(tileType).getElementType()) &&
         "fillOnes requires a float tile type");
  Value one =
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0f)).getResult();
  Value scalerTile = d2m::TileFillOp::create(b, loc, tileType, one).getResult();
  return scalerTile;
}

/// Emit a D2M tile reduction operation corresponding to the given `ReduceKind`.
/// This assumes we have already calculated the appropriate shapes for `srcTile`
/// and `dstTile` and the appropriate `reduceDimAttr` for the reduction axis.
///
/// Known reduction operations:
///   SumFloat -> d2m.tile_reduce_sum (with scaling tile B = tile_fill(1.0))
///   MaxFloat -> d2m.tile_reduce_max (with scaling tile B = tile_fill(1.0))
///   SumInt   -> d2m.tile_sfpu_reduce_sum
///   MaxInt   -> d2m.tile_sfpu_reduce_max
static Value reduceTile(OpBuilder &b, Location loc, Value srcTile,
                        Value dstTile, d2m::ReduceDimAttr reduceDimAttr,
                        ReduceKind reduceKind) {
  switch (reduceKind) {
  case ReduceKind::SumFloat: {
    auto ones = fillOnes(b, loc, srcTile.getType());
    return d2m::TileReduceSumOp::create(b, loc, dstTile.getType(), srcTile,
                                        ones, dstTile, reduceDimAttr)
        .getResult();
  }
  case ReduceKind::MaxFloat: {
    auto ones = fillOnes(b, loc, srcTile.getType());
    return d2m::TileReduceMaxOp::create(b, loc, dstTile.getType(), srcTile,
                                        ones, dstTile, reduceDimAttr)
        .getResult();
  }
  case ReduceKind::SumInt:
    return d2m::TileSFPUReduceSumOp::create(b, loc, dstTile.getType(), srcTile,
                                            dstTile, reduceDimAttr)
        .getResult();
  case ReduceKind::MaxInt:
    return d2m::TileSFPUReduceMaxOp::create(b, loc, dstTile.getType(), srcTile,
                                            dstTile, reduceDimAttr)
        .getResult();
  }
  llvm_unreachable("invalid ReduceKind");
}

/// Convert `tt.reduce` operation to Tenstorrent `d2m.tile_reduce_*` operations.
/// See
/// https://triton-lang.org/main/python-api/triton.language.html#reduction-ops.
struct ConvertReduceOp : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    if (adaptor.getOperands().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected single-operand reduce");

    int32_t axis = static_cast<int32_t>(op.getAxis());
    Value srcMemRef = adaptor.getOperands()[0];
    auto srcMemRefType = cast<MemRefType>(srcMemRef.getType());
    int64_t rank = srcMemRefType.getRank();
    auto tileType = cast<ttcore::TileType>(srcMemRefType.getElementType());

    auto reduceKind = classifyReduceOp(op, tileType.getElementType());
    if (!reduceKind)
      return rewriter.notifyMatchFailure(op, "unsupported reduce combiner");

    LDBG("ReduceOp: src=" << srcMemRefType << " axis=" << axis);

    // Map the Triton reduction axis to a D2M `ReduceDim` value (also see the
    // `TTIRToD2M::dimArgAsReduceDim` convention):
    // - RC: reduce both dimensions if we have a 1D tensor; values are striped
    // across the 2D tile
    // - R: reduce the last dimension (the "row" dimension in TTIR)
    // - C: reduce the second-to-last dimension (the "column" dimension in TTIR)
    assert(op.getSrcs().size() == 1);
    int64_t tensorRank =
        cast<RankedTensorType>(op.getSrcs()[0].getType()).getRank();
    LDBG("originalTensorRank=" << tensorRank);
    d2m::ReduceDim reduceDim;
    if (tensorRank == 1)
      reduceDim = d2m::ReduceDim::RC;
    else if (axis == tensorRank - 1)
      reduceDim = d2m::ReduceDim::R;
    else if (axis == tensorRank - 2)
      reduceDim = d2m::ReduceDim::C;
    else
      return rewriter.notifyMatchFailure(
          op, "only last two dims supported for tile-level reduce");

    // Output shape: RC reduces within the tile so the grid shape is unchanged;
    // R/C reduce drop the reduced axis from the grid shape.
    SmallVector<int64_t> outShape;
    if (reduceDim == d2m::ReduceDim::RC)
      outShape.assign(srcMemRefType.getShape().begin(),
                      srcMemRefType.getShape().end());
    else
      for (int64_t i = 0; i < rank; ++i)
        if (i != axis)
          outShape.push_back(srcMemRefType.getShape()[i]);

    // Allocate the output buffer. TODO: must we manually zero-initialise these
    // CBs or can we rely on hardware zeroing?
    auto outCBLayout = ttcore::CBLayoutAttr::get(outShape, tileType,
                                                 /*buffers=*/2);
    auto outMemRefType = MemRefType::get(outShape, tileType, outCBLayout,
                                         srcMemRefType.getMemorySpace());
    Value outMemRef =
        memref::AllocOp::create(rewriter, loc, outMemRefType).getResult();

    // Initialize output buffer with the correct initial value: the minimum
    // value for `max` and zero for `sum`.
    auto outRank = static_cast<int64_t>(outShape.size());
    linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{},
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{outMemRef},
        ArrayRef<AffineMap>{AffineMap::getMultiDimIdentityMap(outRank, ctx)},
        SmallVector<utils::IteratorType>(outRank,
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location innerLoc, ValueRange /*args*/) {
          Value init = initValue(b, innerLoc, *reduceKind);
          Value initTile =
              d2m::TileFillOp::create(b, innerLoc, tileType, init).getResult();
          linalg::YieldOp::create(b, innerLoc, initTile);
        });

    // Build indexing maps and iterator types.  For RC the grid iteration is
    // fully parallel (the within-tile reduction is encoded in reduceDim). For
    // R/C one grid dimension is a reduction iterator.
    AffineMap srcMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    AffineMap outMap;
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);
    if (reduceDim == d2m::ReduceDim::RC) {
      outMap = srcMap; // Same grid shape, no grid-level reduction.
    } else {
      SmallVector<AffineExpr> outExprs;
      for (int64_t i = 0; i < rank; ++i)
        if (i != axis)
          outExprs.push_back(getAffineDimExpr(i, ctx));
      outMap = AffineMap::get(rank, /*symbolCount=*/0, outExprs, ctx);
      iterTypes[axis] = utils::IteratorType::reduction;
    }
    linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{},
        /*inputs=*/ValueRange{srcMemRef},
        /*outputs=*/ValueRange{outMemRef}, ArrayRef<AffineMap>{srcMap, outMap},
        iterTypes, [&](OpBuilder &b_, Location innerLoc, ValueRange args) {
          // args[0] = src tile (A); args[1] = out tile (C accumulator)
          auto reduceDimAttr = d2m::ReduceDimAttr::get(ctx, reduceDim);
          Value result = reduceTile(b_, innerLoc, args[0], args[1],
                                    reduceDimAttr, *reduceKind);
          linalg::YieldOp::create(b_, innerLoc, result);
        });

    rewriter.replaceOp(op, outMemRef);
    return success();
  }
};

} // namespace

void populateReduceOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertReduceOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
