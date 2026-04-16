#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

// TODO: move (at some point)
struct ConvertTiledConstantOp : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultTensorType)
      return failure();

    if (!isa<npu::tt::TiledEncodingAttr>(resultTensorType.getEncoding()))
      return failure();

    // TODO: check if the constant is 0

    // convert to alloc
    Type newType = getTypeConverter()->convertType(op.getType());
    auto allocOp = memref::AllocOp::create(rewriter, op.getLoc(),
                                           cast<MemRefType>(newType));
    rewriter.replaceOp(op, allocOp.getResult());

    return success();
  }
};

struct ConvertTruncateOp : public OpConversionPattern<arith::TruncFOp> {
  using OpConversionPattern<arith::TruncFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getOperand().getType())) {
      Type newType = getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<d2m::TileTypecastOp>(op, newType,
                                                       adaptor.getIn());
      return success();
    }
    return failure();
  }
};

struct ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Parse the result encoding to determine tile iteration bounds.
    auto resultType = cast<RankedTensorType>(op.getType());
    auto enc = dyn_cast<npu::tt::TiledEncodingAttr>(resultType.getEncoding());
    if (!enc)
      return rewriter.notifyMatchFailure(
          op, "dot op result type must have TiledEncodingAttr for conversion "
              "to D2M");

    auto tilesPerBlock = enc.getTilesPerCore();
    auto tileShape = enc.getTileShape();

    // K-tiles: read from the original (pre-conversion) A type.
    // A has shape [M, K] in elements; divide by tile width to get K in tiles.
    auto aType = cast<RankedTensorType>(op.getA().getType());
    int64_t kTiles = aType.getShape()[1] / static_cast<int64_t>(tileShape[1]);

    LDBG("DotOp tile bounds: M=" << tilesPerBlock[0] << " N="
                                 << tilesPerBlock[1] << " K=" << kTiles);

    // Build matmul indexing maps over the (d0=M, d1=N, d2=K) tile space:
    //   A[d0, d2], B[d2, d1], C[d0, d1]
    auto d0 = getAffineDimExpr(0, ctx);
    auto d1 = getAffineDimExpr(1, ctx);
    auto d2 = getAffineDimExpr(2, ctx);
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {d0, d2}, ctx),
        AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {d2, d1}, ctx),
        AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {d0, d1}, ctx),
    };
    SmallVector<utils::IteratorType> iterTypes = {
        utils::IteratorType::parallel,  // d0 (M)
        utils::IteratorType::parallel,  // d1 (N)
        utils::IteratorType::reduction, // d2 (K)
    };

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{},
        /*inputs=*/ValueRange{a, b},
        /*outputs=*/ValueRange{c}, indexingMaps, iterTypes,
        [&](OpBuilder &b_, Location innerLoc, ValueRange args) {
          // args[0] = tile element from A
          // args[1] = tile element from B
          // args[2] = tile element from C (accumulator)
          auto tileResult = d2m::TileMatmulOp::create(
              b_, innerLoc, args[2].getType(), args[0], args[1], args[2]);
          linalg::YieldOp::create(b_, innerLoc, tileResult.getResult());
        });

    rewriter.replaceOp(op, c);
    return success();
  }
};

} // namespace

void populateDotOpConversionPattern(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<ConvertTiledConstantOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertTruncateOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertDotOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
