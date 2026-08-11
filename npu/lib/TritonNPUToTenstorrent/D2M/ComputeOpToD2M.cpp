#include "PatternTritonNPUToD2M.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertBinaryComputeOp
    : public OpConversionPattern<npu::tt::BinaryComputeOp> {
  using OpConversionPattern<npu::tt::BinaryComputeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto cbType = cast<MemRefType>(lhs.getType());
    Value out = memref::AllocOp::create(rewriter, loc, cbType);

    unsigned rank = cbType.getRank();
    AffineMap id = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(/*2 ins + 1 out=*/3, id);
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{out}, indexingMaps, iterators,
        [&](OpBuilder &b, Location l, ValueRange tiles) {
          Value res = emitTileOp(b, l, op.getOpcode(), tiles[0], tiles[1]);
          linalg::YieldOp::create(b, l, res);
        });
    rewriter.replaceOp(op, out);

    return success();
  }

  static Value emitTileOp(OpBuilder &b, Location l, StringRef kind, Value a,
                          Value c) {
    Type t = a.getType(); // !ttcore.tile
    if (kind == "arith.addf")
      return d2m::TileAddOp::create(b, l, t, a, c);
    if (kind == "arith.mulf")
      return d2m::TileMulOp::create(b, l, t, a, c);
    if (kind == "arith.subf")
      return d2m::TileSubOp::create(b, l, t, a, c);
    if (kind == "arith.maximumf")
      return d2m::TileMaximumOp::create(b, l, t, a, c);
    if (kind == "arith.minimumf")
      return d2m::TileMinimumOp::create(b, l, t, a, c);
    llvm_unreachable("unhandled binary_compute kind");
  }
};

struct ConvertUnaryComputeOp
    : public OpConversionPattern<npu::tt::UnaryComputeOp> {
  using OpConversionPattern<npu::tt::UnaryComputeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(npu::tt::UnaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperand();
    auto cbType = cast<MemRefType>(input.getType());

    // The result element type may differ from the operand's (e.g. truncf /
    // trunci narrow the bitwidth), so the output CB must be allocated with
    // the converted result type rather than assumed to match the input.
    Type outType = getTypeConverter()->convertType(op->getResult(0).getType());
    auto outMemRefType = cast<MemRefType>(outType);
    Value out = memref::AllocOp::create(rewriter, loc, outMemRefType);

    unsigned rank = cbType.getRank();
    AffineMap id = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(/*1 in + 1 out=*/2, id);
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{out}, indexingMaps, iterators,
        [&](OpBuilder &b, Location l, ValueRange tiles) {
          Value res =
              emitTileOp(b, l, op.getOpcode(), tiles[0], tiles[1].getType());
          linalg::YieldOp::create(b, l, res);
        });
    rewriter.replaceOp(op, out);

    return success();
  }

  static Value emitTileOp(OpBuilder &b, Location l, StringRef kind, Value a,
                          Type resultType) {
    if (kind == "arith.truncf" || kind == "arith.trunci")
      return d2m::TileTypecastOp::create(b, l, resultType, a);
    Type t = a.getType(); // !ttcore.tile
    if (kind == "math.absf")
      return d2m::TileAbsOp::create(b, l, t, a);
    if (kind == "math.ceil")
      return d2m::TileCeilOp::create(b, l, t, a);
    if (kind == "math.floor")
      return d2m::TileFloorOp::create(b, l, t, a);
    if (kind == "math.exp")
      return d2m::TileExpOp::create(b, l, t, a);
    if (kind == "math.exp2")
      return d2m::TileExp2Op::create(b, l, t, a);
    if (kind == "math.log")
      return d2m::TileLogOp::create(b, l, t, a);
    if (kind == "math.rsqrt")
      return d2m::TileRsqrtOp::create(b, l, t, a);
    if (kind == "math.sqrt")
      return d2m::TileSqrtOp::create(b, l, t, a);
    if (kind == "math.sin")
      return d2m::TileSinOp::create(b, l, t, a);
    if (kind == "math.cos")
      return d2m::TileCosOp::create(b, l, t, a);
    llvm_unreachable("unhandled unary_compute kind");
  }
};

} // namespace

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertBinaryComputeOp, ConvertUnaryComputeOp>(
      typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
