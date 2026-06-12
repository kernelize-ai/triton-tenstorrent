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
    llvm_unreachable("unhandled binary_compute kind");
  }
};

} // namespace

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertBinaryComputeOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
