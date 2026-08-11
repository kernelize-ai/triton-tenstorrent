#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "Utility.h"

#include "llvm/Support/Debug.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

#define S(v) StringAttr::get(context, (v))

// Per-operand tiling info shared by the binary/unary compute op lowerings:
// the tiled encoding (synthesized as a single tile if the operand type has
// none), and the register-tile linear layout derived from it.
struct TiledComputeLayout {
  npu::tt::TiledEncodingAttr tiledEncoding;
  ArrayRef<unsigned> tileShape;
  ArrayRef<unsigned> order;
  LinearLayout layout;
  SmallVector<int32_t> tilesPerCore;
  int32_t numTiles;
};

static TiledComputeLayout
getTiledComputeLayout(MLIRContext *context, RankedTensorType operandType) {
  auto tiledEncoding =
      dyn_cast<npu::tt::TiledEncodingAttr>(operandType.getEncoding());
  if (!tiledEncoding) {
    LDBG("Compute op has non-tiled type: " << operandType);
    // synthesize tiled encoding with 1 tile
    unsigned rank = operandType.getShape().size();
    SmallVector<unsigned> order(rank);
    SmallVector<unsigned> tileShape(rank, 32);
    if (rank > 1)
      std::iota(order.rbegin(), order.rend(), 0);
    else
      tileShape[0] *= 32;
    tiledEncoding = npu::tt::TiledEncodingAttr::get(
        context, /*tilesPerCore=*/SmallVector<unsigned>(rank, 1), order,
        tileShape);
    LDBG("Synthesized tiled encoding attr: " << tiledEncoding);
  }
  assert(tiledEncoding && "expecting tiled layouts for compute ops");

  TiledComputeLayout result;
  result.tiledEncoding = tiledEncoding;
  result.tileShape = tiledEncoding.getTileShape();
  result.order = tiledEncoding.getOrder();

  result.layout = gpu::toLinearLayout(operandType.getShape(), tiledEncoding);
  result.layout =
      result.layout.sublayout({S("register"), S("tile")},
                              llvm::to_vector(result.layout.getOutDimNames()));

  result.tilesPerCore = llvm::map_to_vector(
      result.layout.getOutDimSizes(), [](auto v) { return v / 32; });

  result.numTiles = result.layout.getInDimSize(S("tile"));
  return result;
}

// Computes the destination-register slot for tile `tileIndex`, honoring the
// tiled encoding's dimension order.
static int32_t computeTileSlot(MLIRContext *context,
                               const TiledComputeLayout &tl,
                               int32_t tileIndex) {
  auto crtIndex =
      tl.layout.apply({{S("tile"), tileIndex}, {S("register"), 0}});
  LLVM_DEBUG({
    DBGS() << "Tile " << tileIndex << " has start index: ";
    for (auto [dim, idx] : crtIndex) {
      DBGS() << dim.getValue() << ": " << idx << ", ";
    }
    DBGS() << "\n";
  });

  int32_t slot = 0;
  SmallVector<int32_t> localTiles(tl.tileShape.size());
  for (size_t d = 0; d < tl.tileShape.size(); ++d) {
    int32_t elem = crtIndex[d].second;
    localTiles[d] = elem / tl.tileShape[d];
  }

  // Linearize index based on order
  int32_t stride = 1;
  for (size_t d = 0; d < tl.tileShape.size(); ++d) {
    unsigned dim = tl.order[d];
    slot += localTiles[dim] * stride;
    stride *= tl.tilesPerCore[dim];
  }
  return slot;
}

struct ConvertBinaryComputeOp
    : public OpConversionPattern<npu::tt::BinaryComputeOp> {
  using OpConversionPattern<npu::tt::BinaryComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();
    int64_t lhsRegStart = lookupRegisterIndex(op.getLhs());
    int64_t rhsRegStart = lookupRegisterIndex(op.getRhs());
    int64_t destRegStart = lookupRegisterIndex(op->getResult(0));

    std::string opcode = op.getOpcode().str();
    if (failed(createInit(rewriter, loc, opcode))) {
      return failure();
    }

    auto operandType = cast<RankedTensorType>(op.getLhs().getType());
    TiledComputeLayout tl = getTiledComputeLayout(context, operandType);
    LDBG("Lowering compute op using layout: " << tl.layout);
    LDBG("Generating " << tl.numTiles << " tiled compute ops");

    // TODO: unify with MemoryOpToTTKernel lowering (to the extent possible,
    // we need something similar to applyLinearLayout from the
    // TritonGPUToLLVM side)
    for (int32_t i = 0; i < tl.numTiles; i++) {
      int32_t slot = computeTileSlot(context, tl, i);

      int64_t lhsReg = lhsRegStart + slot;
      int64_t rhsReg = rhsRegStart + slot;
      int64_t destReg = destRegStart + slot;

      Value lhs = arith::createIndexConstant(loc, rewriter, lhsReg);
      Value rhs = arith::createIndexConstant(loc, rewriter, rhsReg);
      Value dest = arith::createIndexConstant(loc, rewriter, destReg);

      if (failed(createOp(rewriter, loc, opcode, lhs, rhs, dest))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult createInit(ConversionPatternRewriter &rewriter, Location loc,
                           const std::string &opcode) const {
    // Initialize the binary tiles operation
    if (opcode == "arith.addf") {
      ttkernel::AddBinaryTilesInitOp::create(rewriter, loc);
    } else if (opcode == "arith.subf") {
      ttkernel::SubBinaryTilesInitOp::create(rewriter, loc);
    } else if (opcode == "arith.mulf") {
      ttkernel::MulBinaryTilesInitOp::create(rewriter, loc);
    } else if (opcode == "arith.divf") {
      ttkernel::DivBinaryTilesInitOp::create(rewriter, loc);
    } else if (opcode == "arith.maximumf") {
      ttkernel::BinaryMaxTileInitOp::create(rewriter, loc);
    } else if (opcode == "arith.minimumf") {
      ttkernel::BinaryMinTileInitOp::create(rewriter, loc);
    } else {
      LDBG("Unsupported opcode: " << opcode.c_str());
      return failure();
    }
    return success();
  }

  LogicalResult createOp(ConversionPatternRewriter &rewriter, Location loc,
                         const std::string &opcode, Value lhs, Value rhs,
                         Value dest) const {
    if (opcode == "arith.addf") {
      ttkernel::AddBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.subf") {
      ttkernel::SubBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.mulf") {
      ttkernel::MulBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.divf") {
      ttkernel::DivBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.maximumf") {
      ttkernel::BinaryMaxTileOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.minimumf") {
      ttkernel::BinaryMinTileOp::create(rewriter, loc, lhs, rhs, dest);
    } else {
      LDBG("Unsupported opcode: " << opcode.c_str());
      return failure();
    }
    return success();
  }
};

struct ConvertUnaryComputeOp
    : public OpConversionPattern<npu::tt::UnaryComputeOp> {
  using OpConversionPattern<npu::tt::UnaryComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(npu::tt::UnaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();
    int64_t srcRegStart = lookupRegisterIndex(op.getOperand());
    int64_t destRegStart = lookupRegisterIndex(op->getResult(0));

    std::string opcode = op.getOpcode().str();
    if (!isSupportedOpcode(opcode)) {
      LDBG("Unsupported unary opcode: " << opcode.c_str());
      return failure();
    }

    auto operandType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    TiledComputeLayout tl = getTiledComputeLayout(context, operandType);
    LDBG("Lowering unary compute op using layout: " << tl.layout);
    LDBG("Generating " << tl.numTiles << " tiled unary compute ops");

    auto dataFormat =
        ttcore::elementTypeToDataType(operandType.getElementType());
    auto resultDataFormat =
        ttcore::elementTypeToDataType(resultType.getElementType());
    const bool needsCopy = srcRegStart != destRegStart;

    // Unary SFPU ops are in-place; copy into dest slots first so SFPU init is
    // not clobbered by copy_dest_values_init.
    if (needsCopy)
      ttkernel::CopyDestValuesInitOp::create(rewriter, loc);

    SmallVector<Value> destVals;
    destVals.reserve(tl.numTiles);
    for (int32_t i = 0; i < tl.numTiles; i++) {
      int32_t slot = computeTileSlot(context, tl, i);

      int64_t srcReg = srcRegStart + slot;
      int64_t destReg = destRegStart + slot;
      Value dest = arith::createIndexConstant(loc, rewriter, destReg);

      if (needsCopy) {
        Value src = arith::createIndexConstant(loc, rewriter, srcReg);
        ttkernel::CopyDestValuesOp::create(rewriter, loc, src, dest,
                                           dataFormat);
      }
      destVals.push_back(dest);
    }

    createInit(rewriter, loc, opcode, dataFormat, resultDataFormat);
    for (Value dest : destVals)
      createOp(rewriter, loc, opcode, dest, dataFormat, resultDataFormat);

    rewriter.eraseOp(op);
    return success();
  }

  static bool isSupportedOpcode(const std::string &opcode) {
    return opcode == "math.absf" || opcode == "math.ceil" ||
           opcode == "math.floor" || opcode == "math.exp" ||
           opcode == "math.exp2" || opcode == "math.log" ||
           opcode == "math.rsqrt" || opcode == "math.sqrt" ||
           opcode == "math.sin" || opcode == "math.cos" ||
           opcode == "arith.truncf" || opcode == "arith.trunci";
  }

  void createInit(ConversionPatternRewriter &rewriter, Location loc,
                  const std::string &opcode, ttcore::DataType inDataFormat,
                  ttcore::DataType outDataFormat) const {
    if (opcode == "arith.truncf" || opcode == "arith.trunci") {
      ttkernel::TypecastTileInitOp::create(rewriter, loc, inDataFormat,
                                           outDataFormat);
    } else if (opcode == "math.absf") {
      ttkernel::AbsTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.ceil" || opcode == "math.floor") {
      ttkernel::RoundingTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.exp") {
      ttkernel::ExpTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.exp2") {
      ttkernel::Exp2TileInitOp::create(rewriter, loc);
    } else if (opcode == "math.log") {
      ttkernel::LogTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.rsqrt") {
      ttkernel::RsqrtTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.sqrt") {
      ttkernel::SqrtTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.sin") {
      ttkernel::SinTileInitOp::create(rewriter, loc);
    } else if (opcode == "math.cos") {
      ttkernel::CosTileInitOp::create(rewriter, loc);
    }
  }

  void createOp(ConversionPatternRewriter &rewriter, Location loc,
                const std::string &opcode, Value dest,
                ttcore::DataType inDataFormat,
                ttcore::DataType outDataFormat) const {
    if (opcode == "arith.truncf" || opcode == "arith.trunci") {
      ttkernel::TypecastTileOp::create(rewriter, loc, dest, inDataFormat,
                                       outDataFormat);
    } else if (opcode == "math.absf") {
      ttkernel::AbsTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.ceil") {
      ttkernel::CeilTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.floor") {
      ttkernel::FloorTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.exp") {
      ttkernel::ExpTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.exp2") {
      ttkernel::Exp2TileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.log") {
      ttkernel::LogTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.rsqrt") {
      ttkernel::RsqrtTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.sqrt") {
      ttkernel::SqrtTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.sin") {
      ttkernel::SinTileOp::create(rewriter, loc, dest);
    } else if (opcode == "math.cos") {
      ttkernel::CosTileOp::create(rewriter, loc, dest);
    }
  }
};

} // namespace

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertBinaryComputeOp, ConvertUnaryComputeOp>(
      typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
