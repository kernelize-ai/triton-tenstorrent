#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

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
    auto tileShape = tiledEncoding.getTileShape();
    auto order = tiledEncoding.getOrder();

    auto layout = gpu::toLinearLayout(operandType.getShape(), tiledEncoding);
    layout = layout.sublayout({S("register"), S("tile")},
                              llvm::to_vector(layout.getOutDimNames()));
    LDBG("Lowering compute op using layout: " << layout);
    SmallVector tilesPerCore = llvm::map_to_vector(
        layout.getOutDimSizes(), [](auto v) { return v / 32; });

    int32_t numTiles = layout.getInDimSize(S("tile"));
    LDBG("Generating " << numTiles << " tiled compute ops");

    for (int32_t i = 0; i < numTiles; i++) {
      // TODO: unify with MemoryOpToTTKernel lowering (to the extent possible,
      // we need something similar to applyLinearLayout from the TritonGPUToLLVM
      // side)
      auto crtIndex = layout.apply({{S("tile"), i}, {S("register"), 0}});
      LLVM_DEBUG({
        DBGS() << "Tile " << i << " has start index: ";
        for (auto [dim, idx] : crtIndex) {
          DBGS() << dim.getValue() << ": " << idx << ", ";
        }
        DBGS() << "\n";
      });

      int32_t slot = 0;
      SmallVector<int32_t> localTiles(tileShape.size());
      for (size_t d = 0; d < tileShape.size(); ++d) {
        int32_t elem = crtIndex[d].second;
        localTiles[d] = elem / tileShape[d];
      }

      // Linearize index based on order
      int32_t stride = 1;
      for (size_t d = 0; d < tileShape.size(); ++d) {
        unsigned dim = order[d];
        slot += localTiles[dim] * stride;
        stride *= tilesPerCore[dim];
      }

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
    } else {
      LDBG("Unsupported opcode: " << opcode.c_str());
      return failure();
    }
    return success();
  }
};

} // namespace

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertBinaryComputeOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
