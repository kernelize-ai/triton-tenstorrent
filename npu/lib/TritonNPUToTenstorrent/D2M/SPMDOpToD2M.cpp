#include "PatternTritonNPUToD2M.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel
#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "../Utility.h"
#include "SPMDArgs.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {
namespace experimental {

namespace {

struct CurrentBlockConversion
    : public OpConversionPattern<cpu::CurrentBlockOp> {
  using OpConversionPattern<cpu::CurrentBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::CurrentBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

// Derive each core's grid-stride loop parameters from the Triton grid instead
// of passing block start/end/stride as per-kernel args. The Triton grid
// dimensions arrive as uniforms (SPMD args); we map that grid onto the device
// grid and have every core walk the full block range at a uniform stride
// equal to the core count, rather than owning a fixed contiguous sub-range.
// For a device grid {deviceGridWidth, deviceGridHeight} with numCores =
// deviceGridWidth * deviceGridHeight:
//
//   coreIndex   = MyLogicalX + MyLogicalY * deviceGridWidth  // linear core id
//   numBlocks   = tritonGridX * tritonGridY
//   blockStart  = coreIndex
//   blockStride = numCores
//   blockEnd    = numBlocks
//
// i.e. core `coreIndex` processes blocks coreIndex, coreIndex + numCores,
// coreIndex + 2*numCores, ... until reaching numBlocks. Every block in
// [0, numBlocks) is covered exactly once regardless of whether numBlocks is
// evenly divisible by numCores.

static Value computeCoreIndex(Location loc,
                              ConversionPatternRewriter &rewriter,
                              ModuleOp mod) {
  auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
  SmallVector<int64_t> deviceGrid = llvm::to_vector(gridAttr.getShape());
  assert(deviceGrid.size() == 2 && "expected rank-2 device grid");
  const int64_t deviceGridWidth = deviceGrid[0];

  Value xLogicalIndex = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(),
      ttkernel::MyLogicalXOp::create(rewriter, loc));
  Value yLogicalIndex = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(),
      ttkernel::MyLogicalYOp::create(rewriter, loc));
  // linear core ID on the device grid
  return arith::AddIOp::create(
      rewriter, loc, xLogicalIndex,
      arith::MulIOp::create(
          rewriter, loc, yLogicalIndex,
          arith::createConstantI32(loc, rewriter, deviceGridWidth)));
}

static Value computeNumCores(Location loc, ConversionPatternRewriter &rewriter,
                             ModuleOp mod) {
  auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
  SmallVector<int64_t> deviceGrid = llvm::to_vector(gridAttr.getShape());
  assert(deviceGrid.size() == 2 && "expected rank-2 device grid");
  return arith::createConstantI32(loc, rewriter, deviceGrid[0] * deviceGrid[1]);
}

static Value computeNumBlocks(Location loc, ConversionPatternRewriter &rewriter,
                              func::FuncOp func) {
  Value tritonGridX = getSpmdArg(func, SpmdArg::x_grid);
  Value tritonGridY = getSpmdArg(func, SpmdArg::y_grid);
  return arith::MulIOp::create(rewriter, loc, tritonGridX, tritonGridY);
}

struct BlockStartOpConversion : public OpConversionPattern<cpu::BlockStartOp> {
  using OpConversionPattern<cpu::BlockStartOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    rewriter.replaceOp(op, computeCoreIndex(op.getLoc(), rewriter, mod));
    return success();
  }
};

// Every core walks the same block range [0, numBlocks); the range itself
// does not depend on which core is asking, only the start offset and stride
// (see BlockStartOpConversion / BlockStrideOpConversion) do.
struct BlockEndOpConversion : public OpConversionPattern<cpu::BlockEndOp> {
  using OpConversionPattern<cpu::BlockEndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    assert(func && "expected op to have MLIR func dialect FuncOp parent during "
                   "Triton NPU to D2M op lowering");
    rewriter.replaceOp(op, computeNumBlocks(op.getLoc(), rewriter, func));
    return success();
  }
};

struct BlockStrideOpConversion
    : public OpConversionPattern<cpu::BlockStrideOp> {
  using OpConversionPattern<cpu::BlockStrideOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockStrideOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    rewriter.replaceOp(op, computeNumCores(op.getLoc(), rewriter, mod));
    return success();
  }
};

} // namespace

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<BlockStartOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockEndOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockStrideOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
