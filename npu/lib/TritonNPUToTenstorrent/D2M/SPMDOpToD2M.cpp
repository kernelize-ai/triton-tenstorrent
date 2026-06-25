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

// Derive each core's block range from the Triton grid instead of passing block
// start/end as per-kernel args. The Triton grid dimensions arrive as uniforms
// (SPMD args); we map that grid onto the device grid and split blocks as evenly
// as possible across cores. For a device grid {deviceGridWidth,
// deviceGridHeight} with numCores = deviceGridWidth * deviceGridHeight:
//
//   coreIndex                 = MyLogicalX + MyLogicalY * deviceGridWidth  //
//   linear core id numBlocks                 = tritonGridX * tritonGridY
//   baseBlocksPerCore         = numBlocks / numCores
//   coresWithExtraBlock       = numBlocks % numCores
//   extraBlocksBeforeThisCore = min(coreIndex, coresWithExtraBlock)
//   blockStart = coreIndex * baseBlocksPerCore + extraBlocksBeforeThisCore
//
// The first coresWithExtraBlock cores take one extra block; the rest take
// baseBlocksPerCore, so every block in [0, numBlocks) is covered exactly once
struct BlockStartOpConversion : public OpConversionPattern<cpu::BlockStartOp> {
  using OpConversionPattern<cpu::BlockStartOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> deviceGrid = llvm::to_vector(gridAttr.getShape());
    assert(deviceGrid.size() == 2 && "expected rank-2 device grid");

    const int64_t deviceGridWidth = deviceGrid[0];
    const int64_t deviceGridHeight = deviceGrid[1];
    const int64_t numCoresConst = deviceGridWidth * deviceGridHeight;

    Value xLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalXOp::create(rewriter, loc));
    Value yLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalYOp::create(rewriter, loc));
    // linear core ID on the device grid
    Value coreIndex = arith::AddIOp::create(
        rewriter, loc, xLogicalIndex,
        arith::MulIOp::create(
            rewriter, loc, yLogicalIndex,
            arith::createConstantI32(loc, rewriter, deviceGridWidth)));

    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    assert(func && "expected op to have MLIR func dialect FuncOp parent during "
                   "Triton NPU to D2M op lowering");
    Value tritonGridX = getSpmdArg(func, SpmdArg::x_grid);
    Value tritonGridY = getSpmdArg(func, SpmdArg::y_grid);
    Value numBlocks =
        arith::MulIOp::create(rewriter, loc, tritonGridX, tritonGridY);

    Value numCoresVal = arith::createConstantI32(loc, rewriter, numCoresConst);
    Value baseBlocksPerCore =
        arith::FloorDivSIOp::create(rewriter, loc, numBlocks, numCoresVal);

    Value baseBlockStart =
        arith::MulIOp::create(rewriter, loc, coreIndex, baseBlocksPerCore);
    // leftover blocks => first coresWithExtraBlock cores get one more
    Value coresWithExtraBlock =
        arith::RemSIOp::create(rewriter, loc, numBlocks, numCoresVal);
    Value extraBlocksBeforeThisCore =
        arith::MinSIOp::create(rewriter, loc, coreIndex, coresWithExtraBlock);

    Value blockStart = arith::AddIOp::create(rewriter, loc, baseBlockStart,
                                             extraBlocksBeforeThisCore);

    rewriter.replaceOp(op, blockStart);

    return success();
  }
};

// blockEnd = blockStart + baseBlocksPerCore + (coreIndex < coresWithExtraBlock
// ? 1 : 0) Recomputes blockStart via cpu.block_start; the duplicated arithmetic
// CSEs away.
struct BlockEndOpConversion : public OpConversionPattern<cpu::BlockEndOp> {
  using OpConversionPattern<cpu::BlockEndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value blockStart = cpu::BlockStartOp::create(rewriter, loc);

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> deviceGrid = llvm::to_vector(gridAttr.getShape());
    assert(deviceGrid.size() == 2 && "expected rank-2 device grid");

    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    assert(func && "expected op to have MLIR func dialect FuncOp parent during "
                   "Triton NPU to D2M op lowering");
    Value tritonGridX = getSpmdArg(func, SpmdArg::x_grid);
    Value tritonGridY = getSpmdArg(func, SpmdArg::y_grid);
    Value numBlocks =
        arith::MulIOp::create(rewriter, loc, tritonGridX, tritonGridY);

    const int64_t deviceGridWidth = deviceGrid[0];
    const int64_t deviceGridHeight = deviceGrid[1];
    const int64_t numCoresConst = deviceGridWidth * deviceGridHeight;

    Value xLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalXOp::create(rewriter, loc));
    Value yLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalYOp::create(rewriter, loc));
    // linear core ID on the device grid
    Value coreIndex = arith::AddIOp::create(
        rewriter, loc, xLogicalIndex,
        arith::MulIOp::create(
            rewriter, loc, yLogicalIndex,
            arith::createConstantI32(loc, rewriter, deviceGridWidth)));

    Value numCoresVal = arith::createConstantI32(loc, rewriter, numCoresConst);
    Value baseBlocksPerCore =
        arith::FloorDivSIOp::create(rewriter, loc, numBlocks, numCoresVal);
    // leftover blocks => first coresWithExtraBlock cores get one more
    Value coresWithExtraBlock =
        arith::RemSIOp::create(rewriter, loc, numBlocks, numCoresVal);

    Value baseBlockEnd =
        arith::AddIOp::create(rewriter, loc, blockStart, baseBlocksPerCore);

    Value thisCoreGetsExtra =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                              coreIndex, coresWithExtraBlock);
    Value oneIfThisCoreGetsExtra =
        arith::SelectOp::create(rewriter, loc, thisCoreGetsExtra,
                                arith::createConstantI32(loc, rewriter, 1),
                                arith::createConstantI32(loc, rewriter, 0));

    Value blockEnd = arith::AddIOp::create(rewriter, loc, baseBlockEnd,
                                           oneIfThisCoreGetsExtra);

    rewriter.replaceOp(op, blockEnd);
    return success();
  }
};

} // namespace

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<BlockStartOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockEndOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
