#include "PatternTritonNPUToD2M.h"

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

    const int64_t gridWidth = deviceGrid[0];
    const int64_t gridHeight = deviceGrid[1];
    const int64_t numCores = gridWidth * gridHeight;

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
            arith::createConstantI32(loc, rewriter, gridWidth)));

    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    assert(func && "expected op to have MLIR func dialect FuncOp parent during "
                   "Triton NPU to D2M op lowering");
    Value tritonGridX = getSpmdArg(func, SpmdArg::x_grid);
    Value tritonGridY = getSpmdArg(func, SpmdArg::y_grid);
    Value numBlocks =
        arith::MulIOp::create(rewriter, loc, tritonGridX, tritonGridY);

    Value numCoresVal = arith::createConstantI32(loc, rewriter, numCores);
    Value baseBlocksPerCore =
        arith::FloorDivSIOp::create(rewriter, loc, numBlocks, numCoresVal);

    Value baseOffset =
        arith::MulIOp::create(rewriter, loc, coreIndex, baseBlocksPerCore);
    // leftover blocks => first coresWithExtraBlock cores get one more
    Value coresWithExtraBlock =
        arith::RemSIOp::create(rewriter, loc, numBlocks, numCoresVal);
    Value extraBlocksBeforeThisCore =
        arith::MinSIOp::create(rewriter, loc, coreIndex, coresWithExtraBlock);

    Value blockStart = arith::AddIOp::create(rewriter, loc, baseOffset,
                                             extraBlocksBeforeThisCore);

    rewriter.replaceOp(op, blockStart);

    return success();
  }
};

struct BlockEndOpConversion : public OpConversionPattern<cpu::BlockEndOp> {
  using OpConversionPattern<cpu::BlockEndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value constant = arith::createConstantI32(loc, rewriter, 1);
    rewriter.replaceOp(op, constant);
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
