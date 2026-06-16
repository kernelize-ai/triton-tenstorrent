#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "Utility.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

namespace {

struct ConvertGetProgramIdOp : public OpConversionPattern<GetProgramIdOp> {
  using OpConversionPattern<GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError("Expected GetProgramId op to be replaced during "
                        "MakePersistentKernel pass");
  }
};

// Note: we currently ignore the axis attribute since the (x,y,z) grid is folded
// to (x,y) on the host
struct ConvertGetNumProgramsOp : public OpConversionPattern<GetNumProgramsOp> {
  using OpConversionPattern<GetNumProgramsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected FuncOp as a parent of GetProgramIdOp");
    auto commonArgsBase =
        funcOp->getAttrOfType<IntegerAttr>(kTTNumCommonArgsAttr).getInt();
    Value xGridIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kGridX);
    Value xGrid = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), xGridIndex);

    Value yGridIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kGridY);
    Value yGrid = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), yGridIndex);
    Value numBlocks = arith::MulIOp::create(rewriter, loc, xGrid, yGrid);
    rewriter.replaceOp(op, numBlocks);

    return success();
  }
};

struct BlockStartOpConversion : public OpConversionPattern<cpu::BlockStartOp> {
  using OpConversionPattern<cpu::BlockStartOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cpu::BlockStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected FuncOp as a parent of cpu::BlockStartOp");
    auto commonArgsBase =
        funcOp->getAttrOfType<IntegerAttr>(kTTNumCommonArgsAttr).getInt();

    Value xStrideIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kStrideX);
    Value xStride = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), xStrideIndex);

    Value yStrideIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kStrideY);
    Value yStride = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), yStrideIndex);

    Value xLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalXOp::create(rewriter, loc));
    Value yLogicalIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(),
        ttkernel::MyLogicalYOp::create(rewriter, loc));

    Value xStart = arith::MulIOp::create(rewriter, loc, xLogicalIndex, xStride);
    Value yStart = arith::MulIOp::create(rewriter, loc, yLogicalIndex, yStride);

    Value start = arith::AddIOp::create(rewriter, loc, xStart, yStart);
    rewriter.replaceOp(op, start);
    return success();
  }
};

struct BlockEndOpConversion : public OpConversionPattern<cpu::BlockEndOp> {
  using OpConversionPattern<cpu::BlockEndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::BlockEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected FuncOp as a parent of cpu::BlockStartOp");
    auto commonArgsBase =
        funcOp->getAttrOfType<IntegerAttr>(kTTNumCommonArgsAttr).getInt();

    Value xStrideIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kStrideX);
    Value xStride = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), xStrideIndex);

    Value xGridIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kGridX);
    Value xGrid = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), xGridIndex);

    Value yGridIndex = arith::createIndexConstant(
        loc, rewriter, commonArgsBase + GridArgffsets::kGridY);
    Value yGrid = ttkernel::GetCommonArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), yGridIndex);
    Value numBlocks = arith::MulIOp::create(rewriter, loc, xGrid, yGrid);

    Value blockStart = cpu::BlockStartOp::create(rewriter, loc);
    Value currentBlockEnd =
        arith::AddIOp::create(rewriter, loc, blockStart, xStride);

    Value end =
        arith::MinSIOp::create(rewriter, loc, currentBlockEnd, numBlocks);
    rewriter.replaceOp(op, end);
    return success();
  }
};

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

} // namespace

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ConvertGetProgramIdOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertGetNumProgramsOp>(typeConverter, patterns.getContext());
  patterns.add<BlockEndOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockStartOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
