#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel
#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "D2M/SPMDArgs.h"
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
    Location loc = op.getLoc();
    auto axis = adaptor.getAxis() == ProgramIDDim::X   ? 0
                : adaptor.getAxis() == ProgramIDDim::Y ? 1
                                                       : 2;

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected FuncOp as a parent of GetProgramIdOp");
    auto launchParamIndex =
        funcOp->getAttrOfType<IntegerAttr>(kTTNumPerCoreArgsAttr).getInt();
    // TODO: this needs to support multiple dimensions for each per core arg
    // offset
    Value paramIndexValue =
        arith::createIndexConstant(loc, rewriter, launchParamIndex + axis);
    auto launchParam = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), paramIndexValue);
    rewriter.replaceOp(op, launchParam);

    return success();
  }
};

struct ConvertGetNumProgramsOp : public OpConversionPattern<GetNumProgramsOp> {
  using OpConversionPattern<GetNumProgramsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "expected FuncOp as a parent of GetProgramIdOp");
    auto perCoreArgsBase =
        funcOp->getAttrOfType<IntegerAttr>(kTTNumPerCoreArgsAttr).getInt();
    // TODO: this needs to support multiple dimensions for each per core arg
    // offset
    Value index = arith::createIndexConstant(
        loc, rewriter, perCoreArgsBase + PerCoreArgOffsets::kNumBlocks);
    auto runtimeArgVal = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), index);
    rewriter.replaceOp(op, runtimeArgVal);

    return success();
  }
};
// Derive each core's grid-stride loop parameters the same way as the D2M
// pipeline (see D2M/SPMDOpToD2M.cpp): every core walks the *entire* Triton
// grid at a uniform stride equal to the physical core count, instead of
// owning a fixed contiguous sub-range baked as a per-core runtime arg.
//
//   coreIndex   = MyLogicalX + MyLogicalY * deviceGridWidth  // linear core id
//   numBlocks   = tritonGridX * tritonGridY
//   blockStart  = coreIndex
//   blockStride = numCores
//   blockEnd    = numBlocks
//
// Unlike D2M -- where x_grid/y_grid arrive as trailing function arguments --
// this pipeline's converted functions take no arguments at all (see
// ConvertTritonFunc in TritonFuncOpToFuncOp.cpp): every value is read out of
// the common/per-core runtime arg arrays via GetCommonArgValOp/GetArgValOp.
// x_grid and y_grid are common args (identical on every core), so they are
// read here as two extra common args living right after the user args
// counted by kTTNumCommonArgsAttr.
static Value getSpmdCommonArg(Location loc, ConversionPatternRewriter &rewriter,
                              func::FuncOp func, SpmdArg a) {
  auto numUserArgs =
      func->getAttrOfType<IntegerAttr>(kTTNumCommonArgsAttr).getInt();
  Value argIndex =
      arith::createIndexConstant(loc, rewriter, numUserArgs + (int)a);
  return ttkernel::GetCommonArgValOp::create(rewriter, loc,
                                             rewriter.getI32Type(), argIndex);
}

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
  Value tritonGridX = getSpmdCommonArg(loc, rewriter, func, SpmdArg::x_grid);
  Value tritonGridY = getSpmdCommonArg(loc, rewriter, func, SpmdArg::y_grid);
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
    auto func = op->getParentOfType<func::FuncOp>();
    assert(func && "expected op to have MLIR func dialect FuncOp parent during "
                   "Triton NPU to TTKernel op lowering");
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
  patterns.add<BlockStartOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockEndOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BlockStrideOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
