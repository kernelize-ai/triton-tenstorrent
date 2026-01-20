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

namespace LaunchIDOffsets {
constexpr int kBlockStart = 0;
constexpr int kBlockEnd = 1;
constexpr int kThreadId = 2;

} // namespace LaunchIDOffsets

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
    Value paramIndexValue =
        arith::createIndexConstant(loc, rewriter, launchParamIndex + axis);
    auto launchParam = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), paramIndexValue);
    rewriter.replaceOp(op, launchParam);

    return success();
  }
};

template <typename OpTy>
class BlockIndexOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

public:
  explicit BlockIndexOpConversion(TypeConverter &typeConverter,
                                  const int funcArgIndexOffset,
                                  MLIRContext *context)
      : OpConversionPattern<OpTy>(typeConverter, context),
        funcArgIndexOffset(funcArgIndexOffset) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->template getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected FuncOp as a parent of BlockIndexOp");

    auto launchParamIndex =
        funcOp->template getAttrOfType<IntegerAttr>(kTTNumPerCoreArgsAttr)
            .getInt();
    Value paramIndexValue = arith::createIndexConstant(
        loc, rewriter, launchParamIndex + funcArgIndexOffset);
    auto launchParam = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), paramIndexValue);
    rewriter.replaceOp(op, launchParam);

    return success();
  }

private:
  const int funcArgIndexOffset;
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
  patterns.add<BlockIndexOpConversion<mlir::triton::cpu::BlockStartOp>>(
      typeConverter, LaunchIDOffsets::kBlockStart, patterns.getContext());
  patterns.add<BlockIndexOpConversion<mlir::triton::cpu::BlockEndOp>>(
      typeConverter, LaunchIDOffsets::kBlockEnd, patterns.getContext());
  patterns.add<CurrentBlockConversion>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
