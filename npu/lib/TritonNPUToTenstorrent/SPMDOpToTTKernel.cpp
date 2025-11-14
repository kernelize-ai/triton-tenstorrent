#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

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
    Location loc = op.getLoc();
    auto axis = adaptor.getAxis() == ProgramIDDim::X   ? 0
                : adaptor.getAxis() == ProgramIDDim::Y ? 1
                                                       : 2;

    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto launchParamIndex =
        funcOp->getAttrOfType<IntegerAttr>("tt.num_args").getInt();
    Value paramIndexValue =
        arith::createIndexConstant(loc, rewriter, launchParamIndex + axis);
    auto launchParam = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), paramIndexValue);
    rewriter.replaceOp(op, launchParam);

    return success();
  }
};

} // namespace

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ConvertGetProgramIdOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
