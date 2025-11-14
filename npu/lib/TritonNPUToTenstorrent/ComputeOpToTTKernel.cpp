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

struct ConvertBinaryComputeOp
    : public OpConversionPattern<npu::tt::BinaryComputeOp> {
  using OpConversionPattern<npu::tt::BinaryComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (op.getOpcode().str() != "arith.addf")
      return failure();

    ttkernel::AddBinaryTilesInitOp::create(rewriter, loc);
    Value lhsIndex = arith::createIndexConstant(loc, rewriter, 0);
    Value rhsIndex = arith::createIndexConstant(loc, rewriter, 1);
    Value destIndex = arith::createIndexConstant(loc, rewriter, 2);
    ttkernel::AddBinaryTilesOp::create(rewriter, loc, lhsIndex, rhsIndex,
                                       destIndex);

    rewriter.eraseOp(op);
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
