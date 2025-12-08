#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "Utility.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func.getSymName().ends_with("__compute")) {
      // Hack until we can fix CoreSpecialize
      LDBG("Deleting dot op from non-compute kernel");
      rewriter.replaceOp(op, adaptor.getC());
      return success();
    }

    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // add wait fronts so we know the CBs are ready
    Value numPages = arith::createConstantI32(loc, rewriter, 1);
    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getA(), numPages);
    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getB(), numPages);

    Value c0 = arith::createIndexConstant(loc, rewriter, 0);
    // Note that this must match the dest register index for PackTile
    Value destRegisterIndex = arith::createIndexConstant(loc, rewriter, 2);
    Value transpose = arith::createConstantI32(loc, rewriter, 0);
    ttkernel::MatmulTilesOp::create(rewriter, loc, adaptor.getA(),
                                    adaptor.getB(), c0, c0, destRegisterIndex,
                                    transpose);

    // foward the dot op c operand to the users of the dot op
    rewriter.replaceOp(op, adaptor.getC());
    return success();
  }
};

} // namespace

void populateDotOpConversionPattern(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<ConvertDotOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
