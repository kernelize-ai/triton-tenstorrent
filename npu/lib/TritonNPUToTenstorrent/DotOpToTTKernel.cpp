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
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // TODO: add the matmul inits during post-processing since they must only be
    // run once per kernel

    llvm::errs() << "Converting DotOp: " << *op << "\n";

    llvm::errs() << "converted a = " << adaptor.getA() << "\n";
    llvm::errs() << "converted b = " << adaptor.getB() << "\n";
    llvm::errs() << "converted c = " << adaptor.getC() << "\n";

    Value c0 = arith::createIndexConstant(loc, rewriter, 0);
    Value transpose = arith::createConstantI1(loc, rewriter, false);
    ttkernel::MatmulTilesOp::create(rewriter, loc, adaptor.getA(),
                                    adaptor.getB(), c0, c0, c0, transpose);

    // don't replace the uses of the dot op since matmul tiles updates the
    // accumulator in place
    rewriter.eraseOp(op);
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
