#include "PatternTritonNPUToTenstorrent.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertPrintOp : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op.getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<ttkernel::DPrintOp>(op, op.getPrefix(),
                                                      ValueRange{});
      return success();
    }

    return failure();
  }
};

} // namespace

void populatePrintOpConversionPattern(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<ConvertPrintOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
