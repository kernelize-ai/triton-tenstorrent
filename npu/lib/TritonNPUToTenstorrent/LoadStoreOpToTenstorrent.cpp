#include "PatternTritonNPUOpToTenstorrent.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct LocalAllocOpConversion
    : public OpConversionPattern<triton::gpu::LocalAllocOp> {
  explicit LocalAllocOpConversion(TypeConverter &typeConverter,
                                  MLIRContext *context, PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Lowering LocalAllocOp: " << op << "\n";
    unsigned allocIdx = cast<IntegerAttr>(op->getAttr("alloc_idx")).getInt();

    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();

    Value kernelArgOp;
    for (auto argOp : funcOp.getOps<npu::tt::KernelArgOp>()) {
      // TODO: assume the order of alloc idx matches the order of kernel args,
      // but this is fragile and should be made explicit
      if (argOp.getArgIndex() == allocIdx) {
        kernelArgOp = argOp;
      }
    }
    assert(kernelArgOp &&
           "expected kernel arg op with corresponding alloc idx");
#if 0
    // create an unrealized cast to TTKernel cb type
    auto kernelArgCB = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), /*TODO*/adaptor.getType(),
        kernelArgOp);
    auto untilizeInitOp = rewriter.create<tt::ttkernel::UntilizeInitOp>(op.getLoc(), kernelArgCB);
    llvm::errs() << "untilizeInitOp: " << untilizeInitOp << "\n";
    op.replaceAllUsesWith(kernelArgOp); // this is kind of hacky, but basically we re-associate the uses of the alloc to the argument (which will be a circular buffer) and let the local alloc serve as initialiation of the circular buffer landing registers
    op.erase();
    return success();
#else
    return failure();
#endif
  }
};

struct LoadOpConversion : public OpConversionPattern<LoadOp> {
  explicit LoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return failure();
  }
};

} // namespace

void mlir::triton::npu::tt::populateLoadStoreOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalAllocOpConversion>(typeConverter, patterns.getContext(),
                                       benefit);
  patterns.add<LoadOpConversion>(typeConverter, patterns.getContext(), benefit);
}
