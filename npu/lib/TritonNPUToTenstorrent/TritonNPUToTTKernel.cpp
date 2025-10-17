#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOTTKERNEL
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

namespace {

struct ConvertAddOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // create init op
    rewriter.create<ttkernel::AddTilesInitOp>(loc, lhs, rhs);

    Value lhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value rhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    Value destIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
    rewriter.create<ttkernel::AddTilesOp>(loc, lhs, rhs, lhsIndex, rhsIndex,
                                          destIndex);

    if (op->use_empty())
      op.erase();
    return success();
  }
};

struct ConvertLocalStoreOp : public OpConversionPattern<gpu::LocalStoreOp> {
  using OpConversionPattern<gpu::LocalStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dst = adaptor.getDst();
    llvm::errs() << "converted dst = " << dst << "\n";

    Value destRegisterIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
    Value outIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    rewriter.create<ttkernel::PackTileOp>(loc, destRegisterIndex, dst, outIndex,
                                          /*outOfOrder=*/true);

    auto srcOp = op.getSrc().getDefiningOp();
    op.erase();

    // erase any unused defining ops (tile commit relationship is implicit
    // through cb refs)
    if (srcOp && srcOp->use_empty())
      srcOp->erase();

    return success();
  }
};

} // namespace

struct ConvertTritonNPUToTTKernelPass
    : public impl::ConvertTritonNPUToTTKernelBase<
          ConvertTritonNPUToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addLegalDialect<tt::ttkernel::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();

#if 1
    target.addDynamicallyLegalOp<arith::AddFOp>(
        [](arith::AddFOp op) { return !isa<RankedTensorType>(op.getType()); });
#endif

    target.addDynamicallyLegalOp<gpu::LocalStoreOp>([](gpu::LocalStoreOp op) {
      auto funcOp = op->getParentOfType<triton::FuncOp>();
      assert(funcOp && "expected triton::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    mlir::RewritePatternSet patterns(context);
    // Going to have to lower local load and store first...
    patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
