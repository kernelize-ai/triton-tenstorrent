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

struct ConvertLocalLoadOp : public OpConversionPattern<gpu::LocalLoadOp> {
  using OpConversionPattern<gpu::LocalLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    llvm::errs() << "load src: " << op.getSrc() << "\n";
    llvm::errs() << "converted load src: " << adaptor.getSrc() << "\n";

    // 1. wait_front on the cb to know data is ready to load from SRAM
    Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    auto waitFrontOp =
        rewriter.create<ttkernel::CBWaitFrontOp>(loc, adaptor.getSrc(), c1);

    llvm::errs() << "adaptor src type: " << adaptor.getSrc().getType();
    assert(isa<MemRefType>(adaptor.getSrc().getType()) &&
           "expected memref type for type converted load src");
    auto cbTileMemref = cast<MemRefType>(adaptor.getSrc().getType());
    llvm::errs() << "cbTileMemref = " << cbTileMemref << "\n";
    // 1.5 reinterpret the cb from 2D to 1D tile shape
#if 1
    MemRefType oneDTileType = MemRefType::get(
        {1}, cbTileMemref.getElementType(), MemRefLayoutAttrInterface{},
        cbTileMemref.getMemorySpace());
    llvm::errs() << "oneDTileType = " << oneDTileType << "\n";
    auto oneDTile = rewriter
                        .create<ttkernel::CBReinterpretShapeOp>(
                            loc, oneDTileType, adaptor.getSrc())
                        .getResult();
#endif

    // 2. copy tile
    rewriter.create<ttkernel::CopyTileInitOp>(loc, oneDTile);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    // TODO: second c0 is wrong, needs to correspond with the set of registers
    // tile index which would be "0" for "a" and "1" for "b" - we need to encode
    // this somewhere in the layout
    rewriter.create<ttkernel::CopyTileOp>(loc, oneDTile, c0, c0);

    // erase the load if it is unused - otherwise let consumers erase it
    if (op->use_empty())
      op.erase();
    return success();
  }
};

struct ConvertLocalAllocOp : public OpConversionPattern<gpu::LocalAllocOp> {
  using OpConversionPattern<gpu::LocalAllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // replace the local allocs with cbs from the function signature
    auto allocIdx = op->getAttrOfType<IntegerAttr>("alloc_idx");
    if (!allocIdx) {
      return rewriter.notifyMatchFailure(op, "missing alloc_idx attribute");
    }
    int64_t allocIdxValue = allocIdx.getInt();

    auto typeConverter = getTypeConverter();
    llvm::errs() << "local alloc op type = " << op.getType() << "\n";
    auto cbMemRefType =
        cast<MemRefType>(typeConverter->convertType(op.getResult().getType()));
    llvm::errs() << "converted cbMemRefType = " << cbMemRefType << "\n";
    Type cbType = ttkernel::CBType::get(rewriter.getContext(), cbMemRefType);

    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(op, cbType,
                                                              allocIdxValue);

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
#if 1
    target.addDynamicallyLegalOp<gpu::LocalLoadOp>([](gpu::LocalLoadOp op) {
      auto funcOp = op->getParentOfType<triton::FuncOp>();
      assert(funcOp && "expected triton::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });
#endif
    target.addDynamicallyLegalOp<gpu::LocalAllocOp>([](gpu::LocalAllocOp op) {
      auto funcOp = op->getParentOfType<triton::FuncOp>();
      assert(funcOp && "expected triton::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](gpu::MemDescType memdesc) {
      // convert memdesc to memref
      // TODO: this currently blindly changes the shape from 1024 to the
      // ttcore::TileSize, but we should encode that info into the layout and
      // use the Triton memdesc provided shape instead.
      auto shape = SmallVector<int64_t>(2, 1);
      auto ttcoreTileType = tt::ttcore::TileType::get(
          memdesc.getContext(), ttcore::TileType::getDefaultShape(),
          ttcore::elementTypeToDataType(memdesc.getElementType()));

      return MemRefType::get(
          shape, ttcoreTileType, MemRefLayoutAttrInterface{},
          ttcore::MemorySpaceAttr::get(memdesc.getContext(),
                                       ttcore::MemorySpace::DeviceL1));
    });

    mlir::RewritePatternSet patterns(context);
    // Going to have to lower local load and store first...
    patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
