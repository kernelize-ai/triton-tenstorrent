#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonNPUToTenstorrent.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

// Tenstorrent TTKernel includes
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOTTKERNEL
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // copy from cb into compute registers
    auto copyAndInitializeRegister = [&](Value operand,
                                         Value convertedOperand) {
      rewriter.create<ttkernel::CopyTileInitOp>(loc, convertedOperand);
      Value c0 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

      RankedTensorType loadType = cast<RankedTensorType>(operand.getType());
      // amusingly we could now deduce the load encoding from the operand index
      npu::tt::TileEncodingAttr loadEncoding =
          cast<npu::tt::TileEncodingAttr>(loadType.getEncoding());
      Value destRegisterIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  loadEncoding.getIndex()));
      rewriter.create<ttkernel::CopyTileOp>(loc, convertedOperand, c0,
                                            destRegisterIndex);
    };

    copyAndInitializeRegister(op.getLhs(), lhs);
    copyAndInitializeRegister(op.getRhs(), rhs);

    rewriter.create<ttkernel::AddBinaryTilesInitOp>(loc);
    Value lhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value rhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    Value destIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
    rewriter.create<ttkernel::AddBinaryTilesOp>(loc, lhsIndex, rhsIndex,
                                                destIndex);

    if (op->use_empty())
      rewriter.eraseOp(op);
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

    auto cbType = cast<ttkernel::CBType>(dst.getType());
    MemRefType cbTileMemref = cbType.getMemref();

    // add the one-D reinterpret cast at the compile time arg site for now
    rewriter.setInsertionPointAfter(dst.getDefiningOp());
    MemRefType oneDTileType = MemRefType::get(
        {1}, cbTileMemref.getElementType(), MemRefLayoutAttrInterface{},
        cbTileMemref.getMemorySpace());
    auto oneDTile = rewriter.create<ttkernel::CBReinterpretShapeOp>(
        loc, ttkernel::CBType::get(rewriter.getContext(), oneDTileType), dst);

    // reserve back the cb for the store
    Value numPages = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    rewriter.create<ttkernel::CBReserveBackOp>(loc, dst, numPages);

    rewriter.setInsertionPoint(op);
    Value destRegisterIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
    Value outIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    rewriter.create<ttkernel::PackTileOp>(loc, destRegisterIndex, oneDTile,
                                          outIndex,
                                          /*outOfOrder=*/true);

    auto srcOp = op.getSrc().getDefiningOp();
    op.erase();

    if (srcOp)
      rewriter.eraseOp(srcOp);
    return success();
  }
};

struct ReplaceFunctionArguments : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = funcOp.getLoc();
    auto typeConverter = getTypeConverter();

    for (int i = funcOp.getNumArguments() - 1; i >= 0; --i) {
      auto arg = funcOp.getArgument(i);
      if (arg.use_empty()) {
        (void)funcOp.eraseArgument(i);
        continue;
      }

      Type newType = typeConverter->convertType(arg.getType());
      Value argIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), i));
      rewriter.replaceAllUsesWith(
          arg, rewriter.create<ttkernel::GetArgValOp>(loc, newType, argIndex));
      //(void)funcOp.eraseArgument(i);
    }

    return success();
  }
};

struct ConvertLocalLoadOp : public OpConversionPattern<gpu::LocalLoadOp> {
  using OpConversionPattern<gpu::LocalLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. wait_front on the cb to know data is ready to load from SRAM
    Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    auto waitFrontOp =
        rewriter.create<ttkernel::CBWaitFrontOp>(loc, adaptor.getSrc(), c1);

    LDBG("Converted load src type = " << adaptor.getSrc().getType() << "\n");
    assert(isa<ttkernel::CBType>(adaptor.getSrc().getType()) &&
           "expected memref type for type converted load src");
    auto cbType = cast<ttkernel::CBType>(adaptor.getSrc().getType());
    MemRefType cbTileMemref = cbType.getMemref();

    // 1.5 reinterpret the cb from 2D to 1D tile shape
    // TODO: materialize this as a reshape op?
    MemRefType oneDTileType = MemRefType::get(
        {1}, cbTileMemref.getElementType(), MemRefLayoutAttrInterface{},
        cbTileMemref.getMemorySpace());
    auto oneDTile = rewriter.create<ttkernel::CBReinterpretShapeOp>(
        loc, ttkernel::CBType::get(rewriter.getContext(), oneDTileType),
        adaptor.getSrc());

    rewriter.replaceOp(
        op, oneDTile); // TODO: this should come from the cb root eventually,
                       // replaceOp is probably not appropriate here
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

    LDBG("Local alloc op result type: " << op.getResult().getType() << "\n");
    Type convertedType = typeConverter->convertType(op.getResult().getType());
    LDBG("is converted to memref type: " << convertedType << "\n");
    auto cbMemRefType = cast<ttkernel::CBType>(convertedType);

    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(op, cbMemRefType,
                                                              allocIdxValue);

    return success();
  }
};

static bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return isa<ttkernel::CBReinterpretShapeOp>(op);
}

class InitializationHelper {
public:
  struct ComputeOpInfo {
    ComputeOpInfo() {}
    ComputeOpInfo(Operation *op) : op(op) {}
    Value input;
    Value output;
    Operation *op;
  };

  InitializationHelper(func::FuncOp F) : funcOp(F) {}

  void insertTileRegsAcquireOps() {
    llvm::SetVector<Block *> visited;
    // TODO: this now visits copy tiles since the compute op operates directly
    // on the DST register. cleanup the naming (and possibly the algorithm)
    // here. We probably want to record some info about the compute op so we put
    // the right init in, but we might be able to do that elsewhere.
    funcOp.walk([&](ttkernel::CopyTileOp copyTilesOp) {
      Block *b = copyTilesOp->getBlock();
      if (visited.insert(b))
        insertTileRegsAcquireInBlock(b, ComputeOpInfo(copyTilesOp));
    });
  }

  void collectInputs() {
    for (auto &[_, computeOpInfo] : acquireRegistersToComputeOps) {

      auto firstComputeOperand = computeOpInfo.op->getOperand(0);
      Operation *op = firstComputeOperand.getDefiningOp();
      // TODO: I think should always have the cb feeding the compute op, but
      // leaving this somewhat generic until we confirm
      if (isCBOp(op)) {
        assert(op->getNumResults() == 1 && "expected single result for cb op");
        computeOpInfo.input = op->getResult(0);
        continue;
      }

      assert(computeOpInfo.input && "expected to find input value");
    }
  }

  void collectOutputs() {
    funcOp.walk([&](ttkernel::PackTileOp packTileOp) {
      // Note: this is completely disassociated from the compute op, which seems
      // wrong
      assert(acquireRegistersToComputeOps.size() == 1 &&
             "expecting single compute op");
      ComputeOpInfo &computeOpInfo =
          acquireRegistersToComputeOps.begin()->second;

      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      (void)getBackwardSlice(packTileOp, &backwardSlice, opt);

      for (auto op : llvm::reverse(backwardSlice)) {
        if (isCBOp(op)) {
          assert(op->getNumResults() == 1 &&
                 "expected single result for cb op");
          computeOpInfo.output = op->getResult(0);
          break;
        }
      }

      assert(computeOpInfo.output && "expected to find output value");
    });
  }

  void insertSFPUInitOps() {
    for (auto [acquireOp, computeOpInfo] : acquireRegistersToComputeOps) {
      OpBuilder builder(acquireOp);
      builder.setInsertionPoint(acquireOp);

      builder.create<ttkernel::InitSFPUOp>(
          acquireOp->getLoc(), computeOpInfo.input, computeOpInfo.output);
    }
  }

private:
  void insertTileRegsAcquireInBlock(Block *block, ComputeOpInfo computeOpInfo) {
    for (Operation &op : *block) {
      // acquire tile registers before the first copy tile
      if (isa<ttkernel::CopyTileInitOp>(op)) {
        OpBuilder builder(&op);
        builder.setInsertionPoint(&op);
        Operation *regsAcquireOp =
            builder.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
        acquireRegistersToComputeOps[regsAcquireOp] = computeOpInfo;
        return;
      }
    }
  }

  DenseMap<Operation *, ComputeOpInfo> acquireRegistersToComputeOps;
  func::FuncOp funcOp;
};

} // namespace

struct ConvertTritonNPUToTTKernelPass
    : public impl::ConvertTritonNPUToTTKernelBase<
          ConvertTritonNPUToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](PointerType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    typeConverter.addConversion([](gpu::MemDescType memdesc) {
      // convert memdesc to memref
      // TODO: this currently blindly changes the shape from 1024 to the
      // ttcore::TileSize, but we should encode that info into the layout and
      // use the Triton memdesc provided shape instead.
      auto shape = SmallVector<int64_t>(2, 1);
      auto ttcoreTileType = ttcore::TileType::get(
          memdesc.getContext(), ttcore::TileType::getDefaultShape(),
          ttcore::elementTypeToDataType(memdesc.getElementType()));

      MemRefType cbMemRefType = MemRefType::get(
          shape, ttcoreTileType, MemRefLayoutAttrInterface{},
          ttcore::MemorySpaceAttr::get(memdesc.getContext(),
                                       ttcore::MemorySpace::DeviceL1));
      return ttkernel::CBType::get(memdesc.getContext(), cbMemRefType);
    });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      if (isa<npu::tt::TileEncodingAttr>(type.getEncoding())) {
        // TODO: same caveats as above re:ttts layout
        auto shape = SmallVector<int64_t>(1, 1);
        auto ttcoreTileType = ttcore::TileType::get(
            type.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(type.getElementType()));
        MemRefType cbMemRefType = MemRefType::get(
            shape, ttcoreTileType, MemRefLayoutAttrInterface{},
            ttcore::MemorySpaceAttr::get(type.getContext(),
                                         ttcore::MemorySpace::DeviceL1));
        return ttkernel::CBType::get(type.getContext(), cbMemRefType);
      }
      return type;
    });

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addLegalDialect<func::FuncDialect>();
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    mlir::RewritePatternSet funcPatterns(context);
    populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                    PatternBenefit(1));
    if (applyPartialConversion(mod, funcTarget, std::move(funcPatterns))
            .failed())
      signalPassFailure();

    mlir::ConversionTarget target{*context};
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    target.addIllegalOp<npu::tt::BinaryComputeOp>();
    target.addDynamicallyLegalOp<gpu::LocalStoreOp>([](gpu::LocalStoreOp op) {
      auto funcOp = op->getParentOfType<func::FuncOp>();
      assert(funcOp && "expected func::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });
    target.addDynamicallyLegalOp<gpu::LocalLoadOp>([](gpu::LocalLoadOp op) {
      auto funcOp = op->getParentOfType<func::FuncOp>();
      assert(funcOp && "expected func::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });
    target.addDynamicallyLegalOp<gpu::LocalAllocOp>([](gpu::LocalAllocOp op) {
      auto funcOp = op->getParentOfType<func::FuncOp>();
      assert(funcOp && "expected func::funcOp parent");
      StringRef funcName = funcOp.getSymName();
      return !funcName.ends_with("__compute");
    });
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp funcOp) {
      StringRef funcName = funcOp.getSymName();
      if (!funcName.ends_with("__compute"))
        return true;

      return funcOp.getNumArguments() == 0;
    });

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertBinaryComputeOp>(typeConverter, patterns.getContext());
    patterns.add<ReplaceFunctionArguments>(typeConverter,
                                           patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();

    // insert tile regs acquire before copy tile ops
    mod.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getSymName().ends_with("__compute"))
        return;

      InitializationHelper initHelper(funcOp);
      initHelper.insertTileRegsAcquireOps();
      initHelper.collectInputs();
      initHelper.collectOutputs();
      initHelper.insertSFPUInitOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
