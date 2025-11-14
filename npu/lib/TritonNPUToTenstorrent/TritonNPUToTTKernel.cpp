#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonNPUToTenstorrent.h"
#include "Utility.h"

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

struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AddIOpConversion : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getLhs().getType()) &&
        isa<RankedTensorType>(op.getRhs().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct DropFunctionArguments : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = funcOp.getLoc();
    auto typeConverter = getTypeConverter();

    auto numArgs = funcOp.getNumArguments();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());

    for (auto arg : llvm::enumerate(funcOp.getArguments())) {
      Type newType = typeConverter->convertType(arg.value().getType());
      Value argIndex = arith::createIndexConstant(loc, rewriter, arg.index());
      auto getArgValOp =
          ttkernel::GetArgValOp::create(rewriter, loc, newType, argIndex);
      arg.value().replaceAllUsesWith(getArgValOp);
    }
    BitVector erasedArgs(numArgs, true);
    (void)funcOp.eraseArguments(erasedArgs);

    return success();
  }
};

template <typename OpTy>
struct DeadCodeEliminationOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

static bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return false;
}

static bool requiresSFPUInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<ttkernel::TTKernelBinaryOpTrait>()) {
      requiresInit = true;
      return;
    }
  });
  return requiresInit;
}

class InitializationHelper {
public:
  InitializationHelper(func::FuncOp F)
      : funcOp(F), addSFPUInit(requiresSFPUInit(F)) {
    // build maps of copy/pack ops to respective indices
    funcOp.walk([&](ttkernel::CopyTileOp copyTileOp) {
      Value cbIndex = copyTileOp.getTileIndexCb();
      Value dstIndex = copyTileOp.getTileIndexDst();
      copyTileOps[copyTileOp]++;
    });

    funcOp.walk([&](ttkernel::CopyTileInitOp copyTileInitOp) {
      copyTileInitOps.push_back(copyTileInitOp);
    });

    funcOp.walk([&](ttkernel::PackTileOp packTileOp) {
      Value cbIndex = packTileOp.getDstIndex();
      packTileOps[packTileOp]++;
    });
  }

  // tile regs aquire ops must be inserted before any copy tiles ops
  void insertTileRegsAcquireOps() {
    assert(!copyTileInitOps.empty() &&
           "expecting at least one copy tile init op");
    OpBuilder builder(copyTileInitOps.front());
    ttkernel::TileRegsAcquireOp::create(builder,
                                        copyTileInitOps.front().getLoc());
  }

  // coalesce copy tile waits before the firsts copy tile ops since tile
  // registers may not be acquired yet
  void insertCopyTileWaits() {
    OpBuilder builder(copyTileInitOps.front());
    for (auto copyTileOpItr : copyTileOps) {
      ttkernel::CopyTileOp copyTileOp = copyTileOpItr.first;
      Value numTiles = arith::createConstantI32(copyTileOp->getLoc(), builder,
                                                copyTileOpItr.second);
      Value cb = copyTileOp.getCb0();
      ttkernel::CBWaitFrontOp::create(builder, copyTileOp->getLoc(), cb,
                                      numTiles);
    }
  }

  void insertSFPUInitOps() {
    if (!addSFPUInit)
      return;

    auto tileRegsAcquireOps = funcOp.getOps<ttkernel::TileRegsAcquireOp>();
    assert(!tileRegsAcquireOps.empty() && "expecting tile regs acquire op");
    Operation *acquireOp = *tileRegsAcquireOps.begin();

    OpBuilder builder(acquireOp);
    Value inCb = copyTileOps.begin()->first.getCb0();
    Value outCb = packTileOps.begin()->first.getOutCb();

    ttkernel::InitSFPUOp::create(builder, acquireOp->getLoc(), inCb, outCb);
  }

private:
  SmallVector<ttkernel::CopyTileInitOp, 4> copyTileInitOps;
  llvm::MapVector<ttkernel::CopyTileOp, unsigned> copyTileOps;
  llvm::MapVector<ttkernel::PackTileOp, unsigned> packTileOps;

  func::FuncOp funcOp;
  const bool addSFPUInit;
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
      return ttkernel::CBType::get(cbMemRefType);
    });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      auto etype = type.getElementType();
      if (isa<triton::PointerType>(etype)) {
        etype = IntegerType::get(type.getContext(), 32);
        return RankedTensorType::get(type.getShape(), etype);
      }
      if (isa<npu::tt::TileEncodingAttr>(type.getEncoding())) {
        // TODO: same caveats as above re:ttts layout
        auto shape = SmallVector<int64_t>(1, 1);
        auto ttcoreTileType = ttcore::TileType::get(
            type.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(etype));
        MemRefType cbMemRefType = MemRefType::get(
            shape, ttcoreTileType, MemRefLayoutAttrInterface{},
            ttcore::MemorySpaceAttr::get(type.getContext(),
                                         ttcore::MemorySpace::DeviceL1));
        return ttkernel::CBType::get(cbMemRefType);
      }
      return type;
    });
    typeConverter.addConversion([](triton::PointerType type) -> Type {
      // convert pointer to i32
      return IntegerType::get(type.getContext(), 32);
    });

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addLegalDialect<func::FuncDialect>();
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    mlir::RewritePatternSet funcPatterns(context);
    populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                    PatternBenefit(1));
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    mlir::ConversionTarget target{*context};

    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp funcOp) { return funcOp.getNumArguments() == 0; });
    target.addDynamicallyLegalOp<arith::AddIOp>([&](arith::AddIOp op) {
      // only legal if not operating on tensors
      return !(isa<RankedTensorType>(op.getLhs().getType()) &&
               isa<RankedTensorType>(op.getRhs().getType()));
    });

    mlir::RewritePatternSet patterns(context);
    populateMemoryOpConversionPattern(typeConverter, patterns,
                                      PatternBenefit(1));
    populateComputeOpConversionPattern(typeConverter, patterns,
                                       PatternBenefit(1));
    populateElementwiseOpConversionPattern(typeConverter, patterns,
                                           PatternBenefit(1));
    populateSPMDOpConversionPattern(typeConverter, patterns, PatternBenefit(1));

    patterns.add<DropFunctionArguments>(typeConverter, patterns.getContext());
    patterns.add<SplatOpConversion>(typeConverter, patterns.getContext());
    patterns.add<AddIOpConversion>(typeConverter, patterns.getContext());
    patterns.add<MakeRangeOpConversion>(typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // insert tile regs acquire before copy tile ops
    mod.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getSymName().ends_with("__compute"))
        return;

      InitializationHelper initHelper(funcOp);
      initHelper.insertCopyTileWaits();
      initHelper.insertTileRegsAcquireOps();
      initHelper.insertSFPUInitOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
