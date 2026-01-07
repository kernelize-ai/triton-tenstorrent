#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonNPUToTenstorrent.h"
#include "PointerInfoAnalysis.h"
#include "Utility.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel

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

struct RemoveLLVMAssume : OpConversionPattern<LLVM::AssumeOp> {
  using OpConversionPattern<LLVM::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

inline bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return false;
}

inline bool requiresSFPUInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<ttkernel::TTKernelBinaryOpTrait>()) {
      requiresInit = true;
      return;
    }
  });
  return requiresInit;
}

inline bool requiresMMInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](ttkernel::MatmulTilesOp dotOp) {
    requiresInit = true;
    return;
  });
  return requiresInit;
}

class InitializationHelper {
public:
  InitializationHelper(func::FuncOp F)
      : funcOp(F), addSFPUInit(requiresSFPUInit(F)),
        addMMInit(requiresMMInit(F)) {
    // build maps of copy/pack ops to respective indices
    funcOp.walk([&](Operation *op) {
      if (auto copyTileOp = dyn_cast<ttkernel::CopyTileOp>(op)) {
        copyTileOps[copyTileOp]++;
      } else if (auto copyTileInitOp = dyn_cast<ttkernel::CopyTileInitOp>(op)) {
        copyTileInitOps.push_back(copyTileInitOp);
      } else if (auto packTileOp = dyn_cast<ttkernel::PackTileOp>(op)) {
        packTileOps.insert(packTileOp);
      } else if (auto matmulTilesOp = dyn_cast<ttkernel::MatmulTilesOp>(op)) {
        matmulTilesOps.insert(matmulTilesOp);
      }
    });
  }

  void insertTileRegsAcquireOps() {
    // if copy tile init ops are non-empty then we can insert before the first
    // op. Otherwise, we need to find the block containing pack tile and move to
    // the start of that block.
    if (!copyTileInitOps.empty()) {
      OpBuilder builder(copyTileInitOps.front());
      ttkernel::TileRegsAcquireOp::create(builder,
                                          copyTileInitOps.front().getLoc());
    } else {
#if 1
      SmallVector<ttkernel::CBWaitFrontOp, 4> cbWaitFrontOps;
      funcOp.walk(
          [&](ttkernel::CBWaitFrontOp op) { cbWaitFrontOps.push_back(op); });
      assert(!cbWaitFrontOps.empty() &&
             "expecting at least one cb wait front op");
      auto firstWaitFrontOp = cbWaitFrontOps.front();
      Block *parentBlock = firstWaitFrontOp->getBlock();
      OpBuilder builder(parentBlock, parentBlock->begin());
      // put the tile regs acquire ops after any get arg val ops. This seems to
      // be mostly cosmetic.
      auto getArgValOpItr = parentBlock->getOps<ttkernel::GetArgValOp>();
      if (!getArgValOpItr.empty()) {
        auto argValOps = llvm::reverse(getArgValOpItr);
        builder.setInsertionPointAfter(*argValOps.begin());
      }
      ttkernel::TileRegsAcquireOp::create(builder, firstWaitFrontOp->getLoc());
#else
      SmallVector<ttkernel::PackTileOp, 4> packTileOps;
      funcOp.walk([&](ttkernel::PackTileOp op) { packTileOps.push_back(op); });
      assert(!packTileOps.empty() && "expecting at least one pack tile op");
      auto firstPackTileOp = packTileOps.front();
      Block *parentBlock = firstPackTileOp->getBlock();
      OpBuilder builder(parentBlock, parentBlock->begin());
      // put the tile regs acquire ops after any get arg val ops. This seems to
      // be mostly cosmetic.
      auto getArgValOpItr = parentBlock->getOps<ttkernel::GetArgValOp>();
      if (!getArgValOpItr.empty()) {
        auto argValOps = llvm::reverse(getArgValOpItr);
        builder.setInsertionPointAfter(*argValOps.begin());
      }
      ttkernel::TileRegsAcquireOp::create(builder, firstPackTileOp->getLoc());
#endif
    }
  }

  void insertComputeInitializationOps() {
    if (addMMInit) {
      // mm_init goes after cb initialization ops. but we need to collect the
      // circular buffers first
      assert(matmulTilesOps.size() == 1 &&
             "only single matmul supported currently");
      auto matmulTilesOp = *matmulTilesOps.begin();

      Location loc = matmulTilesOp.getLoc();
      OpBuilder builder(matmulTilesOp);

      Value aCb = matmulTilesOp.getIn0CbId();
      Value bCb = matmulTilesOp.getIn1CbId();
      ttkernel::PackTileOp packTileOp = *packTileOps.begin();
      Value outCb = packTileOp.getOutCb();

      auto getLatestValue = [](ArrayRef<Value> values) -> Value {
        Operation *latestOp = nullptr;
        Value latestValue;

        for (Value val : values) {
          Operation *defOp = val.getDefiningOp();
          if (!defOp)
            continue; // Skip block arguments

          if (!latestOp || latestOp->isBeforeInBlock(defOp)) {
            latestOp = defOp;
            latestValue = val;
          }
        }

        return latestValue;
      };

      Value latest = getLatestValue({aCb, bCb, outCb});
      builder.setInsertionPointAfterValue(latest);

      // TODO: support transpose. we cannot grab the transpose from the matmul
      // op as transpose could be defined inside the matmul loop
      Value transpose = arith::createConstantI32(loc, builder, 0);
      ttkernel::MatmulInitOp::create(builder, loc, aCb, bCb, outCb, transpose);
    }
    if (addSFPUInit) {
      SmallVector<ttkernel::TileRegsAcquireOp, 4> tileRegsAcquireOps;
      funcOp.walk([&](ttkernel::TileRegsAcquireOp op) {
        tileRegsAcquireOps.push_back(op);
      });
      assert(!tileRegsAcquireOps.empty() && "expecting tile regs acquire op");
      Operation *acquireOp = tileRegsAcquireOps.front();
      OpBuilder builder(acquireOp);
      ttkernel::PackTileOp packTileOp = *packTileOps.begin();
      Value inCb = copyTileOps.begin()->first.getCb0();
      Value outCb = packTileOp.getOutCb();

      ttkernel::InitSFPUOp::create(builder, acquireOp->getLoc(), inCb, outCb);
    }
  }

private:
  SmallVector<ttkernel::CopyTileInitOp, 4> copyTileInitOps;
  llvm::MapVector<ttkernel::CopyTileOp, unsigned> copyTileOps;
  SetVector<ttkernel::PackTileOp> packTileOps;
  SetVector<ttkernel::MatmulTilesOp> matmulTilesOps;

  func::FuncOp funcOp;
  const bool addSFPUInit;
  const bool addMMInit;
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
      auto convertShapeToTileShape = [](gpu::MemDescType type) {
        const auto &shape = type.getShape();
        SmallVector<int64_t, 4> tileShape;
        tileShape.reserve(shape.size());
        for (unsigned i = 0; i < shape.size(); ++i) {
          // TODO: support shape sizes < 32 by clamping to 1?
          if (shape[i] == 1) {
            tileShape.push_back(1);
            continue;
          }
          assert(shape[i] % 32 == 0 &&
                 "expecting shape dimensions to be multiple of 32");
          tileShape.push_back(shape[i] / 32);
        }
        return tileShape;
      };
      auto shape = convertShapeToTileShape(memdesc);
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
      auto convertShapeToTileShape = [](RankedTensorType type) {
        const auto &shape = type.getShape();
        SmallVector<int64_t, 4> tileShape;
        tileShape.reserve(shape.size());
        for (unsigned i = 0; i < shape.size(); ++i) {
          // TODO: support shape sizes < 32 by clamping to 1?
          if (shape[i] == 1) {
            tileShape.push_back(1);
            continue;
          }
          assert(shape[i] % 32 == 0 &&
                 "expecting shape dimensions to be multiple of 32");
          tileShape.push_back(shape[i] / 32);
        }
        return tileShape;
      };
      auto etype = type.getElementType();
      if (isa<triton::PointerType>(etype)) {
        return IntegerType::get(type.getContext(), 32);
      }
      if (isa<npu::tt::TileEncodingAttr>(type.getEncoding())) {
        auto shape = convertShapeToTileShape(type);
        auto ttcoreTileType = ttcore::TileType::get(
            type.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(etype));
        MemRefType cbMemRefType = MemRefType::get(
            shape, ttcoreTileType, MemRefLayoutAttrInterface{},
            ttcore::MemorySpaceAttr::get(type.getContext(),
                                         ttcore::MemorySpace::DeviceL1));
        return ttkernel::CBType::get(cbMemRefType);
      }
      if (auto dotOperandEncoding =
              dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                  type.getEncoding())) {
        // dot operands read directly from cbs, so convert to cb type
        assert(type.getShape().size() == 2 &&
               "expecting rank 2 tensor for dot operand");
        auto shape = convertShapeToTileShape(type);
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
    typeConverter.addSourceMaterialization(
        [](OpBuilder &builder, PointerType type, ValueRange inputs,
           Location loc) -> Value {
          return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
              .getResult(0);
        });

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    funcTarget.addLegalDialect<arith::ArithDialect>();
    funcTarget.addLegalDialect<func::FuncDialect>();
    funcTarget.addLegalOp<UnrealizedConversionCastOp>();
    funcTarget.addLegalOp<ttkernel::GetArgValOp>();

    mlir::RewritePatternSet funcPatterns(context);
    populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                    PatternBenefit(1));
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    LLVM_DEBUG({
      DBGS() << "After FuncOp conversion:\n";
      mod.dump();
    });

    npu::PointerInfoAnalysis pointerInfoAnalysis(mod);

    mlir::ConversionTarget target{*context};

    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::cpu::TritonCPUDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp funcOp) { return funcOp.getNumArguments() == 0; });
    target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *op) {
      // only legal if not operating on tensors
      return llvm::all_of(op->getOperands(), [](Value v) {
        return !(isa<RankedTensorType>(v.getType()));
      });
    });

    mlir::RewritePatternSet patterns(context);
    populateMemoryOpConversionPattern(typeConverter, patterns,
                                      &pointerInfoAnalysis, PatternBenefit(1));
    populateComputeOpConversionPattern(typeConverter, patterns,
                                       PatternBenefit(1));
    populateDotOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    populateElementwiseOpConversionPattern(typeConverter, patterns,
                                           PatternBenefit(1));
    populateMakeRangeOpConversionPattern(typeConverter, patterns,
                                         PatternBenefit(1));
    populateSPMDOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    populateViewOpConversionPattern(typeConverter, patterns, PatternBenefit(1));
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    patterns.add<RemoveLLVMAssume>(typeConverter, context);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // insert tile regs acquire before copy tile ops
    mod.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getSymName().ends_with("__compute"))
        return;

      InitializationHelper initHelper(funcOp);
      // TODO: re-enable (or delete?)
      initHelper.insertTileRegsAcquireOps();
      initHelper.insertComputeInitializationOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
