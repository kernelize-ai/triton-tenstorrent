#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // BlockIndexOps from MakePersistentKernel

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "../PatternTritonNPUToTenstorrent.h"
#include "../TypeConverter.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOD2M
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline bool isCBAlloc(memref::AllocOp alloc) {
  auto memRefType = cast<MemRefType>(alloc.getType());
  return isa<ttcore::CBLayoutAttr>(memRefType.getLayout());
}

static LogicalResult hoistCBAllocs(func::FuncOp func) {
  // collect generic ops up front then rewrite each generic in order
  SmallVector<d2m::GenericOp> generics;
  func.walk([&](d2m::GenericOp g) { generics.push_back(g); });

  IRRewriter rewriter(func.getContext());
  for (auto generic : generics) {
    // 1. collect CB Allocs for hoisting
    SmallVector<memref::AllocOp> cbAllocs;
    generic->walk([&](memref::AllocOp alloc) {
      if (isCBAlloc(alloc))
        cbAllocs.push_back(alloc);
    });
    if (cbAllocs.empty())
      continue;

    // 2. Hoist each CB alloc in collection order
    rewriter.setInsertionPoint(generic);
    for (memref::AllocOp alloc : cbAllocs)
      alloc->moveBefore(generic);

    // 3. Rebuild the generic op, appending each CB alloc to additionalArgs.
    SmallVector<Value> newAdditionalArgs(generic.getAdditionalArgs().begin(),
                                         generic.getAdditionalArgs().end());
    for (memref::AllocOp alloc : cbAllocs)
      newAdditionalArgs.push_back(alloc.getResult());

    assert(generic.getNumRegions() == 1 &&
           "expected explicit form d2m generic to have a single unified thread "
           "region");

    auto newGeneric = d2m::GenericOp::create(
        rewriter, generic.getLoc(), generic.getResultTypes(),
        generic.getInputs(), generic.getOutputs(), newAdditionalArgs,
        generic.getGrid(), generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(), generic.getThreads(),
        generic.getFabricConnectionConfigAttr(),
        /*regionsCount=*/generic.getNumRegions());

    // add body arguments for the newly added cb alloc operands
    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }

    Block *originalBlock = &originalRegion.front();
    Block *newBlock = &newGeneric.getRegion(0).emplaceBlock();

    // Only semaphore-typed block args carry over; CB allocs (additionalArgs)
    // are operand-list declarations on the op, not block args. References to
    // them inside the body resolve via outer-scope SSA (d2m.generic is not
    // IsolatedFromAbove), and the cloned body inherits those references
    // automatically without any mapping entry.
    IRMapping mapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      BlockArgument arg = originalBlock->getArgument(i);
      assert(mlir::isa<d2m::LocalSemaphoreType>(arg.getType()) &&
             "region block arguments must be of local semaphore type");
      mapping.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));
    }

    // clone ops
    rewriter.setInsertionPointToStart(newBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // TODO: cloning the terminator separately matching the convention in
    // D2MSplitUnifiedThreadRewriter, but since we are in unified thread mode I
    // wonder if we clone the terminator in the above loop.
    if (originalBlock->mightHaveTerminator()) {
      Operation *term = originalBlock->getTerminator();
      rewriter.setInsertionPointToEnd(newBlock);
      rewriter.clone(*term, mapping);
    }

    rewriter.eraseOp(generic);
  }

  return success();
}

/// Calculate how a tensor is sharded across tiles. For 1D tensors, we stripe
/// elements across both tile dimensions (rows * cols). For 2D tensors, we
/// divide each tensor dimension by the corresponding tile dimension.
SmallVector<int64_t> calculateShardShape(RankedTensorType tensorType,
                                         ttcore::TileType tileType) {
  if (tensorType.getRank() == 1) {
    auto dim = tensorType.getShape()[0];
    auto rows = tileType.getShape()[0];
    auto cols = tileType.getShape()[1];
    assert(dim % (rows * cols) == 0 &&
           "tensor dimension must be a multiple of tile size");
    return {dim / (rows * cols)};
  } else if (tensorType.getRank() == 2) {
    return llvm::to_vector(map_range(
        (llvm::zip(tensorType.getShape(), tileType.getShape())),
        [](auto pair) -> int64_t {
          auto [dim, tileDim] = pair;
          assert(dim % tileDim == 0 &&
                 "tensor dimension must be a multiple of tile dimension");
          return dim / tileDim;
        }));
  } else {
    llvm::report_fatal_error(Twine("unsupported tensor rank = ") +
                             std::to_string(tensorType.getRank()));
  }
}

} // namespace

struct ConvertTritonNPUToD2MPass
    : public impl::ConvertTritonNPUToD2MBase<ConvertTritonNPUToD2MPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonNPUToTenstorrentTypeConverter typeConverter(context);
    typeConverter.addConversion([](mlir::triton::TensorDescType t,
                                   llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into a memref, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      auto eType = tensorType.getElementType();
      auto tileType = ttcore::TileType::get(
          t.getContext(), ttcore::TileType::getDefaultShape(),
          ttcore::elementTypeToDataType(eType));
      SmallVector<int64_t> shardShape =
          calculateShardShape(tensorType, tileType);

      SmallVector<int64_t> shape(tensorType.getRank(), 1);
      shape.append(shardShape.begin(), shardShape.end());
      auto memRefType = MemRefType::get(
          shape, tileType,
          ttcore::ShardLayoutAttr::get(shardShape, tileType, /*buffers=*/1),
          ttcore::MemorySpaceAttr::get(t.getContext(),
                                       ttcore::MemorySpace::DeviceDRAM));
      out.push_back(memRefType);
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 32));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });
    typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
      auto eType = tensorType.getElementType();
      if (auto ptrType = dyn_cast<triton::PointerType>(eType)) {
        eType = ptrType.getPointeeType();
      }
      if (!tensorType.getEncoding()) {
        return RankedTensorType::get(tensorType.getShape(), eType);
      }
      if (isa<npu::tt::TiledEncodingAttr>(tensorType.getEncoding()) ||
          isa<npu::tt::TiledDotOperandEncodingAttr>(tensorType.getEncoding()) ||
          isa<gpu::DotOperandEncodingAttr>(tensorType.getEncoding()) ||
          isa<gpu::BlockedEncodingAttr>(tensorType.getEncoding())) {
        // Convert to memref in L1.
        auto tileType = ttcore::TileType::get(
            tensorType.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(eType));
        SmallVector<int64_t> shardShape =
            calculateShardShape(tensorType, tileType);

        // Assume all L1 allocations are in CBs.
        auto cbLayout =
            ttcore::CBLayoutAttr::get(tensorType.getContext(), shardShape,
                                      ttcore::getElementSizeBytes(tileType),
                                      /*buffers=*/shardShape.size());
        auto memRefType = MemRefType::get(
            shardShape, tileType, cbLayout,
            ttcore::MemorySpaceAttr::get(tensorType.getContext(),
                                         ttcore::MemorySpace::DeviceL1));
        return memRefType;
      }
      return tensorType;
    });

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    funcTarget.addLegalOp<UnrealizedConversionCastOp>();
    funcTarget.addLegalOp<d2m::GenericOp>();
    funcTarget.addLegalOp<ttir::TTNNMetalLayoutCastOp>();
    funcTarget.addLegalDialect<func::FuncDialect>();

    mlir::RewritePatternSet funcPatterns(context);
    experimental::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                                  PatternBenefit(1));

    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    LLVM_DEBUG({
      DBGS() << "After FuncOp conversion:\n";
      mod.dump();
    });

    mlir::ConversionTarget target{*context};

    target.addLegalDialect<d2m::D2MDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::cpu::TritonCPUDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *op) {
      if (llvm::isa<arith::ConstantOp>(op)) {
        // allow constant ops as long as they do not produce tensor types
        return llvm::all_of(op->getResults(), [](Value v) {
          return !isa<RankedTensorType>(v.getType());
        });
      }
      // other arith ops are only legal if not operating on tensors
      return llvm::all_of(op->getOperands(), [](Value v) {
        return !(isa<RankedTensorType>(v.getType()));
      });
    });

    mlir::RewritePatternSet patterns(context);
    experimental::populateDotOpConversionPattern(typeConverter, patterns,
                                                 PatternBenefit(1));
    experimental::populateMemoryOpConversionPattern(typeConverter, patterns,
                                                    PatternBenefit(1));
    experimental::populateSPMDOpConversionPattern(typeConverter, patterns,
                                                  PatternBenefit(1));
    experimental::populateReduceOpConversionPattern(typeConverter, patterns,
                                                    PatternBenefit(1));

    populateMakeRangeOpConversionPattern(typeConverter, patterns,
                                         PatternBenefit(1));
    populateElementwiseOpConversionPattern(typeConverter, patterns,
                                           PatternBenefit(1));
    populateViewOpConversionPattern(typeConverter, patterns, PatternBenefit(1));

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // don't rely on D2M::Allocate to hoist CB allocs out of the d2m.generic
    for (auto func : mod.getOps<func::FuncOp>()) {
      if (failed(hoistCBAllocs(func)))
        return signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
