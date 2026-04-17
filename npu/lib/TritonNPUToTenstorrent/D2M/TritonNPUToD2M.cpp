#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

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
      auto shardShape = llvm::to_vector(
          map_range((llvm::zip(tensorType.getShape(), tileType.getShape())),
                    [](auto pair) -> int64_t {
                      auto [dim, tileDim] = pair;
                      return dim / tileDim;
                    }));

      SmallVector<int64_t> shape{1, 1};
      shape.append(shardShape.begin(), shardShape.end());
      auto memRefType = MemRefType::get(
          shape, tileType,
          ttcore::ViewLayoutAttr::get(t.getContext(), shape.size()),
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
      if (isa<triton::PointerType>(eType)) {
        return IntegerType::get(tensorType.getContext(), 32);
      }
      if (!tensorType.getEncoding()) {
        return tensorType;
      }
      if (isa<npu::tt::TiledEncodingAttr>(tensorType.getEncoding()) ||
          isa<npu::tt::TiledDotOperandEncodingAttr>(tensorType.getEncoding()) ||
          isa<gpu::DotOperandEncodingAttr>(tensorType.getEncoding())) {
        // convert to memref in L1
        auto tileType = ttcore::TileType::get(
            tensorType.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(eType));
        auto shardShape = llvm::to_vector(
            map_range((llvm::zip(tensorType.getShape(), tileType.getShape())),
                      [](auto pair) -> int64_t {
                        auto [dim, tileDim] = pair;
                        return dim / tileDim;
                      }));
        auto memRefType = MemRefType::get(
            shardShape, tileType, MemRefLayoutAttrInterface{},
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
    // target.addIllegalDialect<triton::cpu::TritonCPUDialect>();
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
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
