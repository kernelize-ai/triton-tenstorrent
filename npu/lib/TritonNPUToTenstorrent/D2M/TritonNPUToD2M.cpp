#include "npu/include/TritonNPUToD2M/Passes.h"

#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

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

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    mlir::ConversionTarget funcTarget(*context);
    funcTarget.addIllegalOp<triton::FuncOp>();
    funcTarget.addIllegalOp<triton::ReturnOp>();

    funcTarget.addLegalOp<UnrealizedConversionCastOp>();
    funcTarget.addLegalOp<d2m::GenericOp>();

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
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
