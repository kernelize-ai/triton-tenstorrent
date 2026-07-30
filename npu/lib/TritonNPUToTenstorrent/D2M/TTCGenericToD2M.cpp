#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // ttc.GenericOp

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "TTCGenericPlan.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTTCGENERICTOD2M
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "convert-ttc-generic-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {}

struct ConvertTTCGenericToD2MPass
    : public impl::ConvertTTCGenericToD2MBase<ConvertTTCGenericToD2MPass> {

  LogicalResult emitFunction(triton::FuncOp tritonFunc,
                             MutableArrayRef<GenericPlan> plans,
                             IRRewriter &rewriter) {
    rewriter.setInsertionPoint(tritonFunc);

    llvm::errs() << "TODO\n";
    return failure();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> gridShape = llvm::to_vector(gridAttr.getShape());

    SmallVector<GenericPlan, 4> plans;
    triton::FuncOp tritonFunc;
    mod.walk([&](triton::FuncOp func) {
      // if there are already generic plans from a previous function, bail
      // - we can't cross function boundaries yet
      if (!plans.empty()) {
        func.emitError(
            "expected one parent func for all generic ops in triton kernel");
        signalPassFailure();
      }
      func.walk([&](cpu::GenericOp generic) {
        auto planResult = GenericPlan::build(generic);
        if (failed(planResult)) {
          signalPassFailure();
          return;
        }

        plans.push_back(*planResult);
        llvm::errs() << "generic = " << generic.getHeader() << "\n";
      });
      tritonFunc = func;
    });

    // TODO: using this as a failure signal, but it might be better to signal
    // the walk was interrupted above (or separate the analyze and rewrite steps
    // into functions here)
    if (plans.empty())
      return;

    IRRewriter rewriter(context);
    if (failed(emitFunction(tritonFunc, plans, rewriter)))
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
