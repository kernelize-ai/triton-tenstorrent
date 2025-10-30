#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h" 

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_DROPFUNCTION
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

struct DropFunctionPass : public npu::impl::DropFunctionBase<DropFunctionPass> {

    DropFunctionPass() = default;
  explicit DropFunctionPass(DropFunctionOptions options) : funcName(options.functionName) {
}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (funcName.empty())
      return;

    func::FuncOp target = module.lookupSymbol<func::FuncOp>(funcName);
    if (!target)
      return;

    target.erase();
  }

  std::string funcName;
};

}
}
}
