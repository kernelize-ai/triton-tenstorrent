#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#include "npu/include/Analysis/Utility.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTTAGINPUTOUTPUTS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

namespace {

// Tag the argument with `desired`, failing if it was already tagged with a
// conflicting io_type (i.e. used as both an input and an output).
static LogicalResult tagAs(unsigned idx, tt::IOType desired,
                           triton::FuncOp funcOp) {
  if (auto existingAttr = funcOp.getArgAttr(idx, kIOTypeAttrName)) {
    auto existing = dyn_cast<tt::IOTypeAttr>(existingAttr);
    if (existing && existing.getValue() == desired)
      return success(); // already tagged consistently, no conflict
    return funcOp.emitError()
           << "argument " << funcOp.getArgument(idx) << " (# " << idx
           << ") is used as both an input and an output; conflicting "
              "io_type tag";
  }
  funcOp.setArgAttr(idx, kIOTypeAttrName,
                    tt::IOTypeAttr::get(funcOp.getContext(), desired));
  return success();
}

} // namespace

class TritonTenstorrentTagInputOutputsPass
    : public npu::impl::TritonTenstorrentTagInputOutputsBase<
          TritonTenstorrentTagInputOutputsPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;

      funcOp.walk([&](Operation *op) {
        Value ptr;
        tt::IOType desired{};
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          ptr = loadOp.getPtr();
          desired = tt::IOType::INPUT;
        } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
          ptr = storeOp.getPtr();
          desired = tt::IOType::OUTPUT;
        } else if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(op)) {
          ptr = descLoad.getDesc();
          desired = tt::IOType::INPUT;
        } else if (auto descStore = dyn_cast<triton::DescriptorStoreOp>(op)) {
          ptr = descStore.getDesc();
          desired = tt::IOType::OUTPUT;
        } else {
          return;
        }

        BlockArgument arg = traceToBlock(ptr, &funcOp.getBody().front());
        if (!arg)
          return;

        unsigned idx = arg.getArgNumber();
        if (failed(tagAs(idx, desired, funcOp)))
          signalPassFailure();
      });
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
