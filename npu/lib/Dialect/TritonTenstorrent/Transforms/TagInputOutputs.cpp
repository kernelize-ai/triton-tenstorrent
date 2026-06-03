#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTTAGINPUTOUTPUTS
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

class TritonTenstorrentTagInputOutputsPass
    : public npu::impl::TritonTenstorrentTagInputOutputsBase<
          TritonTenstorrentTagInputOutputsPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;

      auto &body = funcOp.getBody();
      for (auto arg : body.front().getArguments()) {

        Type argType = arg.getType();
        auto tensorTy = dyn_cast<RankedTensorType>(argType);
        if ((tensorTy && isa<PointerType>(tensorTy.getElementType())) ||
            isa<TensorDescType>(argType)) {
          if (failed(addTag(arg, funcOp))) {
            signalPassFailure();
            return;
          }
        }
      }
    });
  }

private:
  static constexpr llvm::StringLiteral kIOTypeAttrName =
      "triton_tenstorrent.io_type";

  LogicalResult addTag(BlockArgument arg, triton::FuncOp funcOp) {
    MLIRContext *context = &getContext();
    unsigned idx = arg.getArgNumber();

    llvm::errs() << "need to tag input: " << arg << "\n";
    SetVector<Operation *> slice;
    mlir::getForwardSlice(arg, &slice);

    // Tag the argument with `desired`, failing if it was already tagged with a
    // conflicting io_type (i.e. used as both an input and an output).
    auto tagAs = [&](tt::IOType desired) -> LogicalResult {
      if (auto existingAttr = funcOp.getArgAttr(idx, kIOTypeAttrName)) {
        auto existing = dyn_cast<tt::IOTypeAttr>(existingAttr);
        if (existing && existing.getValue() == desired)
          return success(); // already tagged consistently, no conflict
        return funcOp.emitError()
               << "argument " << arg << " (# " << idx
               << ") is used as both an input and an output; conflicting "
                  "io_type tag";
      }
      funcOp.setArgAttr(idx, kIOTypeAttrName,
                        tt::IOTypeAttr::get(context, desired));
      return success();
    };

    for (auto *op : slice) {
      if (isa<triton::LoadOp, triton::DescriptorLoadOp>(op)) {
        llvm::errs() << "need to tag load: " << *op << "\n";
        if (failed(tagAs(tt::IOType::INPUT)))
          return failure();
      } else if (isa<triton::StoreOp, triton::DescriptorStoreOp>(op)) {
        llvm::errs() << "need to tag store: " << *op << "\n";
        if (failed(tagAs(tt::IOType::OUTPUT)))
          return failure();
      }
    }
    return success();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
