#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
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

      if (failed(tagFunctionArguments(funcOp)))
        signalPassFailure();
      return;

      // todo: delete me
      auto &body = funcOp.getBody();
      for (auto arg : body.front().getArguments()) {

        Type argType = arg.getType();
        if ((isa<PointerType>(argType)) || isa<TensorDescType>(argType)) {
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

  BlockArgument traceToFuncArg(Value v, triton::FuncOp funcOp) {
    while (true) {
      if (auto blockArg = dyn_cast<BlockArgument>(v))
        return blockArg.getOwner() == &funcOp.getBody().front() ? blockArg
                                                                : nullptr;

      Operation *def = v.getDefiningOp();
      if (!def)
        return nullptr;

      // traverse the addptr -> splat -> broadcast chain to find the original
      // ptr argument
      v = llvm::TypeSwitch<Operation *, Value>(def)
              .Case<triton::AddPtrOp>(
                  [](auto addPtrOp) { return addPtrOp.getPtr(); })
              .Case<triton::SplatOp>(
                  [](auto splatOp) { return splatOp.getSrc(); })
              .Case<triton::BroadcastOp, triton::ExpandDimsOp,
                    triton::ReshapeOp, triton::BitcastOp>(
                  [](auto o) { return o->getOperand(0); })
              .Default([](Operation *) { return Value(); });

      if (!v)
        return nullptr;
    }
  }

  LogicalResult tagFunctionArguments(triton::FuncOp funcOp) {
    MLIRContext *context = &getContext();

    // Tag the argument with `desired`, failing if it was already tagged with a
    // conflicting io_type (i.e. used as both an input and an output).
    auto tagAs = [&](unsigned idx, tt::IOType desired) -> LogicalResult {
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
                        tt::IOTypeAttr::get(context, desired));
      return success();
    };

    bool signalFailure = false;
    funcOp.walk([&](Operation *op) {
      Value ptr;
      tt::IOType desired;
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

      BlockArgument arg = traceToFuncArg(ptr, funcOp);
      if (!arg)
        return;

      unsigned idx = arg.getArgNumber();
      if (failed(tagAs(idx, desired)))
        signalFailure = true; // todo: uplevel this to the main runOnPOperation
                              // function and just signalpassfailure here
    });
    return failed ? failure() : success();
  }

  LogicalResult addTag(BlockArgument arg, triton::FuncOp funcOp) {
    MLIRContext *context = &getContext();
    unsigned idx = arg.getArgNumber();

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

#if 1
    SetVector<Operation *> slice;
    mlir::ForwardSliceOptions opts;
    opts.filter = [&](Operation *op) {
      if (isa<triton::DescriptorLoadOp, triton::DescriptorStoreOp>(op)) {
        return llvm::is_contained(op->getOperands(), arg);
      }
      return true;
    };
    mlir::getForwardSlice(arg, &slice, opts);

    for (auto *op : slice) {
      if (isa<triton::LoadOp, triton::DescriptorLoadOp>(op)) {
        if (failed(tagAs(tt::IOType::INPUT)))
          return failure();
      } else if (isa<triton::StoreOp, triton::DescriptorStoreOp>(op)) {
        if (failed(tagAs(tt::IOType::OUTPUT)))
          return failure();
      }
    }

#else

    SmallVector<Operation *> worklist;
    SetVector<Operation *> visited;
    worklist.append(arg.getUsers().begin(), arg.getUsers().end());
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();
      if (visited.contains(op))
        continue;
      visited.insert(op);

      llvm::errs() << "visiting op " << *op << "\n";

      // TODO: handle tensor descriptors separately
      if (isa<triton::LoadOp>(op)) {
        llvm::errs() << "found a load!\n";
        assert(false && "TODO");
      } else if (isa<triton::StoreOp>(op)) {
        llvm::errs() << "found a store!\n";
        assert(false && "TODO");
      } else {
        worklist.push_back(op);
      }
    }

#endif
    return success();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
