#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CREATETTNNGENERICOP
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

namespace {

struct Kernels {
  func::FuncOp reader;
  func::FuncOp compute;
  func::FuncOp writer;

  Kernels(ModuleOp m) {
    m.walk([&](func::FuncOp f) {
      if (f.getName().ends_with("__reader")) {
        assert(!reader && "expected only one reader kernel per module");
        reader = f;
      } else if (f.getName().ends_with("compute")) {
        assert(!compute && "expected only one compute kernel per module");
        compute = f;
      } else if (f.getName().ends_with("writer")) {
        assert(!writer && "expected only one writer kernel per module");
        writer = f;
      }
    });
    assert(reader && "expected one reader kernel per module");
    assert(compute && "expected one compute kernel per module");
    assert(writer && "expected one writer kernel per module");
  }
};

static SmallVector<ttnn::KernelCBAttr>
createCBDescriptors(Builder &builder, func::FuncOp func,
                    const ttcore::DeviceAttr &device,
                    const ttnn::CoreRangeSetAttr &coreRangeSet) {
  MLIRContext *ctx = builder.getContext();

  DenseSet<Value> cbs;
  func.walk([&](ttkernel::GetCompileArgValOp op) {
    if (auto cbType = dyn_cast<ttkernel::CBType>(op.getType())) {
      cbs.insert(op.getArgVal());
    }
  });

  SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbs.size());
  // similar to D2MToTTNN.cppp:createCBDescriptors
  for (auto [i, cb] : llvm::enumerate(cbs)) {
    auto cbType = cast<ttkernel::CBType>(cb.getType());

    auto elementType = cast<ttcore::TileType>(cbType.getElementType());
    size_t pageSize = elementType.getSizeBytes();
    size_t numPages =
        static_cast<size_t>(cbType.getNumElements()); // or getNumTiles()?

    ttcore::DataType dtype = elementType.getDataType();
    // ttcore::elementTypeToDataType(elementType);

    ttnn::KernelCBFormatAttr cbFormat =
        ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);

    // currently unused but required by KernelCBAttr builder
    ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
    cbDescriptors[i] =
        ttnn::KernelCBAttr::get(ctx, numPages * pageSize, coreRangeSet,
                                {cbFormat}, globalCBIndexOfTensor);
  }

  return cbDescriptors;
}

} // namespace

struct CreateTTNNGenericOp
    : public impl::CreateTTNNGenericOpBase<CreateTTNNGenericOp> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Create a top-level TTNN.generic operation
    ttcore::DeviceAttr device = ttcore::lookupDevice(m);
    assert(device && "failed to find device op in module");

    auto grid = device.getWorkerGrid();
    assert(grid.getRank() == 2 && "expected rank 2 device grid");
    llvm::errs() << "grid: " << grid << "\n";

    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        context, ttnn::CoreRangeAttr::get(
                     context, ttnn::CoreCoordAttr::get(context, 0, 0),
                     ttnn::CoreCoordAttr::get(context, grid.getShape()[0],
                                              grid.getShape()[1])));

    auto kernels = Kernels(m);

    OpBuilder builder(m);

    SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(builder, kernels.compute, device, coreRangeSet);
    llvm::errs() << "got here?" << "\n";
    // assert(false && "TODO");
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
