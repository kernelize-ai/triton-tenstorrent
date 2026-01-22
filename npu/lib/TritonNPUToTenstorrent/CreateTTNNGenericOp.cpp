#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
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
createCBDescriptors(MLIRContext *ctx, func::FuncOp func,
                    const ttcore::DeviceAttr &device,
                    const ttnn::CoreRangeSetAttr &coreRangeSet) {
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

// map the worker grid to the identity mapping (one block per core)
static SmallVector<ttnn::CoreRuntimeArgsAttr>
populateBlockStartEndArgs(Builder &b,
                          const ttnn::CoreRangeSetAttr &coreRangeSet) {
  MLIRContext *ctx = b.getContext();

  auto coreRanges = coreRangeSet.getCoreRanges();
  assert(coreRanges.size() == 1 && "expected exactly one core range");

  auto coreRange = coreRanges[0];
  auto start = coreRange.getStartCoord();
  assert(start.getX() == 0 && start.getY() == 0 &&
         "expected start coordinate to be (0,0)");

  unsigned rows = coreRange.getEndCoord().getX();
  unsigned cols = coreRange.getEndCoord().getY();

  SmallVector<ttnn::CoreRuntimeArgsAttr> perCore;
  perCore.reserve(rows * cols);

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      auto coord = ttnn::CoreCoordAttr::get(ctx, r, c);

      uint32_t id = r * cols + c;
      Attribute blockStartAttr = b.getI32IntegerAttr(id);
      Attribute blockEndAttr = b.getI32IntegerAttr(id + 1);
      auto rt = ttnn::CoreRuntimeArgsAttr::get(
          ctx, coord, ArrayRef<mlir::Attribute>{blockStartAttr, blockEndAttr});
      perCore.push_back(rt);
    }
  }

  return perCore;
}

static ttnn::ComputeKernelMathFidelity
convertMathFidelity(ttmetal::MathFidelity fidelity) {
  switch (fidelity) {
  case ttmetal::MathFidelity::LoFi:
    return ttnn::ComputeKernelMathFidelity::LoFi;
  case ttmetal::MathFidelity::HiFi2:
    return ttnn::ComputeKernelMathFidelity::HiFi2;
  case ttmetal::MathFidelity::HiFi3:
    return ttnn::ComputeKernelMathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi4:
    return ttnn::ComputeKernelMathFidelity::HiFi4;
  }
  llvm_unreachable("Invalid MathFidelity");
}

static mlir::Attribute convertKernelArg(Builder &builder,
                                        const ttkernel::ArgAttr &arg) {
  switch (arg.getArgType()) {
  case ttkernel::ArgType::BufferAddress: {
    return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(
        arg.getOperandIndex());
  }
  case ttkernel::ArgType::CBPort: {
    return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
        arg.getOperandIndex());
  }
  case ttkernel::ArgType::Semaphore: {
    return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
        arg.getOperandIndex());
  }
  }
  llvm_unreachable("Invalid ArgType");
}

static SmallVector<mlir::Attribute>
createKernelDescriptors(Builder &builder, Kernels &kernels,
                        const ttnn::CoreRangeSetAttr &coreRangeSet,
                        SymbolTable &symbolTable,
                        ttmetal::MathFidelity mathFidelity) {
  SmallVector<mlir::Attribute> kernelConfigs;

  // TODO: these are hardcoded as the TTKernel arg spec does not support RT args
  // We will need to use CoreRuntimeArgsAttr
  SmallVector<ttnn::CoreRuntimeArgsAttr> kernelRTArgs =
      populateBlockStartEndArgs(builder, coreRangeSet);

  // reader
  {
    auto kernelFunc = kernels.reader;
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    auto crtArgs = kernelSpec.getRtArgs();
    auto ctArgs = kernelSpec.getCtArgs();
    SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
    SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
    for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
      kernelCRTArgs[i] = convertKernelArg(builder, arg);
    }
    for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
      kernelCTArgs[i] = convertKernelArg(builder, arg);
    }

    auto symbolRef = SymbolRefAttr::get(kernelFunc.getContext(),
                                        kernelFunc.getSymNameAttr());

    kernelConfigs.push_back(builder.getAttr<ttnn::ReadKernelAttr>(
        symbolRef, coreRangeSet, kernelCRTArgs,
        kernelRTArgs, kernelCTArgs));
  }

  // compute
  {
    auto kernelFunc = kernels.compute;
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    auto crtArgs = kernelSpec.getRtArgs();
    auto ctArgs = kernelSpec.getCtArgs();
    SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
    SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
    for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
      kernelCRTArgs[i] = convertKernelArg(builder, arg);
    }
    for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
      kernelCTArgs[i] = convertKernelArg(builder, arg);
    }

    auto symbolRef = SymbolRefAttr::get(kernelFunc.getContext(),
                                        kernelFunc.getSymNameAttr());

    kernelConfigs.push_back(builder.getAttr<ttnn::ComputeKernelAttr>(
        symbolRef, coreRangeSet,
        /*math_fidelity*/ convertMathFidelity(mathFidelity),
        /*fp32DestAccum*/ false,
        /*dst_full_sync_en*/ false,
        /*unpack_to_dest_mode*/
        ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
            ttnn::ComputeKernelUnpackToDestMode::Default},
        /*bfp8_pack_precise*/ false,
        /*math_approx_mode*/ false, kernelCRTArgs, kernelRTArgs, kernelCTArgs));
  }

  // writer
  {
    auto kernelFunc = kernels.writer;
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    auto crtArgs = kernelSpec.getRtArgs();
    auto ctArgs = kernelSpec.getCtArgs();
    SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
    SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
    for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
      kernelCRTArgs[i] = convertKernelArg(builder, arg);
    }
    for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
      kernelCTArgs[i] = convertKernelArg(builder, arg);
    }

    auto symbolRef = SymbolRefAttr::get(kernelFunc.getContext(),
                                        kernelFunc.getSymNameAttr());

    kernelConfigs.push_back(builder.getAttr<ttnn::WriteKernelAttr>(
        symbolRef, coreRangeSet, kernelCRTArgs,
        kernelRTArgs, kernelCTArgs));
  }

  return kernelConfigs;
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

    // Create CB descriptors.
    SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(context, kernels.compute, device, coreRangeSet);

    // Create KernelDescriptors
    auto mathFidelity = ttmetal::MathFidelity::HiFi4; // TODO: parametrize
    SymbolTable symbolTable(m);
    SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
        builder, kernels, coreRangeSet, symbolTable, mathFidelity);

    // semaphores not yet used
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors;

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        context, kernelDescriptors, cbDescriptors, semaphoreDescriptors);
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
