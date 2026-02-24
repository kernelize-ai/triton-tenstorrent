#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#pragma GCC diagnostic pop
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "ttmlir/FunctionTypes.h"
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

  SmallVector<func::FuncOp> to_vector() const {
    return {reader, compute, writer};
  }
};

static SmallVector<ttnn::KernelCBAttr>
createCBDescriptors(MLIRContext *ctx, func::FuncOp func,
                    const ttcore::DeviceAttr &device,
                    const ttnn::CoreRangeSetAttr &coreRangeSet) {
  SetVector<Value> cbs;
  func.walk([&](ttkernel::GetCompileArgValOp op) {
    if (auto cbType = dyn_cast<ttkernel::CBType>(op.getType())) {
      cbs.insert(op.getArgVal());
    }
  });

  SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbs.size());
  // similar to D2MToTTNN.cppp:createCBDescriptors
  for (auto [i, cb] : llvm::enumerate(llvm::reverse(cbs))) {
    auto cbType = cast<ttkernel::CBType>(cb.getType());

    auto elementType = cast<ttcore::TileType>(cbType.getElementType());
    size_t pageSize = elementType.getSizeBytes();
    size_t numPages =
        static_cast<size_t>(cbType.getNumElements()); // or getNumTiles()?

    ttcore::DataType dtype = elementType.getDataType();
    ttnn::KernelCBFormatAttr cbFormat =
        ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);

    // currently unused but required by KernelCBAttr builder
    ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
    cbDescriptors[i] =
        ttnn::KernelCBAttr::get(ctx, 2 * numPages * pageSize, coreRangeSet,
                                {cbFormat}, globalCBIndexOfTensor);
  }

  return cbDescriptors;
}

static SmallVector<ttnn::CoreRuntimeArgsAttr> populateBlockStartEndArgs(
    Builder &b, const ttnn::CoreRangeSetAttr &coreRangeSet,
    unsigned tilesPerCore, unsigned coresPerRow, unsigned &crtId) {
  MLIRContext *ctx = b.getContext();
  auto coreRanges = coreRangeSet.getCoreRanges();

  Attribute coresPerRowAttr = ttnn::KernelArgNamedArgAttr::get(
      ctx, StringAttr::get(ctx, "tiles_per_core"), coresPerRow);

  SmallVector<ttnn::CoreRuntimeArgsAttr> perCore;
  for (auto coreRange : coreRanges) {
    auto start = coreRange.getStartCoord();
    auto end = coreRange.getEndCoord();

    unsigned rows = end.getX() - start.getX() + 1;
    unsigned cols = end.getY() - start.getY() + 1;

    perCore.reserve(perCore.size() + rows * cols);

    for (unsigned r = start.getX(); r <= end.getX(); ++r) {
      for (unsigned c = start.getY(); c <= end.getY(); ++c) {
        auto coord = ttnn::CoreCoordAttr::get(ctx, r, c);

        uint32_t pid = crtId;
        Attribute blockStartAttr = ttnn::KernelArgNamedArgAttr::get(
            ctx, StringAttr::get(ctx, "block_start"), pid);
        Attribute blockEndAttr = ttnn::KernelArgNamedArgAttr::get(
            ctx, StringAttr::get(ctx, "block_end"), pid + tilesPerCore);
        crtId += tilesPerCore;

        auto rt = ttnn::CoreRuntimeArgsAttr::get(
            ctx, coord,
            ArrayRef<mlir::Attribute>{blockStartAttr, blockEndAttr,
                                      coresPerRowAttr});
        perCore.push_back(rt);
      }
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
                        ArrayRef<ttnn::CoreRuntimeArgsAttr> kernelRTArgs,
                        SymbolTable &symbolTable,
                        ttmetal::MathFidelity mathFidelity) {
  SmallVector<mlir::Attribute> kernelConfigs;

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
        symbolRef, coreRangeSet, kernelCRTArgs, kernelRTArgs, kernelCTArgs));
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
        symbolRef, coreRangeSet, kernelCRTArgs, kernelRTArgs, kernelCTArgs));
  }

  return kernelConfigs;
}

// copied (mostly) from D2MToTTNN.cpp
static SmallVector<ttnn::KernelSemaphoreAttr>
createSemaphoreDescriptors(Builder &builder, Kernels &kernels,
                           const ttnn::CoreRangeSetAttr &coreRangeSet,
                           const SymbolTable &symbolTable) {
  llvm::DenseSet<size_t> seenSemaphoreIndices;

  for (auto kernelFunc : kernels.to_vector()) {
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    if (!kernelSpec) {
      continue;
    }

    for (auto ctArg : kernelSpec.getCtArgs()) {
      if (ctArg.getArgType() == ttkernel::ArgType::Semaphore) {
        seenSemaphoreIndices.insert(ctArg.getOperandIndex());
      }
    }
  }
  size_t numSemaphores = seenSemaphoreIndices.size();
  if (numSemaphores > 0) {
    // Semaphore indices are assigned sequentially in D2MToTTKernel, so they
    // should be dense.
    size_t minIndex = *llvm::min_element(seenSemaphoreIndices);
    size_t maxIndex = *llvm::max_element(seenSemaphoreIndices);
    assert((minIndex == 0u && maxIndex == numSemaphores - 1) &&
           "Semaphore indices must be dense (0, 1, 2, ..., n-1)");
  }
  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors(numSemaphores);
  for (size_t i = 0; i < numSemaphores; ++i) {
    semaphoreDescriptors[i] = builder.getAttr<ttnn::KernelSemaphoreAttr>(
        /*id=*/i, ttnn::KernelCoreType::Worker, coreRangeSet,
        /*initial_value=*/0);
  }

  return semaphoreDescriptors;
}

static constexpr int32_t M = 64;
static constexpr int32_t N = 2560;
static constexpr int32_t K = 9728;
static constexpr int32_t stride_AM = K;
static constexpr int32_t stride_BK = N;
static constexpr int32_t stride_CM = N;

// TODO: take the args map as a parameter
// 0: A ptr
// 1: M (A.shape[0])
// 2: K (A.shape[1])
// 3: stride_AM = K (A.strides[0])
// 4: 1 (A.strides[1])
// 5: 0 (A.has_padding)
// 6: M (A.shape[0])
// 7: K (A.shape[1])
// 8: stride_AM (A.strides[0])
// 9: 1 (A.strides[1])
// 10: B ptr
// 11: K (B.shape[0])
// 12: N (B.shape[1])
// 13: stride_BK (B.strides[0])
// 14: 1 (B.strides[1])
// 15: 0 (B.has_padding)
// 16: K (B.shape[0])
// 17: N (B.shape[1])
// 18: stride_BK (B.strides[0])
// 19: 1 (B.strides[1])
// 20: C ptr
// 21: M (C.shape[0])
// 22: N (C.shape[1])
// 23: stride_CM (C.strides[0])
// 24: 1 (C.strides[1])
// 25: 0 (C.has_padding)
// 26: M (C.shape[0])
// 27: N (C.shape[1])
// 28: stride_CM (C.strides[0])
// 29: 1 (C.strides[1])
// #ifdef BIAS
// 30: bias ptr
// 31: M (bias.shape[0])
// 32: N (bias.shape[1])
// 33: stride_CM (bias.strides[0])
// 34: 1 (bias.strides[1])
// 35: 0 (bias.has_padding)
// 36: M (bias.shape[0])
// 37: N (bias.shape[1])
// 38: stride_CM (bias.strides[0])
// 39: 1
// #endif
// 40: M
// 41: N
// 42: K
#define BIAS
#ifdef BIAS
static constexpr int32_t commonRuntimeArgs[] = {
    0, M, K, stride_AM, 1, 0, M, K, stride_AM, 1, 0, K, N, stride_BK, 1,
    0, K, N, stride_BK, 1, 0, M, N, stride_CM, 1, 0, M, N, stride_CM, 1,
    0, M, N, stride_CM, 1, 0, M, N, stride_CM, 1, // for bias term
    M, N, K};
static const std::set<int32_t> commonRuntimeArgsPtrIndices{
    0, 10, 20, 30}; // TODO: read from argspec
#else
static constexpr int32_t commonRuntimeArgs[] = {
    0, M, K, stride_AM, 1, 0, M, K, stride_AM, 1, 0, K, N, stride_BK, 1,
    0, K, N, stride_BK, 1, 0, M, N, stride_CM, 1, 0, M, N, stride_CM, 1,
    M, N, K};
static const std::set<int32_t> commonRuntimeArgsPtrIndices{
    0, 10, 20}; // TODO: read from argspec
#endif

void replaceNonPtrCommonRuntimeArgs(func::FuncOp f) {
  unsigned commonRuntimeArgsIndex = 0;
  f.walk([&](ttkernel::GetCommonArgValOp op) {
    Value opIndexVal = op.getArgIndex();
    assert(opIndexVal.getDefiningOp() &&
           "expected constant op for common runtime arg index");
    int32_t opIndex =
        cast<IntegerAttr>(
            cast<arith::ConstantOp>(opIndexVal.getDefiningOp()).getValue())
            .getInt();
    OpBuilder builder(op);
    Location loc = op.getLoc();
    if (commonRuntimeArgsPtrIndices.count(opIndex)) {
      op->replaceAllUsesWith(ttkernel::GetCommonArgValOp::create(
          builder, loc, op.getType(),
          arith::ConstantOp::create(
              builder, loc, builder.getIndexAttr(commonRuntimeArgsIndex++))));
    } else {
      op->replaceAllUsesWith(arith::ConstantOp::create(
          builder, loc,
          builder.getI32IntegerAttr(
              reinterpret_cast<int32_t>(commonRuntimeArgs[opIndex]))));
    }
  });
}

// copied from TransformUtils but with a OpBuilder function argument
static ttnn::GetDeviceOp insertGetDeviceOp(OpBuilder &builder,
                                           ttcore::DeviceAttr deviceAttr,
                                           Location loc) {
  SmallVector<int64_t, 2> meshShape{deviceAttr.getMeshShape()};
  if (meshShape.empty()) {
    meshShape = SmallVector<int64_t, 2>{1, 1};
  }
  // TODO (jnie): Currently hardcoding the mesh offset to 0x0
  // Need a proper plan to dynamically determine this.
  SmallVector<int64_t, 2> meshOffset{0, 0};
  return builder.create<ttnn::GetDeviceOp>(
      loc, builder.getType<ttnn::DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), meshShape[0],
                               meshShape[1]),
      ttnn::MeshOffsetAttr::get(builder.getContext(), meshOffset[0],
                                meshOffset[1]));
}

void createMainFunc(MLIRContext *context, OpBuilder &builder,
                    ttcore::DeviceAttr deviceAttr, ttnn::ProgramAttr program) {
  ttcore::TileType aElementType = ttcore::TileType::get(
      context, ttcore::TileType::getDefaultShape(),
      ttcore::elementTypeToDataType(builder.getBF16Type()));

  auto getLayoutForShape = [&](ArrayRef<int64_t> shape) {
    return ttnn::TTNNLayoutAttr::get(
        context, shape, aElementType, ttnn::BufferType::DRAM,
        ttcore::GridAttr::get(context),
        ttnn::TensorMemoryLayoutAttr::get(
            context, ttnn::TensorMemoryLayout::Interleaved));
  };

  auto aLayout = getLayoutForShape({M, K});
  auto bLayout = getLayoutForShape({K, N});
  auto cLayout = getLayoutForShape({M, N});

  auto aType =
      RankedTensorType::get({M, K}, aElementType.getElementType(), aLayout);
  auto bType =
      RankedTensorType::get({K, N}, aElementType.getElementType(), bLayout);
  auto cType =
      RankedTensorType::get({M, N}, aElementType.getElementType(), cLayout);

  SmallVector<Type> kernelArgTypes{aType, bType};
#ifdef BIAS
  // bias matches type of dot output
  kernelArgTypes.push_back(cType);
#endif
  auto funcType = builder.getFunctionType(kernelArgTypes, {cType});
  auto mainFunc =
      builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
  mainFunc.setPublic(); // does this do anything?
  ttmlir::utils::setFunctionType(mainFunc,
                                 ttmlir::utils::FunctionType::ForwardDevice);

  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  ttnn::GetDeviceOp deviceOp =
      insertGetDeviceOp(builder, deviceAttr, builder.getUnknownLoc());

  Value cTensor = ttnn::EmptyOp::create(
      builder, builder.getUnknownLoc(), cType, deviceOp.getDevice(),
      ttnn::ShapeAttr::get(context, {M, N}),
      ttcore::DataTypeAttr::get(context,
                                ttcore::elementTypeToDataType(aElementType)),
      ttnn::LayoutAttr::get(context, ttnn::Layout::Tile),
      ttnn::MemoryConfigAttr::get(
          context,
          ttnn::TensorMemoryLayoutAttr::get(
              context, ttnn::TensorMemoryLayout::Interleaved),
          ttnn::BufferTypeAttr::get(context, ttnn::BufferType::DRAM),
          std::nullopt));

  SmallVector<Value> ios;
  ios.push_back(entryBlock->getArgument(0));
  ios.push_back(entryBlock->getArgument(1));
  ios.push_back(cTensor);
#ifdef BIAS
  ios.push_back(entryBlock->getArgument(2)); // bias tensor argument
#endif

  ttnn::GenericOp::create(builder, builder.getUnknownLoc(), ios, program,
                          ttnn::MemoryConfigAttr());

  for (auto [idx, val] : llvm::enumerate(ios)) {
    if (idx == 2) {
      // cTensor is returned so we don't deallocate it here
      continue;
    }
    ttnn::DeallocateOp::create(builder, builder.getUnknownLoc(), val);
  }

  func::ReturnOp::create(builder, builder.getUnknownLoc(), ValueRange{cTensor});
}

} // namespace

struct CreateTTNNGenericOp
    : public impl::CreateTTNNGenericOpBase<CreateTTNNGenericOp> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // replace all non-ptr common runtime args with scalar constants
    m.walk([&](func::FuncOp f) { replaceNonPtrCommonRuntimeArgs(f); });

    // Create a top-level TTNN.generic operation
    ttcore::DeviceAttr device = ttcore::lookupDevice(m);
    assert(device && "failed to find device op in module");

    auto grid = device.getWorkerGrid();
    assert(grid.getRank() == 2 && "expected rank 2 device grid");

    auto getCoreCoord = [&](int x, int y) {
      return ttnn::CoreCoordAttr::get(context, x, y);
    };

    auto getCoreRange = [&](int x0, int y0, int x1, int y1) {
      return ttnn::CoreRangeAttr::get(context, getCoreCoord(x0, y0),
                                      getCoreCoord(x1, y1));
    };

    // All cores: {[(x=0,y=0) - (x=7,y=6)]}
    ttnn::CoreRangeSetAttr allCores =
        ttnn::CoreRangeSetAttr::get(context, {getCoreRange(0, 0, 7, 6)});

    unsigned gridId = 0;
    auto populateBlockStartEndArgsForSet =
        [&](OpBuilder &builder, ttnn::CoreRangeSetAttr coreRangeSet,
            const unsigned tilesPerCore, const unsigned coresPerRow,
            SmallVector<ttnn::CoreRuntimeArgsAttr> &kernelRTArgs) {
          auto perCoreArgs = populateBlockStartEndArgs(
              builder, coreRangeSet, tilesPerCore, coresPerRow, gridId);
          kernelRTArgs.append(perCoreArgs.begin(), perCoreArgs.end());
        };

    auto kernels = Kernels(m);
    OpBuilder builder(m);

    SmallVector<ttnn::CoreRuntimeArgsAttr> kernelRTArgs;
    // 64x64xK with multicast
    ttnn::CoreRangeSetAttr coreRangeSet1 =
        ttnn::CoreRangeSetAttr::get(context, {getCoreRange(0, 0, 7, 4)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1,
                                    /*coresPerRow=*/5, kernelRTArgs);
    assert(kernelRTArgs.size() == 40 && "expected dispatch for 40 cores");
    allCores = coreRangeSet1;

    // Create CB descriptors.
    SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(context, kernels.compute, device, allCores);

    // Create KernelDescriptors
    auto mathFidelity = ttmetal::MathFidelity::HiFi4; // TODO: parametrize
    SymbolTable symbolTable(m);
    SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
        builder, kernels, allCores, kernelRTArgs, symbolTable, mathFidelity);

    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
        createSemaphoreDescriptors(builder, kernels, allCores, symbolTable);

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        context, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

    auto deviceOps = llvm::to_vector(m.getOps<ttcore::DeviceOp>());
    assert(deviceOps.size() == 1 && "expected only one device op");
    builder.setInsertionPointAfter(deviceOps.front());
    createMainFunc(context, builder, device, program);
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
