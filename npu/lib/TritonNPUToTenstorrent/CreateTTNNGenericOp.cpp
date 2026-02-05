#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#pragma GCC diagnostic pop
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
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
    llvm::errs() << "Creating CB descriptor for CB " << i << ": " << cbType
                 << "\n";
    auto elementType = cast<ttcore::TileType>(cbType.getElementType());
    size_t pageSize = elementType.getSizeBytes();
    size_t numPages =
        static_cast<size_t>(cbType.getNumElements()); // or getNumTiles()?

    llvm::errs() << "  pageSize = " << pageSize << ", numPages = " << numPages
                 << ", numTiles = " << cbType.getNumTiles() << "\n";
    ttcore::DataType dtype = elementType.getDataType();
    // ttcore::elementTypeToDataType(elementType);

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

// map the worker grid to the identity mapping (one block per core)
static SmallVector<ttnn::CoreRuntimeArgsAttr>
populateBlockStartEndArgs(Builder &b,
                          const ttnn::CoreRangeSetAttr &coreRangeSet,
                          unsigned tilesPerCore, unsigned &crtId) {
  MLIRContext *ctx = b.getContext();
  auto coreRanges = coreRangeSet.getCoreRanges();

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
        uint32_t pid;
        if (r < 4) {
          uint32_t n = 4 * c + r; // 0..19
          pid = 2 * n;            // even
        } else {
          uint32_t n = 4 * c + (r - 4); // 0..19
          pid = 2 * n + 1;              // odd
        }
        llvm::errs() << "CoreRuntimeArgsAttr for core (" << r << ", " << c
                     << ") = " << pid << "\n";
        // TODO: fix the printer so quotes are not required
        Attribute blockStartAttr = ttnn::KernelNamedArgAttr::get(
            ctx, std::string("\"block_start\""), pid);
        Attribute blockEndAttr = ttnn::KernelNamedArgAttr::get(
            ctx, std::string("\"block_end\""), pid + tilesPerCore);
        crtId += tilesPerCore;

        auto rt = ttnn::CoreRuntimeArgsAttr::get(
            ctx, coord,
            ArrayRef<mlir::Attribute>{blockStartAttr, blockEndAttr});
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
// 30: M
// 31: N
// 32: K
static constexpr int32_t commonRuntimeArgs[] = {
    0, M, K, stride_AM, 1, 0, M, K, stride_AM, 1, 0, K, N, stride_BK, 1,
    0, K, N, stride_BK, 1, 0, M, N, stride_CM, 1, 0, M, N, stride_CM, 1,
    M, N, K};
static const std::set<int32_t> commonRuntimeArgsPtrIndices{
    0, 10, 20}; // TODO: read from argspec

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
  llvm::errs() << "cType = " << cType << "\n";

  auto funcType = builder.getFunctionType({aType, bType}, {cType});
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

  ttnn::GenericOp::create(builder, builder.getUnknownLoc(), ios, program,
                          ttnn::MemoryConfigAttr());

  ttnn::DeallocateOp::create(builder, builder.getUnknownLoc(), ios[0]);
  ttnn::DeallocateOp::create(builder, builder.getUnknownLoc(), ios[1]);

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
    llvm::errs() << "grid: " << grid << "\n";

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

    // split work to cores typically creates two core range sets.
    // Distributing 80 output tiles across 56 cores: 24 cores ({[(x=0,y=0) -
    // (x=2,y=6)], [(x=3,y=0) - (x=3,y=2)]}) x 2 tiles/core + 32 cores
    // ({[(x=3,y=3) - (x=3,y=6)], [(x=4,y=0) - (x=7,y=6)]}) x 1 tiles/core

    unsigned gridId = 0;
    auto populateBlockStartEndArgsForSet =
        [&](OpBuilder &builder, ttnn::CoreRangeSetAttr coreRangeSet,
            const unsigned tilesPerCore,
            SmallVector<ttnn::CoreRuntimeArgsAttr> &kernelRTArgs) {
          auto perCoreArgs = populateBlockStartEndArgs(builder, coreRangeSet,
                                                       tilesPerCore, gridId);
          kernelRTArgs.append(perCoreArgs.begin(), perCoreArgs.end());
        };

    auto kernels = Kernels(m);
    OpBuilder builder(m);

    SmallVector<ttnn::CoreRuntimeArgsAttr> kernelRTArgs;
// 20260202

// 1024x1024x1024
#if 0
  // 64x128xK
  // Distributing 128 output tiles across 56 cores: 16 cores ({[(x=0,y=0) - (x=1,y=6)], [(x=2,y=0) - (x=2,y=1)]}) x 3 tiles/core + 40 cores ({[(x=2,y=2) - (x=2,y=6)], [(x=3,y=0) - (x=7,y=6)]}) x 2 tiles/core
  ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
      context, {getCoreRange(0, 0, 1, 6), getCoreRange(2, 0, 2, 1)});
  populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/3,
                                  kernelRTArgs);
  ttnn::CoreRangeSetAttr coreRangeSet2 = ttnn::CoreRangeSetAttr::get(
      context, {getCoreRange(2, 2, 2, 6), getCoreRange(3, 0, 7, 6)});
  populateBlockStartEndArgsForSet(builder, coreRangeSet2, /*tilesPerCore=*/2,
                                  kernelRTArgs);
  assert(kernelRTArgs.size() == 56 && "expected dispatch for 56 cores");
#endif

#if 0
  // 64x64xK
  // Distributing 256 output tiles across 56 cores: 32 cores ({[(x=0,y=0) - (x=3,y=6)], [(x=4,y=0) - (x=4,y=3)]}) x 5 tiles/core + 24 cores ({[(x=4,y=4) - (x=4,y=6)], [(x=5,y=0) - (x=7,y=6)]}) x 4 tiles/core
  ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
      context, {getCoreRange(0, 0, 3, 6), getCoreRange(4, 0, 4, 3)});
  populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/5,
                                  kernelRTArgs);
  ttnn::CoreRangeSetAttr coreRangeSet2 = ttnn::CoreRangeSetAttr::get(
      context, {getCoreRange(4, 4, 4, 6), getCoreRange(5, 0, 7, 6)});
  populateBlockStartEndArgsForSet(builder, coreRangeSet2, /*tilesPerCore=*/4,
                                  kernelRTArgs);
  assert(kernelRTArgs.size() == 56 && "expected dispatch for 56 cores");
#endif

// Qwen
#if 1
    // 32x128xK with multicast
    ttnn::CoreRangeSetAttr coreRangeSet1 =
        ttnn::CoreRangeSetAttr::get(context, {getCoreRange(0, 0, 7, 4)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1,
                                    kernelRTArgs);
    assert(kernelRTArgs.size() == 40 && "expected dispatch for 40 cores");
    allCores = coreRangeSet1;
#endif
#if 0
    // 64x64xK or 32x128xK
    // Distributing 40 output tiles across 40 cores: 40 cores ({[(x=0,y=0) -
    // (x=4,y=6)], [(x=5,y=0) - (x=5,y=4)]}) x 1 tiles/core + 0 cores ({}) x 0
    // tiles/core
    ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 4, 6), getCoreRange(5, 0, 5, 4)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1,
                                    kernelRTArgs);
    assert(kernelRTArgs.size() == 40 && "expected dispatch for 40 cores");
    allCores = coreRangeSet1;
#endif
#if 0
    // 32x64xK
    // Distributing 80 output tiles across 56 cores: 24 cores ({[(x=0,y=0) - (x=2,y=6)], [(x=3,y=0) - (x=3,y=2)]}) x 2 tiles/core + 32 cores ({[(x=3,y=3) - (x=3,y=6)], [(x=4,y=0) - (x=7,y=6)]}) x 1 tiles/core
    ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 2, 6), getCoreRange(3, 0, 3, 2)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/2,
                                    kernelRTArgs);
    ttnn::CoreRangeSetAttr coreRangeSet2 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(3, 3, 3, 6), getCoreRange(4, 0, 7, 6)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet2, /*tilesPerCore=*/1,
                                    kernelRTArgs);
    assert(kernelRTArgs.size() == 56 && "expected dispatch for 56 cores");

#endif
#if 0
      // 64x128xK or 32x256xK
      // Distributing 20 output tiles across 20 cores: 20 cores ({[(x=0,y=0) - (x=1,y=6)], [(x=2,y=0) - (x=2,y=5)]}) x 1 tiles/core + 0 cores ({}) x 0 tiles/core
      ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 1, 6), getCoreRange(2, 0, 2, 5)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1,
                                    kernelRTArgs);
    assert(kernelRTArgs.size() == 20 && "expected dispatch for 20 cores");
#endif

#if 0
    // did not work - out of DEST?
    // 64x256x256
    // Distributing 10 output tiles across 10 cores: 10 cores ({[(x=0,y=0) - (x=0,y=6)], [(x=1,y=0) - (x=1,y=2)]}) x 1 tiles/core + 0 cores ({}) x 0 tiles/core
// All cores: {[(x=0,y=0) - (x=0,y=6)], [(x=1,y=0) - (x=1,y=2)]}
// Matrix A: 64x9728 (608 tiles), Matrix B: 9728x2560 (24320 tiles), Output C: 64x2560 (160 tiles)
  ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 0, 6), getCoreRange(1, 0, 1, 2)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1, kernelRTArgs);
     assert(kernelRTArgs.size() == 10 && "expected dispatch for 10 cores");
#endif
#if 0
    // ~64x64x64 but using only 8 cores~ (getCoreRange(0, 0, 7, 0))
    // 64x128x128 but using only 10 cores ({[(x=0,y=0) - (x=0,y=6)], [(x=1,y=0) - (x=1,y=2)]})
    ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 0, 6), getCoreRange(1, 0, 1, 2)});
    unsigned mBlocks = M / 64;
    unsigned nBlocks = N / 128;
    unsigned totalBlocks = mBlocks * nBlocks;
    unsigned totalCores = 10;
    assert(totalBlocks % totalCores == 0 &&
           "expected total blocks to be evenly divisible by total cores");
    unsigned totalBlocksPerCore = totalBlocks / totalCores;
    llvm::errs() << "totalBlocksPerCore = " << totalBlocksPerCore << " (mBlocks=" << mBlocks << ", nBlocks=" << nBlocks << ")\n";
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/totalBlocksPerCore,
                                    kernelRTArgs);
#endif

#if 0
    // 64x64x64
    // Distributing 40 output tiles across 40 cores: 40 cores ({[(x=0,y=0) -
    // (x=4,y=6)], [(x=5,y=0) - (x=5,y=4)]}) x 1 tiles/core + 0 cores ({}) x 0
    // tiles/core
    ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 4, 6), getCoreRange(5, 0, 5, 4)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/1,
                                    kernelRTArgs);
#endif

#if 0
    // TODO: these are hardcoded as the TTKernel arg spec does not support RT
    // args We will need to use CoreRuntimeArgsAttr
    ttnn::CoreRangeSetAttr coreRangeSet1 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(0, 0, 2, 6), getCoreRange(3, 0, 3, 2)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet1, /*tilesPerCore=*/2,
                                    kernelRTArgs);

    ttnn::CoreRangeSetAttr coreRangeSet2 = ttnn::CoreRangeSetAttr::get(
        context, {getCoreRange(3, 3, 3, 6), getCoreRange(4, 0, 7, 6)});
    populateBlockStartEndArgsForSet(builder, coreRangeSet2, /*tilesPerCore=*/1,
                                    kernelRTArgs);
    llvm::errs() << "final grid id = " << gridId << "\n";
    assert(gridId == 80 && "expected dispatch for 80 tiles");
    assert(kernelRTArgs.size() == 56 && "expected dispatch for 56 cores");
#endif

    // Create CB descriptors.
    SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(context, kernels.compute, device, allCores);

    // Create KernelDescriptors
    auto mathFidelity = ttmetal::MathFidelity::HiFi4; // TODO: parametrize
    SymbolTable symbolTable(m);
    SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
        builder, kernels, allCores, kernelRTArgs, symbolTable, mathFidelity);

    // semaphores not yet used
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
        createSemaphoreDescriptors(builder, kernels, allCores, symbolTable);

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        context, kernelDescriptors, cbDescriptors, semaphoreDescriptors);
    llvm::errs() << "Program: " << program << "\n";

    auto deviceOps = llvm::to_vector(m.getOps<ttcore::DeviceOp>());
    assert(deviceOps.size() == 1 && "expected only one device op");
    builder.setInsertionPointAfter(deviceOps.front());
    createMainFunc(context, builder, device, program);
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
