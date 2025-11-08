#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonNPUToTenstorrent.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

// Tenstorrent TTKernel includes
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTRITONNPUTOTTKERNEL
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

static Value getI32Const(ConversionPatternRewriter &rewriter, Location loc,
                         int64_t value) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), value));
}

static Value getI1Const(ConversionPatternRewriter &rewriter, Location loc,
                        bool value) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI1Type(),
      rewriter.getIntegerAttr(rewriter.getI1Type(), value));
}

static Value getIConst(ConversionPatternRewriter &rewriter, Location loc,
                       int64_t value) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), value));
}

struct ConvertBinaryComputeOp
    : public OpConversionPattern<npu::tt::BinaryComputeOp> {
  using OpConversionPattern<npu::tt::BinaryComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (op.getOpcode().str() != "arith.addf")
      return failure();

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // copy from cb into compute registers
    auto copyAndInitializeRegister = [&](Value operand,
                                         Value convertedOperand) {
      rewriter.create<ttkernel::CopyTileInitOp>(loc, convertedOperand);
      Value c0 = getIConst(rewriter, loc, 0);

      RankedTensorType loadType = cast<RankedTensorType>(operand.getType());
      // amusingly we could now deduce the load encoding from the operand index
      npu::tt::TileEncodingAttr loadEncoding =
          cast<npu::tt::TileEncodingAttr>(loadType.getEncoding());
      Value destRegisterIndex =
          getIConst(rewriter, loc, loadEncoding.getIndex());
      rewriter.create<ttkernel::CopyTileOp>(loc, convertedOperand, c0,
                                            destRegisterIndex);
    };

    copyAndInitializeRegister(op.getLhs(), lhs);
    copyAndInitializeRegister(op.getRhs(), rhs);

    rewriter.create<ttkernel::AddBinaryTilesInitOp>(loc);
    Value lhsIndex = getIConst(rewriter, loc, 0);
    Value rhsIndex = getIConst(rewriter, loc, 1);
    Value destIndex = getIConst(rewriter, loc, 2);
    rewriter.create<ttkernel::AddBinaryTilesOp>(loc, lhsIndex, rhsIndex,
                                                destIndex);

    rewriter.eraseOp(op);
    return success();
  }
};

static bool isScalar(Value v) { return !isa<RankedTensorType>(v.getType()); }

static Value traceToScalar(Value ptr, bool isPtr = true) {
  auto op = ptr.getDefiningOp();
  if (isScalar(ptr) || op == nullptr) {
    return ptr;
  }
  if (auto addPtr = dyn_cast<triton::AddPtrOp>(op)) {
    if (isPtr)
      return traceToScalar(addPtr.getPtr(), isPtr);
    return traceToScalar(addPtr.getOffset(), false);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
    return splatOp.getSrc();
  } else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    auto lhs = traceToScalar(addOp.getLhs(), isPtr);
    auto rhs = traceToScalar(addOp.getRhs(), isPtr);
    if (isScalar(lhs)) {
      assert(!isScalar(rhs) && "expected non-scalar rhs");
      return lhs;
    }
    if (isScalar(rhs))
      return rhs;
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
    //
  } else {
    assert(0 && "unhandled op");
  }
  return ptr;
}

static int64_t findAllocIdx(Operation *op) {
  if (auto localLoadOp = dyn_cast<gpu::LocalLoadOp>(op)) {
    return findAllocIdx(localLoadOp.getSrc().getDefiningOp());
  }
  if (auto localStoreOp = dyn_cast<gpu::LocalStoreOp>(op)) {
    return findAllocIdx(localStoreOp.getDst().getDefiningOp());
  }
  if (auto localAllocOp = dyn_cast<gpu::LocalAllocOp>(op)) {
    return localAllocOp->getAttrOfType<IntegerAttr>("alloc_idx").getInt();
  }
  return -1;
}

static Value computeNocAddr(ConversionPatternRewriter &rewriter, Location loc,
                            Value ptr, Value pageSize, Value cb) {
  // Trace to base address and offset
  Value baseAddr = traceToScalar(ptr, true);
  Value offset = traceToScalar(ptr, false);
  assert(isScalar(offset) && "expected scalar offset");

  // Convert offset to bytes
  triton::PointerType ptrType;
  if (isScalar(ptr)) {
    ptrType = cast<triton::PointerType>(ptr.getType());
  } else {
    auto tensorType = cast<RankedTensorType>(ptr.getType());
    auto elemType = tensorType.getElementType();
    ptrType = cast<triton::PointerType>(elemType);
  }
  auto elemType = ptrType.getPointeeType();
  Value elemSizeValue =
      getI32Const(rewriter, loc, elemType.getIntOrFloatBitWidth() / 8);
  offset = rewriter.create<arith::MulIOp>(loc, offset, elemSizeValue);

  Value tile_id = rewriter.create<arith::DivUIOp>(loc, offset, pageSize);

  Value const1 = getI32Const(rewriter, loc, 1);
  Value const0 = getI32Const(rewriter, loc, 0);

  // Create interleaved address generator and get noc address
  auto dataFormat = rewriter.create<ttkernel::GetDataFormatOp>(loc, cb);
  Value c1bit = getI1Const(rewriter, loc, 1);
  Value addrGen = rewriter.create<ttkernel::GetInterleavedAddrGenFastOp>(
      loc, c1bit, baseAddr, pageSize, dataFormat);
  Value nocAddr = rewriter.create<ttkernel::InterleavedAddrGenFastGetNocAddrOp>(
      loc, addrGen, tile_id, const0, Value());

  return nocAddr;
}

struct ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    int64_t allocIdx = -1;
    for (auto user : op.getResult().getUsers()) {
      allocIdx = std::max(allocIdx, findAllocIdx(user));
    }
    if (allocIdx == -1) {
      return rewriter.notifyMatchFailure(op, "dependent load not supported");
    }

    auto cbMemRefType = cast<ttkernel::CBType>(
        typeConverter->convertType(op.getResult().getType()));
    Value cb = rewriter.create<ttkernel::GetCompileArgValOp>(loc, cbMemRefType,
                                                             allocIdx);

    // Compute page size in bytes
    auto pageSize = rewriter.create<ttkernel::GetTileSizeOp>(loc, cb);

    // Compute the noc address
    Value nocAddr = computeNocAddr(rewriter, loc, op.getPtr(), pageSize, cb);

    // 1. reserve back on the cb to know data is ready to store to SRAM
    //       ttkernel.cb_reserve_back(%0, %c1_i32)
    // add later?
    Value const1 = getI32Const(rewriter, loc, 1);
    rewriter.create<ttkernel::CBReserveBackOp>(loc, cb, const1);

    // 3. get L1 address
    //       %11 = ttkernel.get_write_ptr(%0)
    Value l1Addr = rewriter.create<ttkernel::GetWritePtrOp>(loc, cb);

    // 4. async read from noc to l1, size in bytes
    //       ttkernel.noc_async_read(%10, %13, %c4096_i32)
    rewriter.create<ttkernel::NocAsyncReadOp>(loc, nocAddr, l1Addr, pageSize);

    // 5. barrier to ensure data is read from noc to l1
    //       ttkernel.noc_async_read_barrier() : () -> ()
    rewriter.create<ttkernel::NocAsyncReadBarrierOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertStoreOp : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Should always be a cb?
    auto cb = adaptor.getValue();
    assert(isa<ttkernel::CBType>(cb.getType()) && "expected cb type");

    // Get tile size in bytes
    auto pageSize = rewriter.create<ttkernel::GetTileSizeOp>(loc, cb);

    Value nocAddr = computeNocAddr(rewriter, loc, op.getPtr(), pageSize, cb);

    Value l1Addr = rewriter.create<ttkernel::GetReadPtrOp>(loc, cb);
    rewriter.create<ttkernel::NocAsyncWriteOp>(loc, l1Addr, nocAddr, pageSize);
    rewriter.create<ttkernel::NocAsyncWriteBarrierOp>(loc);
    Value numPages = getI32Const(rewriter, loc, 1);
    rewriter.create<ttkernel::CBPopFrontOp>(loc, cb, numPages);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertLocalStoreOp : public OpConversionPattern<gpu::LocalStoreOp> {
  using OpConversionPattern<gpu::LocalStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dst = adaptor.getDst();
    auto srcOp = op.getSrc().getDefiningOp();

    if (!isa<triton::LoadOp>(srcOp)) {
      // COMPUTE KERNEL
      // reserve back the cb for the store
      Value numPages = getI32Const(rewriter, loc, 1);
      rewriter.create<ttkernel::CBReserveBackOp>(loc, dst, numPages);

      // Pack the tile into the cb
      Value destRegisterIndex = getIConst(rewriter, loc, 2);
      Value outIndex = getIConst(rewriter, loc, 0);
      rewriter.create<ttkernel::PackTileOp>(loc, destRegisterIndex, dst,
                                            outIndex,
                                            /*outOfOrder=*/true);

      rewriter.eraseOp(op);
      return success();
    }

    Value const1 = getI32Const(rewriter, loc, 1);
    //       ttkernel.cb_push_back(%0, %c1_i32)
    rewriter.create<ttkernel::CBPushBackOp>(loc, dst, const1);

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertLocalLoadOp : public OpConversionPattern<gpu::LocalLoadOp> {
  using OpConversionPattern<gpu::LocalLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto src = adaptor.getSrc();
    auto dst = op.getResult();
    // 1. wait_front on the cb to know data is ready to load from SRAM
    Value c1 = getI32Const(rewriter, loc, 1);
    auto waitFrontOp = rewriter.create<ttkernel::CBWaitFrontOp>(loc, src, c1);

    LDBG("Converted load src type = " << src.getType() << "\n");
    assert(isa<ttkernel::CBType>(src.getType()) &&
           "expected memref type for type converted load src");

    // TODO: this should come from the cb root eventually,
    // but replaceOp is probably not appropriate here
    rewriter.replaceOp(op, src);
    return success();
  }
};

struct ConvertLocalAllocOp : public OpConversionPattern<gpu::LocalAllocOp> {
  using OpConversionPattern<gpu::LocalAllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // replace the local allocs with cbs from the function signature
    int64_t allocIdx = -1;
    for (auto user : op.getResult().getUsers()) {
      allocIdx = std::max(allocIdx, findAllocIdx(user));
    }
    if (allocIdx == -1) {
      return rewriter.notifyMatchFailure(op, "missing alloc_idx attribute");
    }

    auto cbMemRefType = cast<ttkernel::CBType>(
        typeConverter->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(op, cbMemRefType,
                                                              allocIdx);

    return success();
  }
};

struct DropFunctionArguments : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = funcOp.getLoc();
    auto typeConverter = getTypeConverter();

    auto numArgs = funcOp.getNumArguments();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());

    for (auto arg : llvm::enumerate(funcOp.getArguments())) {
      Type newType = typeConverter->convertType(arg.value().getType());
      Value argIndex = getIConst(rewriter, loc, arg.index());
      auto getArgValOp =
          rewriter.create<ttkernel::GetArgValOp>(loc, newType, argIndex);
      arg.value().replaceAllUsesWith(getArgValOp);
    }
    BitVector erasedArgs(numArgs, true);
    (void)funcOp.eraseArguments(erasedArgs);

    return success();
  }
};

struct ConvertAddPtrOp : public OpConversionPattern<AddPtrOp> {
  using OpConversionPattern<AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto baseAddr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (!isa<IntegerType>(baseAddr.getType()) ||
        !isa<IntegerType>(offset.getType())) {
      return failure();
    }
    auto type = cast<triton::PointerType>(op.getPtr().getType());
    auto elemType = type.getPointeeType();
    auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
    Value elemSizeValue = getI32Const(rewriter, loc, elemSize);
    offset = rewriter.create<arith::MulIOp>(loc, offset, elemSizeValue);
    auto newAddPtrOp = rewriter.create<arith::AddIOp>(loc, baseAddr, offset);
    rewriter.replaceOp(op, newAddPtrOp.getResult());

    return success();
  }
};

template <typename OpTy>
struct DeadCodeEliminationOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct ConvertGetProgramIdOp : public OpConversionPattern<GetProgramIdOp> {
  using OpConversionPattern<GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto axis = adaptor.getAxis() == ProgramIDDim::X   ? 0
                : adaptor.getAxis() == ProgramIDDim::Y ? 1
                                                       : 2;

    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto launchParamIndex =
        funcOp->getAttrOfType<IntegerAttr>("tt.num_args").getInt();
    Value paramIndexValue = getIConst(rewriter, loc, launchParamIndex + axis);
    auto launchParam = rewriter.create<ttkernel::GetArgValOp>(
        loc, rewriter.getI32Type(), paramIndexValue);
    rewriter.replaceOp(op, launchParam);

    return success();
  }
};

static bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return false;
}

class InitializationHelper {
public:
  struct ComputeOpInfo {
    ComputeOpInfo() {}
    ComputeOpInfo(Operation *op) : op(op) {}
    Value input;
    Value output;
    Operation *op;
  };

  InitializationHelper(func::FuncOp F) : funcOp(F) {}

  void insertTileRegsAcquireOps() {
    llvm::SetVector<Block *> visited;
    // TODO: this now visits copy tiles since the compute op operates directly
    // on the DST register. cleanup the naming (and possibly the algorithm)
    // here. We probably want to record some info about the compute op so we put
    // the right init in, but we might be able to do that elsewhere.
    funcOp.walk([&](ttkernel::CopyTileOp copyTilesOp) {
      Block *b = copyTilesOp->getBlock();
      if (visited.insert(b))
        insertTileRegsAcquireInBlock(b, ComputeOpInfo(copyTilesOp));
    });
  }

  void collectInputs() {
    for (auto &[_, computeOpInfo] : acquireRegistersToComputeOps) {

      auto firstComputeOperand = computeOpInfo.op->getOperand(0);
      Operation *op = firstComputeOperand.getDefiningOp();
      // TODO: I think should always have the cb feeding the compute op, but
      // leaving this somewhat generic until we confirm
      if (isCBOp(op)) {
        assert(op->getNumResults() == 1 && "expected single result for cb op");
        computeOpInfo.input = op->getResult(0);
        continue;
      }

      assert(computeOpInfo.input && "expected to find input value");
    }
  }

  void collectOutputs() {
    funcOp.walk([&](ttkernel::PackTileOp packTileOp) {
      // Note: this is completely disassociated from the compute op, which seems
      // wrong
      assert(acquireRegistersToComputeOps.size() == 1 &&
             "expecting single compute op");
      ComputeOpInfo &computeOpInfo =
          acquireRegistersToComputeOps.begin()->second;

      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      (void)getBackwardSlice(packTileOp, &backwardSlice, opt);

      for (auto op : llvm::reverse(backwardSlice)) {
        if (isCBOp(op)) {
          assert(op->getNumResults() == 1 &&
                 "expected single result for cb op");
          computeOpInfo.output = op->getResult(0);
          break;
        }
      }

      assert(computeOpInfo.output && "expected to find output value");
    });
  }

  void insertSFPUInitOps() {
    for (auto [acquireOp, computeOpInfo] : acquireRegistersToComputeOps) {
      OpBuilder builder(acquireOp);
      builder.setInsertionPoint(acquireOp);

      builder.create<ttkernel::InitSFPUOp>(
          acquireOp->getLoc(), computeOpInfo.input, computeOpInfo.output);
    }
  }

private:
  void insertTileRegsAcquireInBlock(Block *block, ComputeOpInfo computeOpInfo) {
    for (Operation &op : *block) {
      // acquire tile registers before the first copy tile
      if (isa<ttkernel::CopyTileInitOp>(op)) {
        OpBuilder builder(&op);
        builder.setInsertionPoint(&op);
        Operation *regsAcquireOp =
            builder.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
        acquireRegistersToComputeOps[regsAcquireOp] = computeOpInfo;
        return;
      }
    }
  }

  DenseMap<Operation *, ComputeOpInfo> acquireRegistersToComputeOps;
  func::FuncOp funcOp;
};

} // namespace

struct ConvertTritonNPUToTTKernelPass
    : public impl::ConvertTritonNPUToTTKernelBase<
          ConvertTritonNPUToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](gpu::MemDescType memdesc) {
      // convert memdesc to memref
      // TODO: this currently blindly changes the shape from 1024 to the
      // ttcore::TileSize, but we should encode that info into the layout and
      // use the Triton memdesc provided shape instead.
      auto shape = SmallVector<int64_t>(2, 1);
      auto ttcoreTileType = ttcore::TileType::get(
          memdesc.getContext(), ttcore::TileType::getDefaultShape(),
          ttcore::elementTypeToDataType(memdesc.getElementType()));

      MemRefType cbMemRefType = MemRefType::get(
          shape, ttcoreTileType, MemRefLayoutAttrInterface{},
          ttcore::MemorySpaceAttr::get(memdesc.getContext(),
                                       ttcore::MemorySpace::DeviceL1));
      return ttkernel::CBType::get(cbMemRefType);
    });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      auto etype = type.getElementType();
      if (isa<triton::PointerType>(etype)) {
        etype = IntegerType::get(type.getContext(), 32);
        return RankedTensorType::get(type.getShape(), etype);
      }
      if (isa<npu::tt::TileEncodingAttr>(type.getEncoding())) {
        // TODO: same caveats as above re:ttts layout
        auto shape = SmallVector<int64_t>(1, 1);
        auto ttcoreTileType = ttcore::TileType::get(
            type.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::elementTypeToDataType(etype));
        MemRefType cbMemRefType = MemRefType::get(
            shape, ttcoreTileType, MemRefLayoutAttrInterface{},
            ttcore::MemorySpaceAttr::get(type.getContext(),
                                         ttcore::MemorySpace::DeviceL1));
        return ttkernel::CBType::get(cbMemRefType);
      }
      return type;
    });
    typeConverter.addConversion([](triton::PointerType type) -> Type {
      // convert pointer to i32
      return IntegerType::get(type.getContext(), 32);
    });

    {
      mlir::ConversionTarget funcTarget(*context);
      funcTarget.addLegalDialect<func::FuncDialect>();
      funcTarget.addIllegalOp<triton::FuncOp>();
      funcTarget.addIllegalOp<triton::ReturnOp>();

      mlir::RewritePatternSet funcPatterns(context);
      populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                      PatternBenefit(1));
      if (applyPartialConversion(mod, funcTarget, std::move(funcPatterns))
              .failed())
        signalPassFailure();
    }

    //  Pass 1: TritonGPU to TTKernel
    {
      mlir::ConversionTarget target{*context};

      target.addLegalDialect<ttkernel::TTKernelDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<func::FuncDialect>();

      target.addLegalOp<UnrealizedConversionCastOp>();
      target.addDynamicallyLegalOp<func::FuncOp>(
          [](func::FuncOp funcOp) { return funcOp.getNumArguments() == 0; });

      mlir::RewritePatternSet patterns(context);
      // triton-gpu ops
      patterns.add<ConvertLoadOp>(typeConverter, patterns.getContext());
      patterns.add<ConvertStoreOp>(typeConverter, patterns.getContext());
      patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
      patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
      patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
      // triton-tt ops
      patterns.add<ConvertBinaryComputeOp>(typeConverter,
                                           patterns.getContext());
      patterns.add<ConvertAddPtrOp>(typeConverter, patterns.getContext());
      patterns.add<ConvertGetProgramIdOp>(typeConverter, patterns.getContext());
      patterns.add<DropFunctionArguments>(typeConverter, patterns.getContext());

      if (applyPartialConversion(mod, target, std::move(patterns)).failed())
        llvm::errs() << "Failed to convert TritonNPU to TTKernel\n"; // message
    }

    //  Pass 2: Dead code elimination
    {
      // Expensive iterative DCE
      // TODO: Walk once, then track inputs to recursively erase dead ops
      int cnt = 1;
      while (cnt > 0) {
        cnt = 0;
        mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (op != mod.getOperation() && isOpTriviallyDead(op)) {
            op->erase();
            cnt++;
          }
        });
      }
    }

    // insert tile regs acquire before copy tile ops
    mod.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getSymName().ends_with("__compute"))
        return;

      InitializationHelper initHelper(funcOp);
      initHelper.insertTileRegsAcquireOps();
      initHelper.collectInputs();
      initHelper.collectOutputs();
      initHelper.insertSFPUInitOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
