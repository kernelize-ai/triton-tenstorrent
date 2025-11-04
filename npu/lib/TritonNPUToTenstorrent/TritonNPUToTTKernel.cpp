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
      Value c0 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

      RankedTensorType loadType = cast<RankedTensorType>(operand.getType());
      // amusingly we could now deduce the load encoding from the operand index
      npu::tt::TileEncodingAttr loadEncoding =
          cast<npu::tt::TileEncodingAttr>(loadType.getEncoding());
      Value destRegisterIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  loadEncoding.getIndex()));
      rewriter.create<ttkernel::CopyTileOp>(loc, convertedOperand, c0,
                                            destRegisterIndex);
    };

    copyAndInitializeRegister(op.getLhs(), lhs);
    copyAndInitializeRegister(op.getRhs(), rhs);

    rewriter.create<ttkernel::AddBinaryTilesInitOp>(loc);
    Value lhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value rhsIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    Value destIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
    rewriter.create<ttkernel::AddBinaryTilesOp>(loc, lhsIndex, rhsIndex,
                                                destIndex);

    rewriter.eraseOp(op);
    return success();
  }
};

static Value traceToScalar(Value ptr, bool isPtr = true) {
  auto isScalar = [](Value v) { return !isa<RankedTensorType>(v.getType()); };
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

struct ConvertLocalStoreOp : public OpConversionPattern<gpu::LocalStoreOp> {
  using OpConversionPattern<gpu::LocalStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dst = adaptor.getDst();
    auto srcOp = op.getSrc().getDefiningOp();

    if (!isa<triton::LoadOp>(srcOp)) {
      auto cbType = cast<ttkernel::CBType>(dst.getType());
      MemRefType cbTileMemref = cbType.getMemref();

      // add the one-D reinterpret cast at the compile time arg site for now
      rewriter.setInsertionPointAfter(dst.getDefiningOp());
      MemRefType oneDTileType = MemRefType::get(
          {1}, cbTileMemref.getElementType(), MemRefLayoutAttrInterface{},
          cbTileMemref.getMemorySpace());
      auto oneDTile = rewriter.create<ttkernel::CBReinterpretShapeOp>(
          loc, ttkernel::CBType::get(rewriter.getContext(), oneDTileType), dst);

      // reserve back the cb for the store
      Value numPages = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
      rewriter.create<ttkernel::CBReserveBackOp>(loc, dst, numPages);

      rewriter.setInsertionPoint(op);
      Value destRegisterIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
      Value outIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
      rewriter.create<ttkernel::PackTileOp>(loc, destRegisterIndex, oneDTile,
                                            outIndex,
                                            /*outOfOrder=*/true);

      rewriter.eraseOp(op);
      return success();
    }

    // FOR PATTERN: tt.load -> ttg.local_store
    auto loadOp = cast<triton::LoadOp>(srcOp);

    // ASSUME: tilize has padded to full tiles. Drop masking.
    Value baseAddr = traceToScalar(loadOp.getPtr(), true);
    Value offset = traceToScalar(loadOp.getPtr(), false);

    Value const1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    Value const0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

    // Compute page size in bytes
    auto dataFormat = rewriter.create<ttkernel::GetDataFormatOp>(loc, dst);
    auto pageSize = rewriter.create<ttkernel::GetTileSizeOp>(loc, dst);

    // 0. Create tensor accessor
    // TODO: move to top scope
    Value c1bit = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    Value addrGen = rewriter.create<ttkernel::GetInterleavedAddrGenFastOp>(
        loc, c1bit, baseAddr, pageSize, dataFormat);

    // 1. reserve back on the cb to know data is ready to store to SRAM
    //       ttkernel.cb_reserve_back(%0, %c1_i32)
    rewriter.create<ttkernel::CBReserveBackOp>(loc, dst, const1);

    // 3. get L1 address
    //       %11 = ttkernel.get_write_ptr(%0)
    Value l1Addr = rewriter.create<ttkernel::GetWritePtrOp>(loc, dst);

    // 4. async read from noc to l1, size in bytes
    //       ttkernel.noc_async_read(%10, %13, %c4096_i32)
    rewriter.create<ttkernel::NocAsyncReadTileOp>(loc, offset, addrGen, l1Addr);

    // 5. barrier to ensure data is read from noc to l1
    //       ttkernel.noc_async_read_barrier() : () -> ()
    rewriter.create<ttkernel::NocAsyncReadBarrierOp>(loc);

    // 6. push back on the cb to know data is ready to load from SRAM
    //       ttkernel.cb_push_back(%0, %c1_i32)
    rewriter.create<ttkernel::CBPushBackOp>(loc, dst, const1);

    rewriter.eraseOp(op);

    // cleanup the global load op
    auto mask = loadOp.getMask();
    rewriter.eraseOp(loadOp);

    // Masking is handled by tilize, so we can just erase the mask op
    if (mask && mask.getDefiningOp()) {
      rewriter.eraseOp(mask.getDefiningOp());
    }

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

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());

    for (auto arg : llvm::enumerate(funcOp.getArguments())) {
      Type newType = typeConverter->convertType(arg.value().getType());
      Value argIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), arg.index()));
      auto getArgValOp =
          rewriter.create<ttkernel::GetArgValOp>(loc, newType, argIndex);
      arg.value().replaceAllUsesWith(getArgValOp);
    }
    BitVector erasedArgs(funcOp.getNumArguments(), true);
    (void)funcOp.eraseArguments(erasedArgs);

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
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
    auto waitFrontOp =
        rewriter.create<ttkernel::CBWaitFrontOp>(loc, adaptor.getSrc(), c1);

    LDBG("Converted load src type = " << adaptor.getSrc().getType() << "\n");
    assert(isa<ttkernel::CBType>(adaptor.getSrc().getType()) &&
           "expected memref type for type converted load src");
    auto cbType = cast<ttkernel::CBType>(adaptor.getSrc().getType());
    MemRefType cbTileMemref = cbType.getMemref();

    // 1.5 reinterpret the cb from 2D to 1D tile shape
    // TODO: materialize this as a reshape op?
    MemRefType oneDTileType = MemRefType::get(
        {1}, cbTileMemref.getElementType(), MemRefLayoutAttrInterface{},
        cbTileMemref.getMemorySpace());
    auto oneDTile =
        rewriter
            .create<ttkernel::CBReinterpretShapeOp>(
                loc, ttkernel::CBType::get(rewriter.getContext(), oneDTileType),
                adaptor.getSrc())
            .getResult();

    auto user = *dst.getUsers().begin();

    if (dst.hasOneUse() && isa<triton::StoreOp>(user)) {
      // FOR PATTERN: ttg.local_load -> tt.store
      auto storeOp = cast<triton::StoreOp>(user);
      auto mask = storeOp.getMask();
      if (mask && mask.getDefiningOp()) {
        rewriter.eraseOp(mask.getDefiningOp());
      }
      Value baseAddr = traceToScalar(storeOp.getPtr(), true);
      Value offset = traceToScalar(storeOp.getPtr(), false);
      // Compute page size in bytes
      auto dataFormat = rewriter.create<ttkernel::GetDataFormatOp>(loc, src);
      auto pageSize = rewriter.create<ttkernel::GetTileSizeOp>(loc, src);

      // 0. Create tensor accessor
      // TODO: move to top scope
      Value c1bit = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI1Type(),
          rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value addrGen = rewriter.create<ttkernel::GetInterleavedAddrGenFastOp>(
          loc, c1bit, baseAddr, pageSize, dataFormat);

      Value l1Addr = rewriter.create<ttkernel::GetWritePtrOp>(loc, oneDTile);
      rewriter.create<ttkernel::NocAsyncWriteTileOp>(loc, offset, addrGen,
                                                     l1Addr);
      rewriter.create<ttkernel::NocAsyncWriteBarrierOp>(loc);
      rewriter.create<ttkernel::CBPopFrontOp>(loc, oneDTile, c1);
      rewriter.eraseOp(user);
    }
    // TODO: this should come from the cb root eventually,
    // but replaceOp is probably not appropriate here
    rewriter.replaceOp(op, oneDTile);
    return success();
  }
};

struct ConvertLocalAllocOp : public OpConversionPattern<gpu::LocalAllocOp> {
  using OpConversionPattern<gpu::LocalAllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // replace the local allocs with cbs from the function signature
    auto allocIdx = op->getAttrOfType<IntegerAttr>("alloc_idx");
    if (!allocIdx) {
      return rewriter.notifyMatchFailure(op, "missing alloc_idx attribute");
    }
    int64_t allocIdxValue = allocIdx.getInt();

    auto typeConverter = getTypeConverter();

    LDBG("Local alloc op result type: " << op.getResult().getType() << "\n");
    Type convertedType = typeConverter->convertType(op.getResult().getType());
    LDBG("is converted to memref type: " << convertedType << "\n");
    auto cbMemRefType = cast<ttkernel::CBType>(convertedType);

    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(op, cbMemRefType,
                                                              allocIdxValue);

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

    // TODO: map from virtual grid to physical grid
    // - this is a hack to get the program id from the my_x and my_y ops
    Value programId;
    if (adaptor.getAxis() == ProgramIDDim::X) {
      programId =
          rewriter.create<ttkernel::MyXOp>(loc, /*Optional noc=*/Value());
    } else if (adaptor.getAxis() == ProgramIDDim::Y) {
      programId =
          rewriter.create<ttkernel::MyYOp>(loc, /*Optional noc=*/Value());
    } else {
      llvm_unreachable("unsupported program id dimension");
    }

    auto castOp = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), programId);
    rewriter.replaceOp(op, castOp.getResult());

    return success();
  }
};

static bool isCBOp(Operation *op) {
  if (auto compileTimeArg = dyn_cast<ttkernel::GetCompileArgValOp>(op)) {
    return isa<ttkernel::CBType>(compileTimeArg.getType());
  }
  return isa<ttkernel::CBReinterpretShapeOp>(op);
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
      return ttkernel::CBType::get(memdesc.getContext(), cbMemRefType);
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
        return ttkernel::CBType::get(type.getContext(), cbMemRefType);
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
    llvm::errs() << "Pass 0: TritonFunc to Func completed\n";

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
    llvm::errs() << "Pass 1: TritonNPU to TTKernel completed\n";

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
    llvm::errs() << "Pass 2: Dead code elimination completed\n";

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
