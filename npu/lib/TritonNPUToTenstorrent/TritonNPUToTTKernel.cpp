#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
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

static Value getI32Const(OpBuilder &rewriter, Location loc, int64_t value) {
  return arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), value));
}

static Value getI1Const(OpBuilder &rewriter, Location loc, bool value) {
  return arith::ConstantOp::create(
      rewriter, loc, rewriter.getI1Type(),
      rewriter.getIntegerAttr(rewriter.getI1Type(), value));
}

static Value getIConst(OpBuilder &rewriter, Location loc, int64_t value) {
  return arith::ConstantOp::create(
      rewriter, loc, rewriter.getIndexType(),
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

    ttkernel::AddBinaryTilesInitOp::create(rewriter, loc);
    Value lhsIndex = getIConst(rewriter, loc, 0);
    Value rhsIndex = getIConst(rewriter, loc, 1);
    Value destIndex = getIConst(rewriter, loc, 2);
    ttkernel::AddBinaryTilesOp::create(rewriter, loc, lhsIndex, rhsIndex,
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
  offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);

  Value tile_id = arith::DivUIOp::create(rewriter, loc, offset, pageSize);

  Value const1 = getI32Const(rewriter, loc, 1);
  Value const0 = getI32Const(rewriter, loc, 0);

  // Create interleaved address generator and get noc address
  auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
  Value c1bit = getI1Const(rewriter, loc, 1);
  Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
      rewriter, loc, c1bit, baseAddr, pageSize, dataFormat);
  Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
      rewriter, loc, addrGen, tile_id, const0, Value());

  return nocAddr;
}

struct ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    if (!op->hasOneUse()) {
      LDBG("Load op has multiple uses, cannot convert\n");
      return failure();
    }
    Operation *user = *op->getUsers().begin();
    if (!isa<gpu::LocalStoreOp>(user)) {
      LDBG("Load op user is not a local store op: " << *user << "\n");
      return failure();
    }

    Value cb =
        rewriter.getRemappedValue(cast<gpu::LocalStoreOp>(user).getDst());

    // Compute page size in bytes
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    // Compute the noc address
    Value nocAddr = computeNocAddr(rewriter, loc, op.getPtr(), pageSize, cb);

    // 1. reserve back on the cb to know data is ready to store to SRAM
    //       ttkernel.cb_reserve_back(%0, %c1_i32)
    // add later?
    Value const1 = getI32Const(rewriter, loc, 1);
    ttkernel::CBReserveBackOp::create(rewriter, loc, cb, const1);

    // 3. get L1 address
    //       %11 = ttkernel.get_write_ptr(%0)
    Value l1Addr = ttkernel::GetWritePtrOp::create(rewriter, loc, cb);

    // 4. async read from noc to l1, size in bytes
    //       ttkernel.noc_async_read(%10, %13, %c4096_i32)
    ttkernel::NocAsyncReadOp::create(rewriter, loc, nocAddr, l1Addr, pageSize);

    // 5. barrier to ensure data is read from noc to l1
    //       ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel::NocAsyncReadBarrierOp::create(rewriter, loc);

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
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value nocAddr = computeNocAddr(rewriter, loc, op.getPtr(), pageSize, cb);

    Value l1Addr = ttkernel::GetReadPtrOp::create(rewriter, loc, cb);
    ttkernel::NocAsyncWriteOp::create(rewriter, loc, l1Addr, nocAddr, pageSize);
    ttkernel::NocAsyncWriteBarrierOp::create(rewriter, loc);
    Value numPages = getI32Const(rewriter, loc, 1);
    ttkernel::CBPopFrontOp::create(rewriter, loc, cb, numPages);
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
      ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numPages);

      // Pack the tile into the cb
      Value destRegisterIndex = getIConst(rewriter, loc, 2);
      Value outIndex = getIConst(rewriter, loc, 0);
      ttkernel::PackTileOp::create(rewriter, loc, destRegisterIndex, dst,
                                   outIndex,
                                   /*outOfOrder=*/true);

      rewriter.eraseOp(op);
      return success();
    }

    Value const1 = getI32Const(rewriter, loc, 1);
    //       ttkernel.cb_push_back(%0, %c1_i32)
    ttkernel::CBPushBackOp::create(rewriter, loc, dst, const1);

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

    LDBG("Converted load src type = " << src.getType() << "\n");
    assert(isa<ttkernel::CBType>(src.getType()) &&
           "expected memref type for type converted load src");

    assert(op->hasOneUse() &&
           "expected local load with store user to have one use");
    if (isa<StoreOp>(*op->getUsers().begin())) {
      // wait front on the cb to know data is ready
      Value numPages = getI32Const(rewriter, loc, 1);
      auto waitFrontOp =
          ttkernel::CBWaitFrontOp::create(rewriter, loc, src, numPages);
      rewriter.replaceOp(op, src);
      return success();
    }

    ttkernel::CopyTileInitOp::create(rewriter, loc, src);
    Value c0 = getIConst(rewriter, loc, 0);

    auto dst = op.getResult();
    auto dstType = cast<RankedTensorType>(dst.getType());
    npu::tt::TileEncodingAttr loadEncoding =
        cast<npu::tt::TileEncodingAttr>(dstType.getEncoding());
    Value destRegisterIndex = getIConst(rewriter, loc, loadEncoding.getIndex());
    ttkernel::CopyTileOp::create(rewriter, loc, src, c0, destRegisterIndex);
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
          ttkernel::GetArgValOp::create(rewriter, loc, newType, argIndex);
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
    offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);
    auto newAddPtrOp = arith::AddIOp::create(rewriter, loc, baseAddr, offset);
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
    auto launchParam = ttkernel::GetArgValOp::create(
        rewriter, loc, rewriter.getI32Type(), paramIndexValue);
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

static bool requiresSFPUInit(func::FuncOp funcOp) {
  bool requiresInit = false;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<ttkernel::TTKernelBinaryOpTrait>()) {
      requiresInit = true;
      return;
    }
  });
  return requiresInit;
}

class InitializationHelper {
public:
  InitializationHelper(func::FuncOp F)
      : funcOp(F), addSFPUInit(requiresSFPUInit(F)) {
    // build maps of copy/pack ops to respective indices
    funcOp.walk([&](ttkernel::CopyTileOp copyTileOp) {
      Value cbIndex = copyTileOp.getTileIndexCb();
      Value dstIndex = copyTileOp.getTileIndexDst();
      copyTileOps[copyTileOp]++;
    });

    funcOp.walk([&](ttkernel::CopyTileInitOp copyTileInitOp) {
      copyTileInitOps.push_back(copyTileInitOp);
    });

    funcOp.walk([&](ttkernel::PackTileOp packTileOp) {
      Value cbIndex = packTileOp.getDstIndex();
      packTileOps[packTileOp]++;
    });
  }

  // tile regs aquire ops must be inserted before any copy tiles ops
  void insertTileRegsAcquireOps() {
    assert(!copyTileInitOps.empty() &&
           "expecting at least one copy tile init op");
    OpBuilder builder(copyTileInitOps.front());
    ttkernel::TileRegsAcquireOp::create(builder,
                                        copyTileInitOps.front().getLoc());
  }

  // coalesce copy tile waits before the firsts copy tile ops since tile
  // registers may not be acquired yet
  void insertCopyTileWaits() {
    OpBuilder builder(copyTileInitOps.front());
    for (auto copyTileOpItr : copyTileOps) {
      ttkernel::CopyTileOp copyTileOp = copyTileOpItr.first;
      Value numTiles =
          getI32Const(builder, copyTileOp->getLoc(), copyTileOpItr.second);
      Value cb = copyTileOp.getCb0();
      ttkernel::CBWaitFrontOp::create(builder, copyTileOp->getLoc(), cb,
                                      numTiles);
    }
  }

  void insertSFPUInitOps() {
    if (!addSFPUInit)
      return;

    auto tileRegsAcquireOps = funcOp.getOps<ttkernel::TileRegsAcquireOp>();
    assert(!tileRegsAcquireOps.empty() && "expecting tile regs acquire op");
    Operation *acquireOp = *tileRegsAcquireOps.begin();

    OpBuilder builder(acquireOp);
    Value inCb = copyTileOps.begin()->first.getCb0();
    Value outCb = packTileOps.begin()->first.getOutCb();

    ttkernel::InitSFPUOp::create(builder, acquireOp->getLoc(), inCb, outCb);
  }

private:
  SmallVector<ttkernel::CopyTileInitOp, 4> copyTileInitOps;
  llvm::MapVector<ttkernel::CopyTileOp, unsigned> copyTileOps;
  llvm::MapVector<ttkernel::PackTileOp, unsigned> packTileOps;

  func::FuncOp funcOp;
  const bool addSFPUInit;
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
      initHelper.insertCopyTileWaits();
      initHelper.insertTileRegsAcquireOps();
      initHelper.insertSFPUInitOps();
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
