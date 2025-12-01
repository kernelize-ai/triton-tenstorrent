#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "Utility.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

inline bool isScalar(Value v) { return !isa<RankedTensorType>(v.getType()); }

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

inline Value traceToBaseAddress(Value ptr) {
  SetVector<Operation *> baseAddrSlice;
  mlir::BackwardSliceOptions opt;
  opt.filter = [](Operation *op) { return !isa<ttkernel::GetArgValOp>(op); };
  (void)getBackwardSlice(ptr, &baseAddrSlice, opt);
  LLVM_DEBUG(for (Operation *op : baseAddrSlice) {
    DBGS() << "backward slice op: " << *op << "\n";
  });

  Value baseAddr;
  for (auto op : baseAddrSlice) {
    if (op->getNumOperands() == 1 &&
        isa<IntegerType>(op->getOperand(0).getType())) {
      baseAddr = op->getOperand(0);
      break;
    }
  }

  assert(baseAddr && "could not find base address in backward slice");
  LDBG("Found base address: " << baseAddr << ", for ptr: " << ptr);
  return baseAddr;
}

static Value computeNocAddr(ConversionPatternRewriter &rewriter, Location loc,
                            Value ptr, Value pageSize, Value cb) {
  // Trace to base address and offset
  LDBG("Computing NOC address for ptr: " << ptr);

  SetVector<Operation *> baseAddrSlice;
  mlir::BackwardSliceOptions opt;
  opt.filter = [](Operation *op) {
    return !isa<IntegerType>(op->getResult(0).getType());
  };
  (void)getBackwardSlice(ptr, &baseAddrSlice, opt);
  LLVM_DEBUG(for (Operation *op : baseAddrSlice) {
    DBGS() << "backward slice op: " << *op << "\n";
  });

  Value baseAddr, offset;
  // Look for splat ops that populate tensors with integer values. These ops
  // give us the base ptr (converted to integer) and the offset (integer).
  // Because the slice frontier is an integer type we know that only one op with
  // integer valued return can exist for both base addr and offset.
  for (auto op : baseAddrSlice) {
    if (op->getNumOperands() == 1 &&
        isa<IntegerType>(op->getOperand(0).getType())) {
      if (!baseAddr) {
        baseAddr = op->getOperand(0);
      } else {
        offset = op->getOperand(0);
        break;
      }
    }
  }

  assert(baseAddr && "could not find base address in backward slice");
  assert(offset && "could not find offset in backward slice");
  LDBG("Computing NOC address for base address: " << baseAddr
                                                  << ", offset: " << offset);
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
  Value elemSizeValue = arith::createConstantI32(
      loc, rewriter, elemType.getIntOrFloatBitWidth() / 8);
  offset = arith::MulIOp::create(rewriter, loc, offset, elemSizeValue);

  Value tile_id = arith::DivUIOp::create(rewriter, loc, offset, pageSize);

  Value const1 = arith::createConstantI32(loc, rewriter, 1);
  Value const0 = arith::createConstantI32(loc, rewriter, 0);

  // Create interleaved address generator and get noc address
  auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
  Value c1bit = arith::createConstantI1(loc, rewriter, 1);
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

    LDBG("Converting load op: " << *op << "\n");

    Value cb =
        rewriter.getRemappedValue(cast<gpu::LocalStoreOp>(user).getDst());

    Value baseAddr = traceToBaseAddress(op.getPtr());
    LDBG("Ptr adaptor value: " << adaptor.getPtr());

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value c1bit = arith::createConstantI1(loc, rewriter, 1);
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, c1bit, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // convert bytes offset to tile index
    Value offset = adaptor.getPtr();
    Value tile_id = arith::DivUIOp::create(rewriter, loc, offset, pageSize);

    Value const1 = arith::createConstantI32(loc, rewriter, 1);
    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
        rewriter, loc, addrGen, tile_id, const0, Value());

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
    Value numPages = arith::createConstantI32(loc, rewriter, 1);
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
      Value numPages = arith::createConstantI32(loc, rewriter, 1);
      ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numPages);

      // Pack the tile into the cb
      Value destRegisterIndex = arith::createIndexConstant(loc, rewriter, 2);
      Value outIndex = arith::createIndexConstant(loc, rewriter, 0);
      ttkernel::PackTileOp::create(rewriter, loc, destRegisterIndex, dst,
                                   outIndex,
                                   /*outOfOrder=*/true);

      rewriter.eraseOp(op);
      return success();
    }

    Value const1 = arith::createConstantI32(loc, rewriter, 1);
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
      Value numPages = arith::createConstantI32(loc, rewriter, 1);
      auto waitFrontOp =
          ttkernel::CBWaitFrontOp::create(rewriter, loc, src, numPages);
      rewriter.replaceOp(op, src);
      return success();
    }

    ttkernel::CopyTileInitOp::create(rewriter, loc, src);
    Value c0 = arith::createIndexConstant(loc, rewriter, 0);

    auto dst = op.getResult();
    auto dstType = cast<RankedTensorType>(dst.getType());
    npu::tt::TileEncodingAttr loadEncoding =
        cast<npu::tt::TileEncodingAttr>(dstType.getEncoding());
    Value destRegisterIndex =
        arith::createIndexConstant(loc, rewriter, loadEncoding.getIndex());
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

} // namespace

void populateMemoryOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertStoreOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
