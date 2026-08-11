#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "PointerInfoAnalysis.h"
#include "Utility.h"

// Atomic ops on the Tenstorrent NPU backend.
//
// Hardware reality this lowering works within: the NoC has exactly one real
// atomic RMW primitive end-to-end -- a fetch-and-add ("noc_semaphore_inc")
// against an arbitrary 4-byte-aligned *L1* address, which lands its
// pre-increment value in a fixed, well-known local L1 slot on the issuing
// core (tt-metal's MEM_NOC_ATOMIC_RET_VAL_ADDR, identical across
// wormhole/blackhole/quasar). There is no hardware CAS/exchange/max/min/
// and/or/xor, and DRAM has no atomic unit at all. So every atomic op here is
// built as: acquire a software spinlock (itself built from the one real
// fetch-and-add primitive), do an ordinary non-atomic NoC read + scalar
// arith compute + NoC write against the target element, release the lock.
//
// Scope (see kurs/plan discussion): only scalar (non-tensor-shaped) 32-bit
// int/float atomics on a uniform address are supported -- e.g. a global
// counter, or a split-K-style accumulation into a shared output element.
// Tensor-shaped (per-lane scatter) atomics are not lowered here; that needs
// general indirect addressing this backend does not otherwise support.

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Fixed local-L1 byte offset that every NoC atomic-increment transaction
// writes its pre-increment ("old") value to on the issuing core. Defined
// identically (=4) for wormhole/blackhole/quasar in tt-metal's
// dev_mem_map.h; not modeled as a TTKernel op operand today, so it is
// referenced here as the raw constant tt-metal itself uses.
constexpr int32_t kAtomicRetValL1Addr = 4;

// Generous bound on lock-acquire retries so contention cannot spin forever
// in codegen; exceeding it (only under extreme, sustained cross-core
// contention on the same static atomic op) falls through to the critical
// section without a confirmed lock, which is a known, documented limitation
// of this software-spinlock v1 rather than a hard guarantee.
constexpr int32_t kMaxLockRetries = 100000;

// Every atomic op instance (i.e. every distinct static `tt.atomic_rmw` /
// `tt.atomic_cas` in the program) gets its own dedicated lock + scratch pair,
// hosted on a fixed, well-known "owner" core (logical (0,0), guaranteed part
// of any non-empty grid). All cores that execute *that* static op across
// grid iterations contend for the same lock, giving the needed cross-core
// mutual exclusion. Two different static atomic ops that happen to target
// the same runtime buffer get independent locks; this is an accepted v1
// simplification (the dominant real pattern is one static atomic call site
// executed repeatedly across grid iterations, not multiple call sites
// racing on the same address).
Value getLockOwnerNocAddr(ConversionPatternRewriter &rewriter, Location loc,
                          Value semaphore, Value nocId) {
  Value logicalZero = arith::createIndexConstant(loc, rewriter, 0);
  Value ownerX = ttkernel::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), logicalZero);
  Value ownerY = ttkernel::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), logicalZero);
  return ttkernel::GetNocAddrOp::create(rewriter, loc, ownerX, ownerY,
                                        semaphore, nocId);
}

// Read back the pre-increment value of the most recently issued (and
// barriered) atomic increment on this core.
Value readAtomicRetVal(ConversionPatternRewriter &rewriter, Location loc) {
  MLIRContext *ctx = rewriter.getContext();
  Value retAddr =
      arith::createConstantI32(loc, rewriter, kAtomicRetValL1Addr);
  Value retPtr = ttkernel::CastToL1PtrOp::create(
      rewriter, loc, ttkernel::L1AddrPtrType::get(ctx, 32), retAddr);
  Value zeroOffset = arith::createConstantI32(loc, rewriter, 0);
  return ttkernel::LoadFromL1Op::create(rewriter, loc, rewriter.getI32Type(),
                                        retPtr, zeroOffset);
}

// Build the bounded acquire-retry loop for the lock at `lockNocAddr`. Each
// iteration only attempts the increment if not already acquired (so a
// successful acquire on iteration K makes every later iteration a no-op),
// and unconditionally issues a compensating +0/-1 increment so no nested
// conditional is needed to roll a losing attempt back.
void emitLockAcquire(ConversionPatternRewriter &rewriter, Location loc,
                     Value lockNocAddr, Value nocId) {
  Value i1False = arith::createConstantI1(loc, rewriter, false);
  Value lb = arith::createConstantI32(loc, rewriter, 0);
  Value ub = arith::createConstantI32(loc, rewriter, kMaxLockRetries);
  Value step = arith::createConstantI32(loc, rewriter, 1);
  Value oneIdx = arith::createIndexConstant(loc, rewriter, 1);
  Value zeroIdx = arith::createIndexConstant(loc, rewriter, 0);
  Value minusOneIdx = arith::createIndexConstant(loc, rewriter, -1);
  Value i32Zero = arith::createConstantI32(loc, rewriter, 0);

  scf::ForOp acquireLoop = scf::ForOp::create(rewriter, loc, lb, ub, step,
                                              ValueRange{i1False});
  {
    rewriter.setInsertionPointToStart(acquireLoop.getBody());
    Value acquiredSoFar = acquireLoop.getRegionIterArgs()[0];
    Value notAcquired = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, acquiredSoFar,
        arith::createConstantI1(loc, rewriter, false));

    scf::IfOp tryAcquire = scf::IfOp::create(
        rewriter, loc, TypeRange{rewriter.getI1Type()}, notAcquired,
        /*withElseRegion=*/true);
    {
      rewriter.setInsertionPointToStart(tryAcquire.thenBlock());
      ttkernel::NocSemaphoreIncOp::create(rewriter, loc, lockNocAddr, oneIdx,
                                          nocId);
      ttkernel::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocId);
      Value prevVal = readAtomicRetVal(rewriter, loc);
      Value won = arith::CmpIOp::create(rewriter, loc,
                                        arith::CmpIPredicate::eq, prevVal,
                                        i32Zero);
      Value rollbackAmt =
          arith::SelectOp::create(rewriter, loc, won, zeroIdx, minusOneIdx);
      ttkernel::NocSemaphoreIncOp::create(rewriter, loc, lockNocAddr,
                                          rollbackAmt, nocId);
      ttkernel::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocId);
      scf::YieldOp::create(rewriter, loc, ValueRange{won});
    }
    {
      rewriter.setInsertionPointToStart(tryAcquire.elseBlock());
      scf::YieldOp::create(rewriter, loc, ValueRange{acquiredSoFar});
    }
    rewriter.setInsertionPointAfter(tryAcquire);
    scf::YieldOp::create(rewriter, loc, ValueRange{tryAcquire.getResult(0)});
  }
  rewriter.setInsertionPointAfter(acquireLoop);
}

void emitLockRelease(ConversionPatternRewriter &rewriter, Location loc,
                     Value lockNocAddr, Value nocId) {
  Value minusOneIdx = arith::createIndexConstant(loc, rewriter, -1);
  ttkernel::NocSemaphoreIncOp::create(rewriter, loc, lockNocAddr, minusOneIdx,
                                      nocId);
  ttkernel::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocId);
}

// Build the full lock-protected read/compute/write critical section for a
// scalar atomic op and return the pre-mutation ("old") value as raw i32
// bits. `computeNewBits` receives the old raw bits and must return the new
// raw bits to store back.
//
// `op` must have a scalar (non-tensor) `ptr` operand; `convertedPtr` is that
// operand after type conversion (an i32 byte address, base + offset already
// merged per the scalar branch of ConvertAddPtrOp). `mask`, if non-null, is
// the (already scalar, since ptr is scalar) i1 predicate guarding the whole
// op; when false the op is skipped and a default zero is returned.
Value emitAtomicOp(
    ConversionPatternRewriter &rewriter, Operation *op,
    PointerInfoAnalysis *pointerInfoAnalysis, Value convertedPtr, Value mask,
    llvm::function_ref<Value(ConversionPatternRewriter &, Location, Value)>
        computeNewBits) {
  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Type i32Ty = rewriter.getI32Type();

  auto ptrInfo = pointerInfoAnalysis->getInfo(op);
  assert(ptrInfo && "expected pointer info for atomic op");
  Value baseAddr = ptrInfo->basePtr;

  // Recover the pure byte-offset-from-tensor-start: unlike the tensor
  // (tile-load/store) AddPtr lowering, the *scalar* AddPtr lowering keeps
  // the base address folded into the value (see ConvertAddPtrOp's scalar
  // branch in ElementwiseOpsToTTKernel.cpp), so subtract it back out here.
  Value elemSize = arith::createConstantI32(loc, rewriter, 4);
  Value pureOffset =
      arith::SubIOp::create(rewriter, loc, convertedPtr, baseAddr);
  Value elemIndex = arith::DivUIOp::create(rewriter, loc, pureOffset, elemSize);
  Value elemIndexAsIndex = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getIndexType(), elemIndex);

  auto opInsertionPt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfterValue(baseAddr);
  // One "page" per element: reuses the same TensorAccessor machinery
  // load/store use for tile-granular DRAM/L1 addressing, just at 4-byte
  // page granularity instead of a full tile. The page-size constant is
  // created at this same (early) insertion point -- mirroring how
  // ConvertLoadOp creates its `pageSize` right where the accessor is built
  // -- so it dominates the accessor regardless of where in the function the
  // atomic op itself lives.
  Value pageSizeForAccessor = arith::createConstantI32(loc, rewriter, 4);
  Value accessor =
      createTensorAccessor(rewriter, loc, op, baseAddr, pageSizeForAccessor);
  rewriter.restoreInsertionPoint(opInsertionPt);

  Value nocId = createNocId(rewriter, loc, getKernelNocIndex(op));

  // Reserve two fresh compile-time LocalSemaphore args for this atomic op
  // instance: one as the mutex lock word, one as read/write scratch for the
  // target element. Mirrors ConvertMulticastOp's semaphore reservation
  // (MemoryOpToTTKernel.cpp), which is the proven pattern for handing a
  // dialect-conversion pattern a fresh, host-allocated 4-byte L1 slot.
  auto parentFuncOp = op->getParentOfType<func::FuncOp>();
  assert(parentFuncOp && "expected atomic op inside a kernel func");
  auto argSpec = parentFuncOp->getAttrOfType<ttkernel::ArgSpecAttr>(
      ttkernel::ArgSpecAttr::name);
  SmallVector<ttkernel::ArgAttr> ctArgs;
  if (argSpec)
    ctArgs = llvm::to_vector(argSpec.getCtArgs());
  int32_t lockCtIdx = ctArgs.size();
  int32_t scratchCtIdx = ctArgs.size() + 1;

  Value lockIdxVal =
      ttkernel::GetCompileArgValOp::create(rewriter, loc, i32Ty, lockCtIdx);
  Value lockSem = ttkernel::GetSemaphoreOp::create(rewriter, loc, lockIdxVal);
  Value scratchIdxVal = ttkernel::GetCompileArgValOp::create(
      rewriter, loc, i32Ty, scratchCtIdx);
  Value scratchSem =
      ttkernel::GetSemaphoreOp::create(rewriter, loc, scratchIdxVal);

  rewriter.modifyOpInPlace(parentFuncOp, [&]() {
    ttkernel::ArgSpecAttr::appendCompileTimeArg(
        parentFuncOp, rewriter.getAttr<ttkernel::ArgAttr>(
                          ttkernel::ArgType::LocalSemaphore, 0));
    ttkernel::ArgSpecAttr::appendCompileTimeArg(
        parentFuncOp, rewriter.getAttr<ttkernel::ArgAttr>(
                          ttkernel::ArgType::LocalSemaphore, 0));
  });

  Value lockNocAddr = getLockOwnerNocAddr(rewriter, loc, lockSem, nocId);

  auto buildCriticalSection = [&]() -> Value {
    emitLockAcquire(rewriter, loc, lockNocAddr, nocId);

    ttkernel::NocAsyncReadTileOp::create(rewriter, loc, elemIndex, accessor,
                                        scratchSem, nocId);
    ttkernel::NocAsyncReadBarrierOp::create(rewriter, loc, nocId);
    Value scratchL1Ptr = ttkernel::CastToL1PtrOp::create(
        rewriter, loc, ttkernel::L1AddrPtrType::get(ctx, 32), scratchSem);
    Value zeroOffset = arith::createConstantI32(loc, rewriter, 0);
    Value oldBits = ttkernel::LoadFromL1Op::create(rewriter, loc, i32Ty,
                                                   scratchL1Ptr, zeroOffset);

    Value newBits = computeNewBits(rewriter, loc, oldBits);

    ttkernel::StoreToL1Op::create(rewriter, loc, newBits, scratchL1Ptr,
                                  zeroOffset);
    ttkernel::NocAsyncWriteTileOp::create(rewriter, loc, elemIndexAsIndex,
                                         accessor, scratchSem, nocId);
    ttkernel::NocAsyncWriteBarrierOp::create(rewriter, loc, nocId);

    emitLockRelease(rewriter, loc, lockNocAddr, nocId);
    return oldBits;
  };

  if (!mask)
    return buildCriticalSection();

  scf::IfOp guarded = scf::IfOp::create(rewriter, loc, TypeRange{i32Ty}, mask,
                                        /*withElseRegion=*/true);
  {
    rewriter.setInsertionPointToStart(guarded.thenBlock());
    Value oldBits = buildCriticalSection();
    scf::YieldOp::create(rewriter, loc, ValueRange{oldBits});
  }
  {
    rewriter.setInsertionPointToStart(guarded.elseBlock());
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{arith::createConstantI32(loc, rewriter, 0)});
  }
  rewriter.setInsertionPointAfter(guarded);
  return guarded.getResult(0);
}

struct ConvertAtomicRMWOp : public OpConversionPattern<triton::AtomicRMWOp> {
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

  explicit ConvertAtomicRMWOp(TypeConverter &typeConverter,
                              npu::PointerInfoAnalysis *pointerInfoAnalysis,
                              MLIRContext *context)
      : OpConversionPattern<triton::AtomicRMWOp>(typeConverter, context),
        pointerInfoAnalysis(pointerInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getPtr().getType()))
      return rewriter.notifyMatchFailure(
          op, "tensor-shaped (per-lane scatter) atomic_rmw is not "
              "supported; only scalar atomics on a uniform address are "
              "lowered on this backend");

    Type elemType = op.getVal().getType();
    bool isFloat = isa<Float32Type>(elemType);
    if (!(elemType.isInteger(32) || isFloat))
      return rewriter.notifyMatchFailure(
          op, "only 32-bit int/float atomic_rmw is supported");

    Location loc = op.getLoc();
    Value mask = op.getMask() ? adaptor.getMask() : Value();
    triton::RMWOp kind = op.getAtomicRmwOp();
    Value val = adaptor.getVal();

    Value oldBits = emitAtomicOp(
        rewriter, op, pointerInfoAnalysis, adaptor.getPtr(), mask,
        [&](ConversionPatternRewriter &b, Location l, Value old) -> Value {
          switch (kind) {
          case triton::RMWOp::AND:
            return arith::AndIOp::create(b, l, old, val);
          case triton::RMWOp::OR:
            return arith::OrIOp::create(b, l, old, val);
          case triton::RMWOp::XOR:
            return arith::XOrIOp::create(b, l, old, val);
          case triton::RMWOp::ADD:
            return arith::AddIOp::create(b, l, old, val);
          case triton::RMWOp::FADD: {
            Value oldF = arith::BitcastOp::create(b, l, b.getF32Type(), old);
            Value sum = arith::AddFOp::create(b, l, oldF, val);
            return arith::BitcastOp::create(b, l, b.getI32Type(), sum);
          }
          case triton::RMWOp::MAX:
            return arith::MaxSIOp::create(b, l, old, val);
          case triton::RMWOp::MIN:
            return arith::MinSIOp::create(b, l, old, val);
          case triton::RMWOp::UMAX:
            return arith::MaxUIOp::create(b, l, old, val);
          case triton::RMWOp::UMIN:
            return arith::MinUIOp::create(b, l, old, val);
          case triton::RMWOp::XCHG:
            return isFloat ? arith::BitcastOp::create(b, l, b.getI32Type(),
                                                       val)
                                 .getResult()
                           : val;
          }
          llvm_unreachable("unhandled triton::RMWOp");
        });

    Value result =
        isFloat ? arith::BitcastOp::create(rewriter, loc,
                                           rewriter.getF32Type(), oldBits)
                      .getResult()
                : oldBits;
    rewriter.replaceOp(op, result);
    return success();
  }

  npu::PointerInfoAnalysis *pointerInfoAnalysis;
};

struct ConvertAtomicCASOp : public OpConversionPattern<triton::AtomicCASOp> {
  using OpConversionPattern<triton::AtomicCASOp>::OpConversionPattern;

  explicit ConvertAtomicCASOp(TypeConverter &typeConverter,
                              npu::PointerInfoAnalysis *pointerInfoAnalysis,
                              MLIRContext *context)
      : OpConversionPattern<triton::AtomicCASOp>(typeConverter, context),
        pointerInfoAnalysis(pointerInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getPtr().getType()))
      return rewriter.notifyMatchFailure(
          op, "tensor-shaped (per-lane scatter) atomic_cas is not "
              "supported; only scalar atomics on a uniform address are "
              "lowered on this backend");

    Type elemType = op.getVal().getType();
    bool isFloat = isa<Float32Type>(elemType);
    if (!(elemType.isInteger(32) || isFloat))
      return rewriter.notifyMatchFailure(
          op, "only 32-bit int/float atomic_cas is supported");

    Location loc = op.getLoc();
    Value cmp = adaptor.getCmp();
    Value val = adaptor.getVal();

    Value oldBits = emitAtomicOp(
        rewriter, op, pointerInfoAnalysis, adaptor.getPtr(), /*mask=*/Value(),
        [&](ConversionPatternRewriter &b, Location l, Value old) -> Value {
          Value cmpBits =
              isFloat
                  ? arith::BitcastOp::create(b, l, b.getI32Type(), cmp)
                        .getResult()
                  : cmp;
          Value valBits =
              isFloat
                  ? arith::BitcastOp::create(b, l, b.getI32Type(), val)
                        .getResult()
                  : val;
          Value eq = arith::CmpIOp::create(b, l, arith::CmpIPredicate::eq,
                                           old, cmpBits);
          return arith::SelectOp::create(b, l, eq, valBits, old);
        });

    Value result =
        isFloat ? arith::BitcastOp::create(rewriter, loc,
                                           rewriter.getF32Type(), oldBits)
                      .getResult()
                : oldBits;
    rewriter.replaceOp(op, result);
    return success();
  }

  npu::PointerInfoAnalysis *pointerInfoAnalysis;
};

} // namespace

void populateAtomicOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    npu::PointerInfoAnalysis *pointerInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ConvertAtomicRMWOp>(typeConverter, pointerInfoAnalysis,
                                   patterns.getContext());
  patterns.add<ConvertAtomicCASOp>(typeConverter, pointerInfoAnalysis,
                                   patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
