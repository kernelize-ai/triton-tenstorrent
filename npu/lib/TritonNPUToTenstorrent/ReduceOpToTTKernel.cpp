#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "Utility.h"

// Direct `tt.reduce` -> TTKernel lowering, mirroring how DotOpToTTKernel.cpp
// converts `tt.dot` straight to TTKernel ops rather than routing through the
// D2M dialect (see npu/lib/TritonNPUToTenstorrent/D2M/ReduceOpToD2M.cpp,
// which targets a different, currently-unused pipeline for this backend).
//
// Scope: rank-1, axis-0 (i.e. full) reduce over an int32 tensor, with a
// single Sum or Max combiner -- exactly what `ttkernel.sfpu_reduce` supports
// (see its tablegen doc: "Only Sum/Max reductions over Int32 tiles are
// supported today"). `sfpu_reduce` reduces a DST tile in place and only
// accepts Row or Col (not Scalar) as its dimension, so a full reduce is
// expressed as the documented decomposition: Col then Row.
//
// Triton's `tt.reduce` over a 1D tensor produces a genuine scalar SSA value
// (confirmed by inspection: `"tt.reduce"(...) : (tensor<1024xi32>) -> i32`),
// but sfpu_reduce's result stays inside a DST register -- not something a
// plain scalar consumer (a scalar store, an atomic) can use directly. Bridge
// the two by packing the reduced DST tile out to a dedicated scratch CB
// (mirroring how a normal tensor store already packs a DST tile to its
// output CB) and reading its first element back over L1, the same
// CastToL1PtrOp/LoadFromL1Op idiom AtomicOpToTTKernel.cpp uses to read its
// atomic-op return value. That scratch CB doesn't correspond to any kernel
// argument, so it's reserved as an extra compile-time CBPort arg, one per
// static reduce call site -- the same pattern AtomicOpToTTKernel.cpp uses to
// reserve its lock/scratch semaphores.

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

std::optional<ttkernel::ReduceType> classifyReduceOp(triton::ReduceOp op,
                                                      Type elemType) {
  if (!elemType.isInteger(32))
    return std::nullopt;
  Operation *combiner = op.getSingleCombiner();
  if (!combiner)
    return std::nullopt;
  if (isa<arith::AddIOp>(combiner))
    return ttkernel::ReduceType::Sum;
  if (isa<arith::MaxSIOp, arith::MaxUIOp>(combiner))
    return ttkernel::ReduceType::Max;
  return std::nullopt;
}

// A scratch CB just big enough to hold the one tile sfpu_reduce leaves its
// result in; only element [0] is ever meaningful.
ttkernel::CBType getScalarScratchCBType(MLIRContext *ctx, Type elemType) {
  auto tileType = ttcore::TileType::get(ctx, ttcore::TileType::getDefaultShape(),
                                        ttcore::elementTypeToDataType(elemType));
  MemRefType memrefType = MemRefType::get(
      {1}, tileType, MemRefLayoutAttrInterface{},
      ttcore::MemorySpaceAttr::get(ctx, ttcore::MemorySpace::DeviceL1));
  return ttkernel::CBType::get(memrefType);
}

struct ConvertReduceOp : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected single-operand, single-result reduce");

    auto srcType = dyn_cast<RankedTensorType>(op.getSrcs()[0].getType());
    if (!srcType || srcType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "only rank-1 reduce is lowered");
    if (op.getAxis() != 0)
      return rewriter.notifyMatchFailure(op, "only axis=0 reduce is lowered");

    auto reduceType = classifyReduceOp(op, srcType.getElementType());
    if (!reduceType)
      return rewriter.notifyMatchFailure(
          op, "unsupported reduce combiner/type; only int32 sum/max are "
              "lowered (sfpu_reduce's own restriction)");

    Value src = adaptor.getOperands()[0];
    auto srcCBType = dyn_cast<ttkernel::CBType>(src.getType());
    if (!srcCBType)
      return rewriter.notifyMatchFailure(
          op, "expected reduce operand to be CB-resident (e.g. straight off "
              "a load); DST-register-resident operands are not handled yet");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type elemType = srcType.getElementType();
    auto dataFormat = ttcore::elementTypeToDataType(elemType);

    int64_t destReg = lookupRegisterIndex(op->getResult(0));
    assert(destReg >= 0 && "expected register allocation for reduce result");
    Value destRegIdxVal = arith::createIndexConstant(loc, rewriter, destReg);

    // Reserve a dedicated scratch CB for this static reduce call site, the
    // same way AtomicOpToTTKernel.cpp reserves its lock/scratch semaphores:
    // one extra compile-time arg appended to the kernel's ArgSpec.
    auto parentFuncOp = op->getParentOfType<func::FuncOp>();
    assert(parentFuncOp && "expected reduce op inside a kernel func");
    auto argSpec = parentFuncOp->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    SmallVector<ttkernel::ArgAttr> ctArgs;
    if (argSpec)
      ctArgs = llvm::to_vector(argSpec.getCtArgs());
    int32_t scratchCbCtIdx = ctArgs.size();

    ttkernel::CBType scratchCBType = getScalarScratchCBType(ctx, elemType);
    Value scratchCb = ttkernel::GetCompileArgValOp::create(
        rewriter, loc, scratchCBType, scratchCbCtIdx);

    rewriter.modifyOpInPlace(parentFuncOp, [&]() {
      ttkernel::ArgSpecAttr::appendCompileTimeArg(
          parentFuncOp,
          rewriter.getAttr<ttkernel::ArgAttr>(ttkernel::ArgType::CBPort, 0));
    });

    // Bring the source tile into DST, then reduce it in place: Col then Row,
    // since sfpu_reduce itself only supports one axis at a time and its
    // tablegen doc says a full (Scalar) reduce must be decomposed this way.
    ttkernel::CopyTileInitOp::create(rewriter, loc, src);
    Value zeroIdx = arith::createIndexConstant(loc, rewriter, 0);
    ttkernel::CopyTileOp::create(rewriter, loc, src, zeroIdx, destRegIdxVal);

    ttkernel::SFPUReduceInitOp::create(rewriter, loc, *reduceType, dataFormat);
    ttkernel::SFPUReduceTileOp::create(rewriter, loc, destRegIdxVal,
                                       *reduceType, dataFormat,
                                       ttkernel::ReduceDim::Col);
    ttkernel::SFPUReduceTileOp::create(rewriter, loc, destRegIdxVal,
                                       *reduceType, dataFormat,
                                       ttkernel::ReduceDim::Row);

    // Pack the reduced tile out to the scratch CB, then read its first
    // (only meaningful) element back as a genuine scalar SSA value.
    Value onePage = arith::createConstantI32(loc, rewriter, 1);
    ttkernel::CBReserveBackOp::create(rewriter, loc, scratchCb, onePage);
    Value zeroI32 = arith::createConstantI32(loc, rewriter, 0);
    ttkernel::PackTileOp::create(rewriter, loc, destRegIdxVal, scratchCb,
                                 zeroI32, /*out_of_order=*/true);
    ttkernel::CBPushBackOp::create(rewriter, loc, scratchCb, onePage);

    ttkernel::CBWaitFrontOp::create(rewriter, loc, scratchCb, onePage);
    Value readPtr = ttkernel::GetReadPtrOp::create(rewriter, loc, scratchCb);
    Value l1Ptr = ttkernel::CastToL1PtrOp::create(
        rewriter, loc, ttkernel::L1AddrPtrType::get(ctx, 32), readPtr);
    Value zeroOffset = arith::createConstantI32(loc, rewriter, 0);
    Value scalarVal = ttkernel::LoadFromL1Op::create(
        rewriter, loc, rewriter.getIntegerType(32), l1Ptr, zeroOffset);
    ttkernel::CBPopFrontOp::create(rewriter, loc, scratchCb, onePage);

    rewriter.replaceOp(op, scalarVal);
    return success();
  }
};

} // namespace

void populateReduceOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertReduceOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
