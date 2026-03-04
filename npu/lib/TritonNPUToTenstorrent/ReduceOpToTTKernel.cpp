#include "PatternTritonNPUToTenstorrent.h"

#include "Utility.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ReduceOpConversion : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Operation *combiner = op.getSingleCombiner();
    if (!combiner) {
      return emitError(loc) << "ReduceOp must have a single combiner";
    }

    // ReduceTile only supports sum and max
    ttkernel::ReduceType reduceType;
    if (auto maxOp = dyn_cast<arith::MaxNumFOp>(combiner))
      reduceType = ttkernel::ReduceType::Max;
    else if (auto addOp = dyn_cast<arith::AddFOp>(combiner))
      reduceType = ttkernel::ReduceType::Sum;
    else
      return emitError(loc) << "Unsupported combiner for ReduceOp";

    if (op.getAxis() != 0)
      return emitError(loc) << "Only 1D reductions supported, currently";
    auto resultValues = op.getResult();
    if (resultValues.size() != 1)
      return emitError(loc) << "ReduceOp must have a single output";
    auto result = resultValues.front();
    if (isa<RankedTensorType>(result.getType()))
      return emitError(loc) << "ReduceOp result must be a scalar";

    ttkernel::ReduceDim reduceDim = ttkernel::ReduceDim::Scalar;

    auto cbs = adaptor.getSrcs();
    if (cbs.size() != 1)
      return emitError(loc) << "ReduceOp must have a single input";
    auto localLoadInput =
        dyn_cast<gpu::LocalLoadOp>(cbs.front().getDefiningOp());
    if (!localLoadInput)
      return emitError(loc) << "ReduceOp input must be from a circular buffer";

    // TODO: this isn't quite right, we need to get the local alloc _through_
    // the local load
    Value localAlloc = localLoadInput.getSrc();
    Value cbValue = rewriter.getRemappedValue(localAlloc);
    llvm::errs() << "cbValue: " << cbValue << "\n";
    auto cbMemRefType =
        cast<ttkernel::CBType>(typeConverter->convertType(cbValue.getType()));
    llvm::errs() << "cb mem ref type: " << cbMemRefType << "\n";

    // TODO: get the scaling factors CB from somewhere, or create it here and
    // update the arg spec.
    auto scalingCb = ttkernel::GetCompileArgValOp::create(
        rewriter, loc, cbMemRefType, rewriter.getI32IntegerAttr(2));
    // TODO: this should come from a local alloc added by previous stages, and
    // we should check that the result of reduce op feeds into local store
    auto outputCb = ttkernel::GetCompileArgValOp::create(
        rewriter, loc, cbMemRefType, rewriter.getI32IntegerAttr(3));

    const bool isFp32Reduction = true; // TODO: get from resultType
    UnitAttr fullFp32 = isFp32Reduction ? rewriter.getUnitAttr() : nullptr;
    ttkernel::ReduceInitOp::create(rewriter, loc, cbValue, scalingCb, outputCb,
                                   reduceType, reduceDim, isFp32Reduction);

    Value c0 = arith::createIndexConstant(loc, rewriter, 0);
    // TODO: support tiled layout here
    ttkernel::ReduceTileOp::create(rewriter, loc, cbValue, scalingCb, c0, c0,
                                   c0, reduceType, reduceDim, isFp32Reduction);
    ttkernel::ReduceUninitOp::create(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateReduceOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
