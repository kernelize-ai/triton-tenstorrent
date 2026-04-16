#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

#define S(v) StringAttr::get(context, (v))

struct ConvertTensorDescLoadOp
    : public OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();

    auto desc = adaptor.getDesc();

    assert(desc.size() >= 1 && "expected at least one value in the descriptor");
    auto descPtr = desc[0];

    // step 1: allocate L1 space for the load
    MemRefType descType = cast<MemRefType>(descPtr.getType());
    auto gridShardShape = descType.getShape();

    auto loadTensorType = cast<RankedTensorType>(op.getType());
    assert(gridShardShape.size() == 2 * loadTensorType.getRank() &&
           "expecting descriptor to have 2*rank dimensions for grid shape and "
           "shard shape");
    auto tileShape = llvm::to_vector(
        llvm::drop_begin(gridShardShape, loadTensorType.getRank()));

    auto cbLayout = ttcore::CBLayoutAttr::get(
        context, tileShape,
        ttcore::getElementSizeBytes(descType.getElementType()),
        /*buffers=*/tileShape.size());
    auto allocType = MemRefType::get(
        tileShape, descType.getElementType(), cbLayout,
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));
    auto allocOp = memref::AllocOp::create(rewriter, loc, allocType);

    // step 2: process the indices
    auto loadResultType = cast<RankedTensorType>(op.getResult().getType());
    // convert result type to memref with block shape
    // TODO: this is just the same as the alloc type?
    auto resultType = MemRefType::get(
        tileShape, descType.getElementType(), MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));

    SmallVector<Value> indices;
    for (Value idx : op.getIndices())
      indices.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), idx));

    // step 3: create the remote load
    auto remoteLoad = d2m::RemoteLoadOp::create(
        rewriter, loc, resultType, allocOp.getResult(), descPtr, indices);

    rewriter.replaceOp(op, remoteLoad);

    return success();
  }
};

struct ConvertTensorDescStoreOp
    : public OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();

    auto desc = adaptor.getDesc();
    assert(desc.size() >= 1 && "expected at least one value in the descriptor");
    auto descPtr = desc[0];

    SmallVector<Value> indices;
    for (Value idx : op.getIndices())
      indices.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), idx));

    Value src = adaptor.getSrc()[0];

    // local buffer variant of remote store
    auto remoteStore = d2m::RemoteStoreOp::create(
        rewriter, loc, /*resultType=*/descPtr.getType(), descPtr, indices, src);

    // remote store produces a result where descriptor store does not, so we
    // erase the descriptor store instead of replacing it
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMemoryOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertTensorDescLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertTensorDescStoreOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
