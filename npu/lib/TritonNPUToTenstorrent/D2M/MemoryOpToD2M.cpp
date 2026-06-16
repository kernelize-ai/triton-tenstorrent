#include "PatternTritonNPUToD2M.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "../Utility.h"

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

    const int64_t rank = std::max(loadTensorType.getRank(), int64_t{2});
    assert(gridShardShape.size() == 2 * rank &&
           "expecting descriptor to have 2*rank dimensions for grid shape and "
           "shard shape");
    auto tileShape = llvm::to_vector(llvm::drop_begin(gridShardShape, rank));

    // TODO: get this from the type converter?
    auto cbLayout = ttcore::CBLayoutAttr::get(
        context, tileShape,
        ttcore::getElementSizeBytes(descType.getElementType()),
        /*buffers=*/2);
    auto allocType = MemRefType::get(
        tileShape, descType.getElementType(), cbLayout,
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));
    auto allocOp = memref::AllocOp::create(rewriter, loc, allocType);

    // step 2: process the indices
    auto loadResultType = cast<RankedTensorType>(op.getResult().getType());
    // convert result type to memref with tile shape
    // the result type does not use the grid shape (block shape + tile shape)
    auto resultType = MemRefType::get(
        tileShape, descType.getElementType(), MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));

    SmallVector<Value> indices;
    for (Value idx : op.getIndices())
      indices.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), idx));
    if (indices.size() == 1)
      indices.push_back(arith::createIndexConstant(loc, rewriter, 0));

    // step 3: create the remote load
    d2m::RemoteLoadOp::create(rewriter, loc, {}, allocOp.getResult(), descPtr,
                              indices);

    rewriter.replaceOp(op, allocOp.getResult());

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
    if (indices.size() == 1)
      indices.push_back(arith::createIndexConstant(loc, rewriter, 0));

    Value src = adaptor.getSrc()[0];

    // local buffer variant of remote store
    d2m::RemoteStoreOp::create(rewriter, loc, /*resultType=*/{}, descPtr,
                               indices, src);

    rewriter.eraseOp(op);
    return success();
  }
};

static Value traceToBasePtr(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<ttir::TTNNMetalLayoutCastOp>(def))
      return v;

    Value next =
        llvm::TypeSwitch<Operation *, Value>(def)
            .Case<triton::AddPtrOp>([](auto o) { return o.getPtr(); })
            .Case<triton::SplatOp>([](auto o) { return o.getSrc(); })
            .Case<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
                  triton::BitcastOp>([](auto o) { return o->getOperand(0); })
            .Case<UnrealizedConversionCastOp>(
                [](auto o) { return o.getInputs()[0]; })
            .Default([](Operation *) { return Value(); });

    if (!next)
      return nullptr;
    v = next;
  }
  // bail on block arguments
  return nullptr;
}
struct ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();

    Value ptr = adaptor.getPtr();

    // step 1: allocate L1 space for the load
    MemRefType memRef = cast<MemRefType>(ptr.getType());
    SmallVector<int64_t> tileShape = llvm::to_vector(memRef.getShape());
    if (tileShape.size() == 1)
      tileShape.push_back(1);

    auto allocType = cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto allocOp = memref::AllocOp::create(rewriter, loc, allocType);

    // step 2: create the load
    // TODO: this should be the same as the memref shape above, but I suspect we
    // will need grid shape on the memref shape above. If not, we can drop this
    // resultType and unify types
    auto resultType = MemRefType::get(
        tileShape, memRef.getElementType(), MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));

    auto conversionCastOp =
        dyn_cast_or_null<UnrealizedConversionCastOp>(ptr.getDefiningOp());
    assert(conversionCastOp && "expected tt.ptr lowering chain to end in "
                               "integer offset for load ptr argument");
    // TODO: index is in bytes, not tiles. need to divide by tile size
    // TODO: remote load always loads the entire tile - is that a problem here?
    Value index =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                   conversionCastOp.getInputs()[0]);

    Value basePtr = traceToBasePtr(op.getPtr());
    assert(basePtr && "expected load ptr chain to terminate at a layout cast");

    d2m::RemoteLoadOp::create(
        rewriter, loc, {}, allocOp.getResult(), basePtr,
        {index, arith::createIndexConstant(loc, rewriter, 0)});

    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct ConvertStoreOp : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();

    Value ptr = adaptor.getPtr();
    auto conversionCastOp =
        dyn_cast_or_null<UnrealizedConversionCastOp>(ptr.getDefiningOp());
    assert(conversionCastOp && "expected tt.ptr lowering chain to end in "
                               "integer offset for load ptr argument");
    // TODO: index is in bytes, not tiles. need to divide by tile size
    Value index =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                   conversionCastOp.getInputs()[0]);

    Value basePtr = traceToBasePtr(op.getPtr());
    assert(basePtr && "expected store ptr chain to terminate at a layout cast");

    Value src = adaptor.getValue();
    d2m::RemoteStoreOp::create(
        rewriter, loc, /*resultType=*/{}, basePtr,
        {index, arith::createIndexConstant(loc, rewriter, 0)}, src);
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

  patterns.add<ConvertLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertStoreOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
