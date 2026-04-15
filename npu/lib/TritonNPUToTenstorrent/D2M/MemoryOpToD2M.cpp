#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

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
    llvm::errs() << "desc = " << desc << "\n";

    assert(desc.size() >= 1 && "expected at least one value in the descriptor");
    auto descPtr = desc[0];

    // step 1: allocate L1 space for the load
    MemRefType descType = cast<MemRefType>(descPtr.getType());
    auto blockShape = descType.getShape();
    auto allocType = MemRefType::get(
        blockShape, descType.getElementType(), MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));
    auto allocOp = memref::AllocOp::create(rewriter, loc, allocType);

    // step 2: process the indices
#if 1
    auto loadResultType = cast<RankedTensorType>(op.getResult().getType());
    // convert result type to memref with block shape
    // TODO: this is just the same as the alloc type?
    auto resultType = MemRefType::get(
        blockShape, descType.getElementType(), MemRefLayoutAttrInterface{},
        ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceL1));

    auto indices = op.getIndices();
#else
    auto loadResultType = cast<RankedTensorType>(op.getResult().getType());
    LDBG("Lowering load op with encoding " << loadResultType.getEncoding()
                                           << "\n");
    auto layout = gpu::toLinearLayout(loadResultType.getShape(),
                                      loadResultType.getEncoding());
    layout = layout.sublayout({S("register"), S("tile")},
                              llvm::to_vector(layout.getOutDimNames()));
    LDBG("Register/Tile layout:\n" << layout << "\n");
    auto invertedLayout = layout.invert();
    LDBG("Inverted layout:\n" << invertedLayout << "\n");

    auto numTiles = layout.getInDimSize(S("tile"));
    LDBG("Generating " << numTiles << " tile loads");

    auto dotOpEncoding = dyn_cast<npu::tt::TiledDotOperandEncodingAttr>(
        loadResultType.getEncoding());
    auto tiledParent =
        dotOpEncoding
            ? cast<npu::tt::TiledEncodingAttr>(dotOpEncoding.getParent())
            : cast<npu::tt::TiledEncodingAttr>(loadResultType.getEncoding());
    auto tileShape = tiledParent.getTileShape();

    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();

    LDBG("Tile load loop - block shape: (" << blockShape[0] << ", "
                                           << blockShape[1] << ")\n");
    LDBG("Tile load loop - tile shape: (" << tileShape[0] << ", "
                                          << tileShape[1] << ")\n");

    int32_t blockTilesH = static_cast<int32_t>(blockShape[0] / tileShape[0]);
    int32_t blockTilesW = static_cast<int32_t>(blockShape[1] / tileShape[1]);
    LDBG("Tile load loop - block tiles: (" << blockTilesH << ", " << blockTilesW
                                           << ")\n");

    SmallVector<Value> indices;
    for (unsigned tile = 0; tile < numTiles; tile++) {
      auto coords = layout.apply(
          {{S("tile"), static_cast<int32_t>(tile)}, {S("register"), 0}});
      assert(coords.size() == 2 &&
             "expected layout to produce two coordinates");
      LDBG("tile " << tile << " cmputed global x,y coordinate "
                   << coords[0].second << ", " << coords[1].second << "\n");
      // TODO: convert global x,y coordinate to global tile coordinate
    }
#endif

    // step 2: create the remote load?
    auto remoteLoad = d2m::RemoteLoadOp::create(
        rewriter, loc, resultType, allocOp.getResult(), descPtr, indices);

    op.replaceAllUsesWith(remoteLoad.getResult());

    return success();
  }
};

} // namespace

void populateMemoryOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertTensorDescLoadOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
