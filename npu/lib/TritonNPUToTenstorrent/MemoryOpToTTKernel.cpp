#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/Support/Debug.h"

#include "PointerInfoAnalysis.h"
#include "Utility.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

#define S(v) StringAttr::get(context, (v))

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

struct ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  explicit ConvertLoadOp(TypeConverter &typeConverter,
                         npu::PointerInfoAnalysis *pointerInfoAnalysis,
                         MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context),
        pointerInfoAnalysis(pointerInfoAnalysis) {}

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

    auto ptrInfo = pointerInfoAnalysis->getInfo(op);
    assert(ptrInfo && "expected pointer info for load op");
    Value baseAddr = ptrInfo->basePtr;

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value trueVal = arith::createConstantI1(loc, rewriter, 1);
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // convert bytes offset to tile index
    Value offset = adaptor.getPtr();
    Value baseTileIndex =
        arith::DivUIOp::create(rewriter, loc, offset, pageSize);

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numTiles = cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    Value numPages = arith::createConstantI32(loc, rewriter, numTiles);
    ttkernel::CBReserveBackOp::create(rewriter, loc, cb, numPages);

    Value l1Addr = ttkernel::GetWritePtrOp::create(rewriter, loc, cb);

    scf::ForOp loadTileLoop = scf::ForOp::create(
        rewriter, loc, arith::createConstantI32(loc, rewriter, 0), numPages,
        arith::createConstantI32(loc, rewriter, 1),
        ValueRange{l1Addr, baseTileIndex});
    {
      rewriter.setInsertionPointToStart(loadTileLoop.getBody());
      Value crtL1Address = loadTileLoop.getRegionIterArgs()[0];
      Value crtTileIndex = loadTileLoop.getRegionIterArgs()[1];
      // TODO: should the offset be const0 here? the only examples we have are
      // TensorAccessor...
      Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
          rewriter, loc, addrGen, crtTileIndex, const0, Value());
      ttkernel::NocAsyncReadOp::create(rewriter, loc, nocAddr, crtL1Address,
                                       pageSize);
      Value nextL1Address =
          arith::AddIOp::create(rewriter, loc, crtL1Address, pageSize);
      Value nextTileIndex =
          arith::AddIOp::create(rewriter, loc, crtTileIndex,
                                arith::createConstantI32(loc, rewriter, 1));
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{nextL1Address, nextTileIndex});
    }

    rewriter.setInsertionPointAfter(loadTileLoop);
    ttkernel::NocAsyncReadBarrierOp::create(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }

  npu::PointerInfoAnalysis *pointerInfoAnalysis;
};

struct ConvertTensorDescLoadOp
    : public OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    if (!op->hasOneUse()) {
      LDBG("Load op has multiple uses, cannot convert\n");
      return failure();
    }
    Operation *user = *op->getUsers().begin();
    if (!isa<gpu::LocalStoreOp>(user)) {
      LDBG("Descriptor load op user is not a local store op: " << *user
                                                               << "\n");
      return failure();
    }

    Value cb =
        rewriter.getRemappedValue(cast<gpu::LocalStoreOp>(user).getDst());

    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto desc = TensorDescriptorUnpacked(descTy, adaptor.getDesc());

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value trueVal = arith::createConstantI1(loc, rewriter, 1);
    Value baseAddr = desc.generateBasePtr(rewriter, loc, blockShape);
    llvm::errs() << "base addr = " << baseAddr << "\n";
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // convert bytes offset to tile index
    auto offsets = op.getIndices();

    // TODO: this generates a nice ptr chain but unfortunately we need the
    // version without the addptr stuff... probably need to add that to the desc
    // class.
    Value offset =
        desc.generateBaseBlockOffset(rewriter, loc, blockShape, offsets);

    llvm::errs() << "offset = " << offset << "\n";
    Value baseTileIndex =
        arith::DivUIOp::create(rewriter, loc, offset, pageSize);

    auto i32Ty = rewriter.getI32Type();
    auto shape = desc.getShape();
    SmallVector<Value, 4> blockShapeValues;
    for (unsigned i = 0; i < blockShape.size(); ++i) {
      blockShapeValues.push_back(arith::ConstantOp::create(
          rewriter, loc, i32Ty,
          IntegerAttr::get(i32Ty, static_cast<int32_t>(blockShape[i]))));
    }
    // tilesPerDim[i] = ceil(shape[i] / blockShape[i])
    SmallVector<Value, 4> tilesPerDim;
    for (unsigned i = 0; i < blockShape.size(); ++i) {
      tilesPerDim.push_back(arith::CeilDivSIOp::create(rewriter, loc, shape[i],
                                                       blockShapeValues[i]));
    }

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numCbTiles =
        cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    LDBG("Loading from CB of size " << numCbTiles << " tiles");
    Value numPages = arith::createConstantI32(loc, rewriter, numCbTiles);
    ttkernel::CBReserveBackOp::create(rewriter, loc, cb, numPages);

    Value l1Addr = ttkernel::GetWritePtrOp::create(rewriter, loc, cb);

    auto loadResultType = cast<RankedTensorType>(op.getResult().getType());
    const int32_t elementSize =
        loadResultType.getElementType().getIntOrFloatBitWidth() / 8;
    if (auto dotOpEncoding = dyn_cast<npu::tt::TiledDotOperandEncodingAttr>(
            loadResultType.getEncoding())) {
      LDBG("Lowering load op with encoding " << dotOpEncoding << "\n");
      auto layout =
          gpu::toLinearLayout(loadResultType.getShape(), dotOpEncoding);
      layout = layout.sublayout({S("register"), S("tile")},
                                llvm::to_vector(layout.getOutDimNames()));
      LDBG("Register/Tile layout:\n" << layout << "\n");

      auto numTiles = layout.getInDimSize(S("tile"));
      assert(numTiles == numCbTiles &&
             "number of tiles in layout must match number of tiles in CB");
      LDBG("Generating " << numTiles << " tile loads");

      auto outDimNames = llvm::to_vector(layout.getOutDimNames());
      auto tiledParent =
          cast<npu::tt::TiledEncodingAttr>(dotOpEncoding.getParent());
      auto tileShape = tiledParent.getTileShape();
      auto fastChangeDim = outDimNames[1];
      unsigned numFastChangeDimTiles =
          layout.getOutDimSize(fastChangeDim) / tileShape[1];
      LDBG("Fast changing dim: " << fastChangeDim << " with "
                                 << numFastChangeDimTiles << " tiles");

      // auto registerIndexLayout = layout.flattenOuts();
      for (int32_t i = 0; i < numTiles; ++i) {
        auto crtIndex = layout.apply({{S("tile"), i}, {S("register"), 0}});
        assert(crtIndex.size() == 2);
        LLVM_DEBUG({
          DBGS() << "Tile " << i << " has start index: ";
          for (auto [dim, idx] : crtIndex) {
            DBGS() << dim.getValue() << ": " << idx << ", ";
          }
          DBGS() << "\n";
        });
        // linearize tensor index into tiled byte offset
        Value tileIndexDim0 = arith::createConstantI32(
            loc, rewriter, crtIndex[0].second / tileShape[0]);
        Value tileIndexDim1 = arith::createConstantI32(
            loc, rewriter, crtIndex[1].second / tileShape[1]);
        Value tileOffset = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(rewriter, loc, tileIndexDim0, tilesPerDim[1]),
            tileIndexDim1);
#if 1
        Value crtTileIndex = tileOffset;
#else
        // broken, we can't offset like this
        Value crtByteOffset =
            arith::AddIOp::create(rewriter, loc, offset, byteOffsetVal);
        Value crtTileIndex =
            arith::DivUIOp::create(rewriter, loc, crtByteOffset, pageSize);
#endif
        Value localTileIndex = arith::createConstantI32(loc, rewriter, i);
        Value localTileIndexOffset =
            arith::MulIOp::create(rewriter, loc, localTileIndex, pageSize);
        Value crtL1Address =
            arith::AddIOp::create(rewriter, loc, l1Addr, localTileIndexOffset);
        // TODO: should the offset be const0 here? the only examples we have are
        // TensorAccessor...
        Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
            rewriter, loc, addrGen, crtTileIndex, const0, Value());
        ttkernel::NocAsyncReadOp::create(rewriter, loc, nocAddr, crtL1Address,
                                         pageSize);
      }
    }
    ttkernel::NocAsyncReadBarrierOp::create(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertStoreOp : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  explicit ConvertStoreOp(TypeConverter &typeConverter,
                          npu::PointerInfoAnalysis *pointerInfoAnalysis,
                          MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context),
        pointerInfoAnalysis(pointerInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto cb = adaptor.getValue();
    LDBG("Store op value: " << cb << "\nwith type: " << cb.getType());
    assert(isa<ttkernel::CBType>(cb.getType()) && "expected cb type");

    auto ptrInfo = pointerInfoAnalysis->getInfo(op);
    assert(ptrInfo && "expected pointer info for load op");
    Value baseAddr = ptrInfo->basePtr;
    LDBG("Store op base address: " << baseAddr << "\n");

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value trueVal = arith::createConstantI1(loc, rewriter, 1);
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // convert bytes offset to tile index
    Value offset = adaptor.getPtr();
    Value baseTileIndex =
        arith::DivUIOp::create(rewriter, loc, offset, pageSize);

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    auto storeType = cast<RankedTensorType>(op.getValue().getType());
    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numTiles = cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    Value numPages = arith::createConstantI32(loc, rewriter, numTiles);

    Value l1Addr = ttkernel::GetReadPtrOp::create(rewriter, loc, cb);

    scf::ForOp storeTileLoop = scf::ForOp::create(
        rewriter, loc, arith::createConstantI32(loc, rewriter, 0), numPages,
        arith::createConstantI32(loc, rewriter, 1),
        ValueRange{l1Addr, baseTileIndex});
    {
      rewriter.setInsertionPointToStart(storeTileLoop.getBody());

      Value crtL1Address = storeTileLoop.getRegionIterArgs()[0];
      Value crtTileIndex = storeTileLoop.getRegionIterArgs()[1];

      // TODO: should the offset be const0 here? the only examples we have are
      // TensorAccessor..
      Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
          rewriter, loc, addrGen, crtTileIndex, const0, Value());

      ttkernel::NocAsyncWriteOp::create(rewriter, loc, crtL1Address, nocAddr,
                                        pageSize);

      Value nextL1Address =
          arith::AddIOp::create(rewriter, loc, crtL1Address, pageSize);
      Value nextTileIndex =
          arith::AddIOp::create(rewriter, loc, crtTileIndex,
                                arith::createConstantI32(loc, rewriter, 1));
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{nextL1Address, nextTileIndex});
    }
    rewriter.setInsertionPointAfter(storeTileLoop);

    ttkernel::NocAsyncWriteBarrierOp::create(rewriter, loc);
    ttkernel::CBPopFrontOp::create(rewriter, loc, cb, numPages);

    rewriter.eraseOp(op);
    return success();
  }

  npu::PointerInfoAnalysis *pointerInfoAnalysis;
};

struct ConvertLocalStoreOp : public OpConversionPattern<gpu::LocalStoreOp> {
  using OpConversionPattern<gpu::LocalStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dst = adaptor.getDst();
    auto dstCBType = cast<ttkernel::CBType>(dst.getType());
    Value numPages =
        arith::createConstantI32(loc, rewriter, dstCBType.getNumTiles());

    auto srcOp = op.getSrc().getDefiningOp();
    if (!isa<triton::LoadOp>(srcOp)) {
      // reserve back the cb for pack tile
      ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numPages);

      auto srcType = cast<RankedTensorType>(op.getSrc().getType());
      unsigned destIndexOffset = 0;
      if (auto registerEncodingAttr =
              dyn_cast<npu::tt::RegisterEncodingAttr>(srcType.getEncoding())) {
        destIndexOffset = registerEncodingAttr.getIndex();
      }

      // commit and wait before packing tiles
      ttkernel::TileRegsCommitOp::create(rewriter, loc);
      ttkernel::TileRegsWaitOp::create(rewriter, loc);

      for (unsigned i = destIndexOffset;
           i < destIndexOffset + dstCBType.getNumTiles(); ++i) {
        // assume the output CB is always 0-indexed
        ttkernel::PackTileOp::create(
            rewriter, loc, arith::createConstantI32(loc, rewriter, i), dst,
            arith::createConstantI32(loc, rewriter, i - destIndexOffset));
      }

      ttkernel::TileRegsReleaseOp::create(rewriter, loc);

      ttkernel::CBPushBackOp::create(rewriter, loc, dst, numPages);
      rewriter.eraseOp(op);
      return success();
    }

    // store from a load op just signals that the data is ready in the cb
    ttkernel::CBPushBackOp::create(rewriter, loc, dst, numPages);
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
    auto srcType = cast<ttkernel::CBType>(src.getType());
    Value numPages =
        arith::createConstantI32(loc, rewriter, srcType.getNumTiles());

    LDBG("Converted load src type = " << src.getType() << "\n");
    assert(isa<ttkernel::CBType>(src.getType()) &&
           "expected memref type for type converted load src");

    const bool hasDotUser = llvm::any_of(op->getUsers(), [](Operation *user) {
      return isa<triton::DotOp>(user);
    });

    // 1. wait front on the cb to know data is ready
    if (!hasDotUser) {
      // Dot ops read directly from cbs and handle their own waits
      // TODO: consider verifying that all users are dot ops?
      // TODO: could we sink this into last block (3. copy data...)?
      auto waitFrontOp =
          ttkernel::CBWaitFrontOp::create(rewriter, loc, src, numPages);
    }

    // 2. for ops that operate directly on data in cbs, just replace the load op
    // with its ptr
    assert(op->hasOneUse() &&
           "expected local load with store user to have one use");
    if (isStoreLike(*op->getUsers().begin())) {
      rewriter.replaceOp(op, src);
      return success();
    }

    auto dst = op.getResult();
    auto dstType = cast<RankedTensorType>(dst.getType());
    if (isa<npu::tt::TiledDotOperandEncodingAttr>(dstType.getEncoding()) ||
        isa<gpu::DotOperandEncodingAttr>(dstType.getEncoding())) {
      // Dot ops read directly from cbs, so skip the copy tile and just replace
      // the load with its cb src
      rewriter.replaceOp(op, src);
      return success();
    }

    // 3. copy data from the cb into DST register
    ttkernel::CopyTileInitOp::create(rewriter, loc, src);
    Value c0 = arith::createIndexConstant(loc, rewriter, 0);
    npu::tt::RegisterEncodingAttr loadEncoding =
        cast<npu::tt::RegisterEncodingAttr>(dstType.getEncoding());
    Value destRegisterIndex =
        arith::createIndexConstant(loc, rewriter, loadEncoding.getIndex());
    ttkernel::CopyTileOp::create(rewriter, loc, src, c0, destRegisterIndex);
    ttkernel::CBPopFrontOp::create(rewriter, loc, src, numPages);
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

void populateMemoryOpConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    npu::PointerInfoAnalysis *pointerInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ConvertLoadOp>(typeConverter, pointerInfoAnalysis,
                              patterns.getContext());
  patterns.add<ConvertTensorDescLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertStoreOp>(typeConverter, pointerInfoAnalysis,
                               patterns.getContext());
  patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
