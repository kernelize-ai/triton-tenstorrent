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

// TODO: move to utilites, we want to keep this function
static Value applyLinearLayout(ConversionPatternRewriter &rewriter,
                               Location loc, Value indexI32,
                               const std::vector<int32_t> &bases) {
  Value offset = arith::createConstantI32(loc, rewriter, 0);

  for (size_t bit = 0; bit < bases.size(); ++bit) {
    if (bases[bit] == 0)
      continue; // Optimization: Skip zero bases
    // 1. Extract bit 'k' from index: (index >> k) & 1
    Value shiftAmount = arith::createConstantI32(loc, rewriter, bit);
    Value shifted =
        arith::ShRUIOp::create(rewriter, loc, indexI32, shiftAmount);
    Value one = arith::createConstantI32(loc, rewriter, 1);
    Value bitVal = arith::AndIOp::create(rewriter, loc, shifted, one);

    // 2. Multiply by basis: bitVal * basis[k]
    //    (Since bitVal is 0 or 1, this is effectively a selection)
    Value basisVal = arith::createConstantI32(loc, rewriter, bases[bit]);
    Value term = arith::MulIOp::create(rewriter, loc, bitVal, basisVal);

    // 3. Accumulate
    offset = arith::AddIOp::create(rewriter, loc, offset, term);
  }
  return offset;
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
    LDBG("Converting load op w/ cb type: " << cb << "\n");

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
    auto descOp = op.getDesc().getDefiningOp();
    ValueRange descValues =
        isa<UnrealizedConversionCastOp>(descOp)
            ? cast<UnrealizedConversionCastOp>(descOp).getInputs()
            : adaptor.getDesc();
    auto desc = TensorDescriptorUnpacked(descTy, descValues);

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value trueVal = arith::createConstantI1(loc, rewriter, 1);
    Value baseAddr = desc.getPtr();
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

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

    auto offsets = op.getIndices();

    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numCbTiles =
        cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    assert(numTiles == numCbTiles &&
           "number of tiles in layout must match number of tiles in CB");
    LDBG("Descriptor load cb type: " << cb.getType());
    const unsigned cbPageSize = 2048; // TODO: read from cb type
    LDBG("Loading from CB of size " << numCbTiles << " tiles");
    Value numPages = arith::createConstantI32(loc, rewriter, numCbTiles);
    ttkernel::CBReserveBackOp::create(rewriter, loc, cb, numPages);

    Value l1BaseAddr = ttkernel::GetWritePtrOp::create(rewriter, loc, cb);

    StringAttr tileDimName = S("tile");

    LDBG("Tile load loop - block shape: (" << blockShape[0] << ", "
                                           << blockShape[1] << ")\n");
    LDBG("Tile load loop - tile shape: (" << tileShape[0] << ", "
                                          << tileShape[1] << ")\n");

    int32_t blockTilesH = static_cast<int32_t>(blockShape[0] / tileShape[0]);
    int32_t blockTilesW = static_cast<int32_t>(blockShape[1] / tileShape[1]);
    LDBG("Tile load loop - block tiles: (" << blockTilesH << ", " << blockTilesW
                                           << ")\n");

    auto shape = desc.getShape();
    Value strideH = arith::CeilDivSIOp::create(
        rewriter, loc, shape[1],
        arith::createConstantI32(loc, rewriter, tileShape[1]));

    // global base coordinates
    Value baseTileH = arith::DivSIOp::create(
        rewriter, loc, offsets[0],
        arith::createConstantI32(loc, rewriter, tileShape[0]));
    Value baseTileW = arith::DivSIOp::create(
        rewriter, loc, offsets[1],
        arith::createConstantI32(loc, rewriter, tileShape[1]));

    // Base Remote Index = (StartRow * Stride) + StartCol
    Value baseRemoteIdx = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, baseTileH, strideH),
        baseTileW);

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    for (unsigned i = 0; i < blockTilesH; i++) {
      for (unsigned j = 0; j < blockTilesW; j++) {
        // compute the L1 index for this row and column
        int32_t elementOffsetH = i * tileShape[0];
        int32_t elementOffsetW = j * tileShape[1];
        LDBG("Element Offset for block tile (" << i << ", " << j << "): ("
                                               << elementOffsetH << ", "
                                               << elementOffsetW << ")\n");
        auto result = invertedLayout.apply({
            {S("dim0"), elementOffsetH},
            {S("dim1"), elementOffsetW},
        });
        assert(result.size() == 2 &&
               "expected inverted layout to have two output dimensions");
        assert(result[1].first == tileDimName && "expected tile dimension");
        LDBG("Computed L1 index for block tile ("
             << i << ", " << j << "): " << result[1].second << "\n");
        // NOTE: Computing the page size from the CB type here, so it better
        // match!
        Value l1TileOffsetBytes = arith::createConstantI32(
            loc, rewriter, result[1].second * cbPageSize);
        Value crtL1Address =
            arith::AddIOp::create(rewriter, loc, l1BaseAddr, l1TileOffsetBytes);

        // compute the global tile index for this row and column
        Value crtIndex = baseRemoteIdx;
        if (j > 0)
          crtIndex =
              arith::AddIOp::create(rewriter, loc, crtIndex,
                                    arith::createConstantI32(loc, rewriter, j));
        if (i > 0) {
          Value rowOffset = arith::MulIOp::create(
              rewriter, loc, arith::createConstantI32(loc, rewriter, i),
              strideH);
          crtIndex = arith::AddIOp::create(rewriter, loc, crtIndex, rowOffset);
        }

        // issue the read
        Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
            rewriter, loc, addrGen, crtIndex, const0, Value());
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

    auto localLoadOp = op.getValue().getDefiningOp();
    if (!isa<gpu::LocalLoadOp>(localLoadOp)) {
      assert(false && "expected store from a local load op");
      return failure();
    }
    auto loadOp = cast<gpu::LocalLoadOp>(localLoadOp);
    auto cb = rewriter.getRemappedValue(loadOp.getSrc());
    LDBG("Store op cb: " << cb << "\nwith type: " << cb.getType());
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

    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numTiles = cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    // const int32_t numTiles = getNumTiles(cb.getType());
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
    if (!isLoadLike(srcOp)) {
      // reserve back the cb for pack tile
      ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numPages);

      auto destIndexOffset = lookupRegisterIndex(op.getSrc());

      for (unsigned i = destIndexOffset;
           i < destIndexOffset + dstCBType.getNumTiles(); ++i) {
        // assume the output CB is always 0-indexed
        Value destRegisterIndex = arith::createConstantI32(loc, rewriter, i);
        Value cbOutputBufferIndex =
            arith::createConstantI32(loc, rewriter, i - destIndexOffset);
        ttkernel::PackTileOp::create(rewriter, loc, destRegisterIndex, dst,
                                     cbOutputBufferIndex,
                                     /*out_of_order=*/true);
      }

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

    auto dst = op.getResult();
    auto src = adaptor.getSrc();
    assert(isa<ttkernel::CBType>(src.getType()) &&
           "expected memref type for type converted load src");
    auto srcType = cast<ttkernel::CBType>(src.getType());
    // Q: can this be replaced with
    // tt::TritonTenstorrentDialect::getNumTiles(src.getType())?
    Value numPages =
        arith::createConstantI32(loc, rewriter, srcType.getNumTiles());

    LDBG("Converted load src type = " << src.getType() << "\n");

    int64_t registerIndex = lookupRegisterIndex(dst);
    if (registerIndex == -1) {
      // Ops that read directly from cbs, so skip the copy tile and just replace
      // the load with its cb src
      if (isStoreLike(*op->getUsers().begin())) {
        assert(op->hasOneUse() &&
               "expected local load with store user to have one use");
        ttkernel::CBWaitFrontOp::create(rewriter, loc, src, numPages);
      } else {
        // Dot ops handle their own waits
        const bool hasDotUser =
            llvm::all_of(op->getUsers(), [](Operation *user) {
              return isa<triton::DotOp>(user);
            });
        assert(hasDotUser && "expected local load with only dot users");

        auto dstType = cast<RankedTensorType>(dst.getType());
        assert(
            isa<npu::tt::TiledDotOperandEncodingAttr>(dstType.getEncoding()) ||
            isa<gpu::DotOperandEncodingAttr>(dstType.getEncoding()));
      }

      rewriter.replaceOp(op, src);
      return success();
    }
    // Ops that read from registers, so wait for the data to be ready
    ttkernel::CBWaitFrontOp::create(rewriter, loc, src, numPages);

    // Copy data from the cb into DST register
    ttkernel::CopyTileInitOp::create(rewriter, loc, src);

    for (unsigned i = 0; i < srcType.getNumTiles(); ++i) {
      Value cBindex = arith::createIndexConstant(loc, rewriter, i);
      Value regConst =
          arith::createIndexConstant(loc, rewriter, registerIndex + i);
      ttkernel::CopyTileOp::create(rewriter, loc, src, cBindex, regConst);
    }
    ttkernel::CBPopFrontOp::create(rewriter, loc, src, numPages);
    rewriter.replaceOp(op, src);
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

    auto srcValue = op.getSrc();
    auto srcLocalLoadOp = dyn_cast<gpu::LocalLoadOp>(srcValue.getDefiningOp());
    assert(srcLocalLoadOp &&
           "expected descriptor store op to store from a local load op");

    Value cb = rewriter.getRemappedValue(srcLocalLoadOp.getSrc());
    LDBG("Descriptor store op value: " << cb
                                       << "\nwith type: " << cb.getType());
    assert(isa<ttkernel::CBType>(cb.getType()) && "expected cb type");

    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto descOp = op.getDesc().getDefiningOp();
    ValueRange descValues =
        isa<UnrealizedConversionCastOp>(descOp)
            ? cast<UnrealizedConversionCastOp>(descOp).getInputs()
            : adaptor.getDesc();
    auto desc = TensorDescriptorUnpacked(descTy, descValues);

    // compute noc address
    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, cb);

    Value trueVal = arith::createConstantI1(loc, rewriter, true);
    Value baseAddr = desc.getPtr();
    Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // only support tiled encoding for now
    auto srcType = cast<RankedTensorType>(srcValue.getType());
    auto tiledEncoding =
        cast<npu::tt::TiledEncodingAttr>(srcType.getEncoding());
    auto layout = gpu::toLinearLayout(srcType.getShape(), tiledEncoding);
    layout = layout.sublayout({S("register"), S("tile")},
                              llvm::to_vector(layout.getOutDimNames()));
    LDBG("Register/Tile layout:\n" << layout << "\n");
    auto invertedLayout = layout.invert();
    LDBG("Inverted layout:\n" << invertedLayout << "\n");

    auto numTiles = layout.getInDimSize(S("tile"));
    LDBG("Generating " << numTiles << " tile stores");

    auto tileShape = tiledEncoding.getTileShape();
    auto offsets = op.getIndices();

    // determine how many tiles we need to store by converting the shape to
    // tiles
    const int32_t numCbTiles =
        cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    assert(numTiles == numCbTiles &&
           "number of tiles in layout must match number of tiles in CB");
    const unsigned cbPageSize = 2048; // TODO: read from cb type

    Value numPages = arith::createConstantI32(loc, rewriter, numCbTiles);

    Value l1BaseAddr = ttkernel::GetReadPtrOp::create(rewriter, loc, cb);
    StringAttr tileDimName = S("tile");

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    LDBG("Tile load loop - block shape: (" << blockShape[0] << ", "
                                           << blockShape[1] << ")\n");
    LDBG("Tile load loop - tile shape: (" << tileShape[0] << ", "
                                          << tileShape[1] << ")\n");

    int32_t blockTilesH = static_cast<int32_t>(blockShape[0] / tileShape[0]);
    int32_t blockTilesW = static_cast<int32_t>(blockShape[1] / tileShape[1]);
    LDBG("Tile load loop - block tiles: (" << blockTilesH << ", " << blockTilesW
                                           << ")\n");

    auto shape = desc.getShape();
    Value strideH = arith::CeilDivSIOp::create(
        rewriter, loc, shape[1],
        arith::createConstantI32(loc, rewriter, tileShape[1]));

    // global base coordinates
    Value baseTileH = arith::DivSIOp::create(
        rewriter, loc, offsets[0],
        arith::createConstantI32(loc, rewriter, tileShape[0]));
    Value baseTileW = arith::DivSIOp::create(
        rewriter, loc, offsets[1],
        arith::createConstantI32(loc, rewriter, tileShape[1]));

    // Base Remote Index = (StartRow * Stride) + StartCol
    Value baseRemoteIdx = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, baseTileH, strideH),
        baseTileW);

    for (unsigned i = 0; i < blockTilesH; i++) {
      for (unsigned j = 0; j < blockTilesW; j++) {
        // compute the L1 index for this row and column
        int32_t elementOffsetH = i * tileShape[0];
        int32_t elementOffsetW = j * tileShape[1];
        LDBG("Element Offset for block tile (" << i << ", " << j << "): ("
                                               << elementOffsetH << ", "
                                               << elementOffsetW << ")\n");
        auto result = invertedLayout.apply({
            {S("dim0"), elementOffsetH},
            {S("dim1"), elementOffsetW},
        });
        assert(result.size() == 2 &&
               "expected inverted layout to have two output dimensions");
        assert(result[1].first == tileDimName && "expected tile dimension");
        LDBG("Computed L1 index for block tile ("
             << i << ", " << j << "): " << result[1].second << "\n");
        // NOTE: Computing the page size from the CB type here, so it better
        // match!
        Value l1TileOffsetBytes = arith::createConstantI32(
            loc, rewriter, result[1].second * cbPageSize);
        Value crtL1Address =
            arith::AddIOp::create(rewriter, loc, l1BaseAddr, l1TileOffsetBytes);

        // compute the global tile index for this row and column
        Value crtIndex = baseRemoteIdx;
        if (j > 0)
          crtIndex =
              arith::AddIOp::create(rewriter, loc, crtIndex,
                                    arith::createConstantI32(loc, rewriter, j));
        if (i > 0) {
          Value rowOffset = arith::MulIOp::create(
              rewriter, loc, arith::createConstantI32(loc, rewriter, i),
              strideH);
          crtIndex = arith::AddIOp::create(rewriter, loc, crtIndex, rowOffset);
        }

        // issue the write
        Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
            rewriter, loc, addrGen, crtIndex, const0, Value());

        ttkernel::NocAsyncWriteOp::create(rewriter, loc, crtL1Address, nocAddr,
                                          pageSize);
      }
    }

    ttkernel::NocAsyncWriteBarrierOp::create(rewriter, loc);
    ttkernel::CBPopFrontOp::create(rewriter, loc, cb, numPages);

    rewriter.eraseOp(op);
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
  patterns.add<ConvertTensorDescStoreOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalStoreOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalLoadOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertLocalAllocOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
