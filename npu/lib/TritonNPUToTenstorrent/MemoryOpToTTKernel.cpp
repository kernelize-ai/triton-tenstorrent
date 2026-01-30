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

static Value applyLinearLayout(ConversionPatternRewriter &rewriter,
                               Location loc, Value indexI32,
                               const std::vector<int32_t> &bases) {
  Value offset = arith::createConstantI32(loc, rewriter, 0);
  // Value indexI32 =
  // arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(), index);

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
    LDBG("Converting descriptor load op w/ cb type: " << cb << "\n");

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

    auto i32Ty = rewriter.getI32Type();
    SmallVector<Value, 4> tileSizeValues;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      tileSizeValues.push_back(arith::ConstantOp::create(
          rewriter, loc, i32Ty,
          IntegerAttr::get(i32Ty, static_cast<int32_t>(tileShape[i]))));
    }

    // tileCoord[i] = Global Offset in Tiles (e.g., offset 32 -> tile 1)
    SmallVector<Value, 4> tileCoord;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      tileCoord.push_back(
          arith::DivSIOp::create(rewriter, loc, offsets[i], tileSizeValues[i]));
    }

    // tilesPerDim[i] = Total Tensor Size in Tiles
    auto shape = desc.getShape();
    SmallVector<Value, 4> tilesPerDim;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      tilesPerDim.push_back(arith::CeilDivSIOp::create(rewriter, loc, shape[i],
                                                       tileSizeValues[i]));
    }

    // determine how many tiles we need to load by converting the shape to tiles
    const int32_t numCbTiles =
        cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    assert(numTiles == numCbTiles &&
           "number of tiles in layout must match number of tiles in CB");
    LDBG("Loading from CB of size " << numCbTiles << " tiles");
    Value numPages = arith::createConstantI32(loc, rewriter, numCbTiles);
    ttkernel::CBReserveBackOp::create(rewriter, loc, cb, numPages);

    Value l1BaseAddr = ttkernel::GetWritePtrOp::create(rewriter, loc, cb);

    StringAttr tileDimName = S("tile");

    // bit shifts for element to tile conversion
    int32_t rowShift = llvm::Log2_32(tileShape[0]);
    int32_t colShift = llvm::Log2_32(tileShape[1]);

    // extract basis vectors from inverted layout
    auto invertedInDimsNames = llvm::to_vector(invertedLayout.getInDimNames());

    int32_t blockTilesH = static_cast<int32_t>(blockShape[0] / tileShape[0]);
    int32_t blockTilesW = static_cast<int32_t>(blockShape[1] / tileShape[1]);

    // How many bits in the loop counters?
    int32_t rowBits = llvm::Log2_32_Ceil(blockTilesH);
    int32_t colBits = llvm::Log2_32_Ceil(blockTilesW);

    std::vector<int32_t> rowToL1Bases;
    std::vector<int32_t> colToL1Bases;

    // Populate Row Bases (Maps RowIV -> L1 Index)
    for (int k = 0; k < rowBits; ++k) {
      // Look up dim0 bit (k + 5)
      int32_t val = invertedLayout.getBasis(invertedInDimsNames[0],
                                            k + rowShift, tileDimName);
      rowToL1Bases.push_back(val);
    }

    // Populate Col Bases (Maps ColIV -> L1 Index)
    for (int k = 0; k < colBits; ++k) {
      // Look up dim1 bit (k + 5)
      int32_t val = invertedLayout.getBasis(invertedInDimsNames[1],
                                            k + colShift, tileDimName);
      colToL1Bases.push_back(val);
    }

    // DRAM ordered loop
    Value zero = arith::createConstantI32(loc, rewriter, 0);
    Value one = arith::createConstantI32(loc, rewriter, 1);
    Value loopLimitRow = arith::createConstantI32(loc, rewriter, blockTilesH);
    Value loopLimitCol = arith::createConstantI32(loc, rewriter, blockTilesW);

    auto rowLoop = scf::ForOp::create(rewriter, loc, zero, loopLimitRow, one);
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value rowIv = rowLoop.getInductionVar();

      auto colLoop = scf::ForOp::create(rewriter, loc, zero, loopLimitCol, one);
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value colIv = colLoop.getInductionVar();

        Value globalRow =
            arith::AddIOp::create(rewriter, loc, tileCoord[0], rowIv);
        Value globalCol =
            arith::AddIOp::create(rewriter, loc, tileCoord[1], colIv);

        Value globalStride = tilesPerDim[1];
        Value remoteTileIndex = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(rewriter, loc, globalRow, globalStride),
            globalCol);

        // -----------------------------------------------------------
        // B. L1 Index Calculation (Using INVERTED Bases)
        // -----------------------------------------------------------
        // "Scatters" the linear DRAM reads into the correct L1 slot

        Value l1PartRow = applyLinearLayout(rewriter, loc, rowIv, rowToL1Bases);
        Value l1PartCol = applyLinearLayout(rewriter, loc, colIv, colToL1Bases);
        Value l1TileIndex =
            arith::AddIOp::create(rewriter, loc, l1PartRow, l1PartCol);

        // C. Issue Read
        Value l1OffsetBytes =
            arith::MulIOp::create(rewriter, loc, l1TileIndex, pageSize);
        Value crtL1Address =
            arith::AddIOp::create(rewriter, loc, l1BaseAddr, l1OffsetBytes);

        Value const0 = arith::createConstantI32(loc, rewriter, 0);
        Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
            rewriter, loc, addrGen, remoteTileIndex, const0, Value());

        ttkernel::NocAsyncReadOp::create(rewriter, loc, nocAddr, crtL1Address,
                                         pageSize);
      }
    }
    rewriter.setInsertionPointAfter(rowLoop);

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
        ttkernel::PackTileOp::create(
            rewriter, loc, arith::createConstantI32(loc, rewriter, i), dst,
            arith::createConstantI32(loc, rewriter, i - destIndexOffset));
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

    // Note: unlike the tensor descriptor base case we normalize the shape into
    // 32x32 tiles here
    SmallVector<Value, 4> tileSizeValues;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      tileSizeValues.push_back(arith::createConstantI32(
          loc, rewriter, static_cast<int32_t>(tileShape[i])));
    }

    // tileCoord[i] = tileBaseOffset[i] / blockShape[i]
    SmallVector<Value, 4> tileCoord;
    tileCoord.reserve(blockShape.size());
    for (unsigned i = 0; i < blockShape.size(); ++i) {
      tileCoord.push_back(
          arith::DivSIOp::create(rewriter, loc, offsets[i], tileSizeValues[i]));
    }

    auto shape = desc.getShape();
    // tilesPerDim[i] = ceil(shape[i] / blockShape[i])
    SmallVector<Value, 4> tilesPerDim;
    for (unsigned i = 0; i < blockShape.size(); ++i) {
      tilesPerDim.push_back(arith::CeilDivSIOp::create(rewriter, loc, shape[i],
                                                       tileSizeValues[i]));
    }

    int32_t blockTilesH = static_cast<int32_t>(blockShape[0] / tileShape[0]);
    int32_t blockTilesW = static_cast<int32_t>(blockShape[1] / tileShape[1]);

    int32_t rowBits = llvm::Log2_32_Ceil(blockTilesH);
    int32_t colBits = llvm::Log2_32_Ceil(blockTilesW);
    int32_t rowShift = llvm::Log2_32(tileShape[0]);
    int32_t colShift = llvm::Log2_32(tileShape[1]);

    auto invertedInDimsNames = llvm::to_vector(invertedLayout.getInDimNames());
    StringAttr outTileDim = S("tile");

    std::vector<int32_t> rowToL1Bases;
    std::vector<int32_t> colToL1Bases;

    for (int k = 0; k < rowBits; ++k)
      rowToL1Bases.push_back(invertedLayout.getBasis(invertedInDimsNames[0],
                                                     k + rowShift, outTileDim));

    for (int k = 0; k < colBits; ++k)
      colToL1Bases.push_back(invertedLayout.getBasis(invertedInDimsNames[1],
                                                     k + colShift, outTileDim));
    // determine how many tiles we need to store by converting the shape to
    // tiles
    const int32_t numCbTiles =
        cast<ttkernel::CBType>(cb.getType()).getNumTiles();
    assert(numTiles == numCbTiles &&
           "number of tiles in layout must match number of tiles in CB");
    Value numPages = arith::createConstantI32(loc, rewriter, numCbTiles);

    Value l1Addr = ttkernel::GetReadPtrOp::create(rewriter, loc, cb);

    Value const0 = arith::createConstantI32(loc, rewriter, 0);

    Value zero = arith::createConstantI32(loc, rewriter, 0);
    Value one = arith::createConstantI32(loc, rewriter, 1);
    Value loopLimitRow = arith::createConstantI32(loc, rewriter, blockTilesH);
    Value loopLimitCol = arith::createConstantI32(loc, rewriter, blockTilesW);

    auto rowLoop = scf::ForOp::create(rewriter, loc, zero, loopLimitRow, one);
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value rowIv = rowLoop.getInductionVar();

      auto colLoop = scf::ForOp::create(rewriter, loc, zero, loopLimitCol, one);
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value colIv = colLoop.getInductionVar();

        Value globalRow =
            arith::AddIOp::create(rewriter, loc, tileCoord[0], rowIv);
        Value globalCol =
            arith::AddIOp::create(rewriter, loc, tileCoord[1], colIv);

        Value globalStride = tilesPerDim[1];

        Value remoteTileIndex = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(rewriter, loc, globalRow, globalStride),
            globalCol);

        Value l1PartRow = applyLinearLayout(rewriter, loc, rowIv, rowToL1Bases);
        Value l1PartCol = applyLinearLayout(rewriter, loc, colIv, colToL1Bases);
        Value l1TileIndex =
            arith::AddIOp::create(rewriter, loc, l1PartRow, l1PartCol);

        Value l1OffsetBytes =
            arith::MulIOp::create(rewriter, loc, l1TileIndex, pageSize);
        Value crtL1Address =
            arith::AddIOp::create(rewriter, loc, l1Addr, l1OffsetBytes);

        Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
            rewriter, loc, addrGen, remoteTileIndex, const0, Value());

        ttkernel::NocAsyncWriteOp::create(rewriter, loc, crtL1Address, nocAddr,
                                          pageSize);
      }
    }
    rewriter.setInsertionPointAfter(rowLoop);

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
