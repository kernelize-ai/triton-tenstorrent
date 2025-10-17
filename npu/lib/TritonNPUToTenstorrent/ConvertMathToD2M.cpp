#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTMATHTOD2M
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

using namespace tt;

namespace {

struct ConvertAddOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting AddOp to D2M: " << addOp << "\n";
    auto loc = addOp.getLoc();

    auto tensorTy = cast<RankedTensorType>(addOp.getType());

    auto typeConverter = getTypeConverter();
    auto retType =
        cast<RankedTensorType>(typeConverter->convertType(addOp.getType()));

    SmallVector<Value> inputs = {adaptor.getLhs(), adaptor.getRhs()};

    for (auto v : inputs)
      llvm::errs() << "input: " << v << " of type " << v.getType() << "\n";

    llvm::errs() << "retType: " << retType << "\n";

    auto getTiledType = [&](RankedTensorType tensorType) -> RankedTensorType {
      Type elementType = tensorType.getElementType();

      constexpr std::array<int64_t, 2> defaultShape =
          tt::ttcore::TileType::getDefaultShape();
      SmallVector<int64_t> tileShape{defaultShape[0], defaultShape[1]};
      Type tiledElementType = tt::ttcore::TileType::get(elementType, tileShape);

      llvm::errs() << "tileShape = " << tileShape[0] << " x " << tileShape[1]
                   << "\n";
      llvm::errs() << "tiledElementType = " << tiledElementType << "\n";

      SmallVector<int64_t> logicalShape =
          llvm::to_vector(tensorType.getShape());
      tt::ttcore::MemorySpace memSpace = tt::ttcore::MemorySpace::DeviceL1;

      // create metal layout
      tt::ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, targetSquareGridShape,
          ttcore::OOBVal::Undef, memSpace, ttcore::TensorMemoryLayout::Sharded);

      // Get raw, unsharded physical shape.
      SmallVector<int64_t> unshardedShape = layout.getPhysicalShape(tileShape);
      llvm::errs() << "unshardedShape = ";
      for (auto s : unshardedShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      // Calculate optimal grid for given physical shape.
      llvm::SmallVector<int64_t> optimalGrid =
          computeOptimalGrid(unshardedShape);
      llvm::errs() << "optimalGrid = ";
      for (auto g : optimalGrid)
        llvm::errs() << g << " ";
      llvm::errs() << "\n";

      // Get optimal sharded, on-device shape.
      llvm::SmallVector<int64_t> shardedShape =
          layout.getDeviceShape(optimalGrid, tileShape);
      llvm::errs() << "shardedShape = ";
      for (auto s : shardedShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      return RankedTensorType::get(shardedShape, tiledElementType, layout);
    };

    // Tilize the LHS and RHS inputs using d2m.to_layout
    auto getTiledInput = [&](Value v) -> Value {
      auto tensorType = cast<RankedTensorType>(v.getType());
#if 1
      llvm::errs() << "Casting input: " << v << " of type " << tensorType
                   << "\n";
      RankedTensorType tiledTensorTy = getTiledType(tensorType);
      llvm::errs() << "tiledTensorTy: " << tiledTensorTy << "\n";
      return rewriter
          .create<d2m::ViewLayoutOp>(v.getLoc(), tiledTensorTy, v,
                                     /*reinterpretLayout=*/true)
          ->getResult(0);
#else
      RankedTensorType tiledTensorTy = getTiledType(tensorType);
      auto emptyOp = rewriter.create<d2m::EmptyOp>(v.getLoc(), tiledTensorTy);
      return rewriter.create<d2m::ToLayoutOp>(v.getLoc(), v, emptyOp)
          ->getResult(0);
#endif
    };
    auto lhsTiled = getTiledInput(adaptor.getLhs());
    auto rhsTiled = getTiledInput(adaptor.getRhs());

    // 2. tilize the output type, allocate using d2m.empty, and tilize(?) using
    // d2m.to_layout

    // an empty tensor for the output
    auto retEmpty = rewriter.create<d2m::EmptyOp>(loc, retType);

    // a tiled tensor for the output
    RankedTensorType tiledRetTy = getTiledType(retType);
    auto retTiled = rewriter.create<d2m::EmptyOp>(loc, tiledRetTy);
    auto output =
        rewriter.create<d2m::ToLayoutOp>(loc, retEmpty, retTiled)->getResult(0);
    llvm::errs() << "new output: " << output << "\n";

    // 3. introduce d2m.generic which takes the to_layout outputs as inputs
    const size_t outputGridRank = tiledRetTy.getRank() / 2;
    auto grid = ttcore::GridAttr::get(
        rewriter.getContext(), paddedAndSquaredInputGridShape(outputGridRank));
    const unsigned rank = grid.getShape().size();
    llvm::errs() << "grid rank = " << rank << "\n";

    auto [indexingMaps, iteratorTypes] =
        d2m::GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, inputs.size() + 1, rank);
    llvm::errs() << "# of indexing maps = " << indexingMaps.size() << "\n";
    llvm::errs() << "# of iterator types = " << iteratorTypes.size() << "\n";

    SmallVector<Value> tiledInputs = {lhsTiled, rhsTiled};
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, tiledInputs, Value{output}, indexingMaps, iteratorTypes);

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // populate block
      {
        auto getTypeForBlockArg = [&](Type t) -> Type {
          auto tensorType = cast<RankedTensorType>(t);
          ttcore::MetalLayoutAttr layout =
              cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
          auto shardShape = layout.getShardShape(tensorType);
          return RankedTensorType::get(shardShape, tensorType.getElementType());
        };
        block->addArgument(getTypeForBlockArg(lhsTiled.getType()), loc);
        block->addArgument(getTypeForBlockArg(rhsTiled.getType()), loc);
        block->addArgument(getTypeForBlockArg(output.getType()), loc);
        auto blockArgs = block->getArguments();
        for (auto v : blockArgs)
          llvm::errs() << "block arg: " << v << " of type " << v.getType()
                       << "\n";

        const unsigned numInputs = inputs.size();
        const unsigned numOutputs = 1;

        auto linalgIndexingMaps = SmallVector<mlir::AffineMap>(
            numInputs + numOutputs, rewriter.getMultiDimIdentityMap(rank));
        auto linalgIteratorTypes = SmallVector<mlir::utils::IteratorType>(
            iteratorTypes.size(), mlir::utils::IteratorType::parallel);

        // 4. add d2m.tile_add into the d2m.generic
        auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            /* result tensor types */
            llvm::to_vector(
                mlir::ValueRange(blockArgs.take_back(numOutputs)).getTypes()),
            /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              mlir::Value yield;

              llvm::errs() << "Creating tile add op with input types "
                           << bbArgs[0].getType() << " and "
                           << bbArgs[1].getType() << "\n";
              llvm::errs() << "and output type " << bbArgs[2].getType() << "\n";
              // For regular elementwise ops, create TileOp directly.
              yield = bbBuilder.create<tt::d2m::TileAddOp>(
                  loc,
                  /* resultTypes */ bbArgs.take_back(numOutputs).getTypes(),
                  /* operands */ bbArgs.take_front(numInputs));

              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });

        rewriter.create<tt::d2m::YieldOp>(loc, linalgGeneric->getResults());
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    auto outputUntiledEmpty =
        rewriter.create<d2m::EmptyOp>(addOp.getLoc(), retType);
    auto outputUntiled = rewriter.create<d2m::ToLayoutOp>(
        addOp.getLoc(), generic->getResult(0), outputUntiledEmpty);
    rewriter.replaceOp(addOp, outputUntiled);

    return success();
  }

  // Helper to access a canonicalized form of input grid.  This will ensure two
  // things:
  // 1. We square-ify grids, so that transpose etc. will work. e.g. 13x10 ->
  // 10x10.
  // 2. If we wish to have uncollapsed tensors of rank greater than 2, we will
  // 1-pad the leading grid dims.  E.g. a 3d grid will be 1xXxY.
  const llvm::SmallVector<int64_t>
  paddedAndSquaredInputGridShape(size_t rank) const {
    assert(rank >= targetSquareGridShape.size());
    llvm::SmallVector<int64_t> grid(rank, 1);
    const size_t diff = rank - targetSquareGridShape.size();
    for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
      grid[i + diff] = targetSquareGridShape[i];
    }
    return grid;
  }

  // Compute optimal grid shape that works for all provided layout infos.
  llvm::SmallVector<int64_t>
  computeOptimalGrid(ArrayRef<int64_t> physicalShape) const {
    llvm::SmallVector<int64_t> grid;
    grid.reserve(physicalShape.size());

    assert(physicalShape.size() >= targetSquareGridShape.size());

    const size_t gridRankDiff =
        physicalShape.size() - targetSquareGridShape.size();
    grid.assign(gridRankDiff, 1);

    for (size_t i = gridRankDiff; i < physicalShape.size(); ++i) {
      const int64_t dim = physicalShape[i];
      assert(dim > 0);
      // Find largest grid dimension that divides evenly.
      for (int64_t g = targetSquareGridShape[i - gridRankDiff]; g > 0; g--) {
        if (dim % g == 0) {
          grid.push_back(g);
          break;
        }
      }
    }

    assert(grid.size() == physicalShape.size());

    return grid;
  }

  // TODO: retrieve from device info?
  const SmallVector<int64_t> targetSquareGridShape = {1, 1};
};

struct ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    auto addPtr = loadOp.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtr)
      return failure();

    auto fill = addPtr.getPtr().template getDefiningOp<linalg::FillOp>();
    if (!fill)
      return failure();

    auto tensorOfPtrsType =
        dyn_cast<RankedTensorType>(fill.getResult(0).getType());
    if (!tensorOfPtrsType || tensorOfPtrsType.getRank() != 1)
      return failure();

    // Get the base pointer from the fill
    Value basePtr = fill.getInputs().front();
    auto tritonPtrType = cast<PointerType>(basePtr.getType());
    Type elemTy = tritonPtrType.getPointeeType();

    // Prove offsets = start + i
    Value offsets = addPtr.getOffset();
    Value blockStart;
    // TODO: this is very specific to the 1D pattern, need to think about how to
    // generalize to ND and different offset computations
    {
      auto add = offsets.getDefiningOp<arith::AddIOp>();
      if (!add)
        return failure();

      // Find the linalg.fill (scalar -> tensor) operand among add’s operands.
      linalg::FillOp startFill = add.getLhs().getDefiningOp<linalg::FillOp>();
      if (!startFill)
        startFill = add.getRhs().getDefiningOp<linalg::FillOp>();
      if (!startFill)
        return failure();

      Value startScalar = startFill.getInputs().front();
      if (!isa<IntegerType>(startScalar.getType()))
        return failure();
      blockStart = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), startScalar);

      // sanity check
      auto offTy = dyn_cast<RankedTensorType>(add.getResult().getType());
      if (!offTy || offTy.getRank() != 1 ||
          offTy.getDimSize(0) != tensorOfPtrsType.getDimSize(0))
        return failure();
    }

    // cast the source ptr to memref - we can propagate this cast later, though
    // it might make more sense to just rewrite the arguments up front (will
    // that break? should we introduce a ptr -> memref temporary op?)
    auto ptrToMemref =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, typeConverter->convertType(tritonPtrType), basePtr)
            ->getResult(0);
    auto basePtrAsTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({ShapedType::kDynamic}, elemTy), ptrToMemref,
        /*restrictReadonly=*/true);
    llvm::errs() << "ptrToMemref: " << ptrToMemref << " of type "
                 << ptrToMemref.getType() << "\n";
#if 0
    // TODO: should re-enable some sanity checking here
    auto ptrToMemrefType = dyn_cast<RankedTensorType>(ptrToMemref.getType());
    RankedTensorType ptrToMemrefTensorType = RankedTensorType::get({}, elemTy);
    llvm::errs() << "ptrToMemrefTensorType: " << ptrToMemrefTensorType << "\n";
    if (!ptrToMemrefType || ptrToMemrefType.getRank() != tensorOfPtrsType.getRank())
      assert(false && "Unexpected ptr to memref type");
#endif

    // TODO: this is much easier with make_range, we should leave make_range in
    // place and then replace any dangling versions after doing the loads
    unsigned start, end;
    Value cStart, cEnd;
    {
      auto add = offsets.getDefiningOp<arith::AddIOp>();
      if (!add)
        return failure();

      // Find the tt.make_range (scalar -> tensor) operand among add’s operands.
      triton::MakeRangeOp rangeOp =
          add.getLhs().getDefiningOp<triton::MakeRangeOp>();
      if (!rangeOp)
        rangeOp = add.getRhs().getDefiningOp<triton::MakeRangeOp>();
      if (!rangeOp)
        return failure();

      start = rangeOp.getStart();
      end = rangeOp.getEnd();

      cStart = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(rangeOp.getStart()));
      cEnd = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(rangeOp.getEnd()));
    }

    llvm::errs() << "Loading slice [" << start << ", " << end << ")\n";
    auto slicedTy = RankedTensorType::get({end - start}, elemTy);
    Value c1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(1));
    Value cSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(end - start));

    SmallVector<OpFoldResult> offs{blockStart};
    SmallVector<OpFoldResult> szs{rewriter.getIndexAttr(end - start)};
    SmallVector<OpFoldResult> str{rewriter.getIndexAttr(1)};
    auto inferredSliceTy = tensor::ExtractSliceOp::inferResultType(
        RankedTensorType::get({ShapedType::kDynamic}, elemTy),
        /*offsets=*/offs,
        /*sizes=*/szs,
        /*strides=*/str);
    llvm::errs() << "inferredSliceTy: " << inferredSliceTy << "\n";
#if 1
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, inferredSliceTy, basePtrAsTensor, /*offsets=*/offs, /*sizes=*/szs,
        /*strides=*/str);
#else
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get({end - start}, elemTy), basePtrAsTensor,
        /*offsets=*/ValueRange{blockStart}, /*sizes=*/ValueRange{cSize},
        /*strides=*/ValueRange{c1});
#endif

#if 1
    RankedTensorType loadOpTensorType =
        cast<RankedTensorType>(loadOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, szs, elemTy, loadOpTensorType.getEncoding());
#else
    Value init = rewriter.create<tensor::EmptyOp>(loc, szs, elemTy);
#endif
    SmallVector<AffineMap> maps{
        AffineMap::get(/*dims=*/1, /*syms=*/0,
                       rewriter.getAffineDimExpr(0)), // in: (i)->(i)
        AffineMap::get(/*dims=*/1, /*syms=*/0,
                       rewriter.getAffineDimExpr(0)) // out:(i)->(i)
    };
    SmallVector<mlir::utils::IteratorType> iters{
        mlir::utils::IteratorType::parallel};

    llvm::errs() << "slice ty : " << slice.getType() << "\n";
    llvm::errs() << "init ty : " << init.getType() << "\n";
    llvm::errs() << "desired ty : " << loadOp.getResult().getType() << "\n";
    auto gen = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{loadOp.getResult().getType()},
        /*inputs=*/ValueRange{slice},
        /*outputs=*/ValueRange{init},
        /*indexing_maps=*/maps,
        /*iterator_types=*/iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          // args = [%inElem, %outElem]; yield %inElem
          b.create<linalg::YieldOp>(l, args.front());
        });

    // 6) Replace the original tt.load result with the new produced tensor.
    rewriter.replaceOp(loadOp, gen.getResult(0));
    return success();
  }
};

class TritonToTenstorrentTypeConverter : public TypeConverter {
public:
  TritonToTenstorrentTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([this](RankedTensorType tensorType) {
#if 1

      // Assume all tensors are already tiled. Rank 1 tensors get promoted to
      // rank 2
      auto elementType = tensorType.getElementType();
      // llvm::errs() << "element type = " << elementType << "\n";
      if (isa<PointerType>(elementType)) {
        return tensorType;
      }

      // promote rank-1 to rank-2
      SmallVector<int64_t> shape = llvm::to_vector(tensorType.getShape());
      if (shape.size() == 1) {
        shape.push_back(1);
        tensorType = RankedTensorType::get(shape, elementType,
                                           tensorType.getEncoding());
      }

      constexpr std::array<int64_t, 2> defaultShape =
          tt::ttcore::TileType::getDefaultShape();
      SmallVector<int64_t> tileShape{defaultShape[0], defaultShape[1]};
      Type tiledElementType = tt::ttcore::TileType::get(elementType, tileShape);

      SmallVector<int64_t> logicalShape =
          llvm::to_vector(tensorType.getShape());
      tt::ttcore::MemorySpace memSpace = tt::ttcore::MemorySpace::DeviceL1;

      // create metal layout
      tt::ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
          tensorType.getContext(), logicalShape, targetSquareGridShape,
          ttcore::OOBVal::Undef, memSpace, ttcore::TensorMemoryLayout::Sharded);

      // Get raw, unsharded physical shape.
      SmallVector<int64_t> unshardedShape = layout.getPhysicalShape(tileShape);
      llvm::errs() << "unshardedShape = ";
      for (auto s : unshardedShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      // Calculate optimal grid for given physical shape.
      llvm::SmallVector<int64_t> optimalGrid =
          computeOptimalGrid(unshardedShape);
      llvm::errs() << "optimalGrid = ";
      for (auto g : optimalGrid)
        llvm::errs() << g << " ";
      llvm::errs() << "\n";

      // Get optimal sharded, on-device shape.
      llvm::SmallVector<int64_t> shardedShape =
          layout.getDeviceShape(optimalGrid, tileShape);
      llvm::errs() << "shardedShape = ";
      for (auto s : shardedShape)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";

      return RankedTensorType::get(shardedShape, tiledElementType, layout);
#else
      SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
      if (shape.size() == 1) {
        shape.push_back(1);
      }
      return RankedTensorType::get(shape, type.getElementType(),
                                   type.getEncoding());
#endif
    });
    addConversion([](PointerType type) {
      auto elemTy = type.getPointeeType();
      return MemRefType::get({ShapedType::kDynamic}, elemTy);
    });
    addSourceMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, tensorType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([](OpBuilder &builder, RankedTensorType toType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;

      auto input = inputs[0];
      auto inputTensorType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputTensorType)
        return nullptr;

      // TODO: we should probably check element types and encodings here

      auto toRank = toType.getRank();
      auto inputRank = inputTensorType.getRank();
      if (toRank == inputRank)
        return nullptr;
      if (toRank < inputRank) {
        return builder.create<tensor::CollapseShapeOp>(loc, toType, input);
      } else {
        return builder.create<tensor::ExpandShapeOp>(
            loc, toType, input, ArrayRef<ReassociationIndices>{{0, 1}});
      }
    });
  }

  // Helper to access a canonicalized form of input grid.  This will ensure two
  // things:
  // 1. We square-ify grids, so that transpose etc. will work. e.g. 13x10 ->
  // 10x10.
  // 2. If we wish to have uncollapsed tensors of rank greater than 2, we will
  // 1-pad the leading grid dims.  E.g. a 3d grid will be 1xXxY.
  const llvm::SmallVector<int64_t>
  paddedAndSquaredInputGridShape(size_t rank) const {
    assert(rank >= targetSquareGridShape.size());
    llvm::SmallVector<int64_t> grid(rank, 1);
    const size_t diff = rank - targetSquareGridShape.size();
    for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
      grid[i + diff] = targetSquareGridShape[i];
    }
    return grid;
  }

  // Compute optimal grid shape that works for all provided layout infos.
  llvm::SmallVector<int64_t>
  computeOptimalGrid(ArrayRef<int64_t> physicalShape) const {
    llvm::SmallVector<int64_t> grid;
    grid.reserve(physicalShape.size());

    assert(physicalShape.size() >= targetSquareGridShape.size());

    const size_t gridRankDiff =
        physicalShape.size() - targetSquareGridShape.size();
    grid.assign(gridRankDiff, 1);

    for (size_t i = gridRankDiff; i < physicalShape.size(); ++i) {
      const int64_t dim = physicalShape[i];
      assert(dim > 0);
      // Find largest grid dimension that divides evenly.
      for (int64_t g = targetSquareGridShape[i - gridRankDiff]; g > 0; g--) {
        if (dim % g == 0) {
          grid.push_back(g);
          break;
        }
      }
    }

    assert(grid.size() == physicalShape.size());

    return grid;
  }

  // TODO: retrieve from device info?
  const SmallVector<int64_t> targetSquareGridShape = {1, 1};
};

} // namespace

struct ConvertMathTOD2MPass
    : public impl::ConvertMathToD2MBase<ConvertMathTOD2MPass> {
  using impl::ConvertMathToD2MBase<ConvertMathTOD2MPass>::ConvertMathToD2MBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addLegalDialect<tt::d2m::D2MDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<mlir::bufferization::ToTensorOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

     // should make this dynamically illegal
                                        // and just for the tensor version(s)
    // target.addIllegalOp<arith::AddFOp>();

    target.addIllegalOp<triton::LoadOp>();
    // addPtr too?

    TritonToTenstorrentTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    // patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());
    patterns.add<ConvertLoadOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
