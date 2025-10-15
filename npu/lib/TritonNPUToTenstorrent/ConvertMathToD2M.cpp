#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTMATHTOD2M
#include "npu/include/TritonNPUToTenstorrent/Passes.h.inc"

using namespace tt;

struct ConvertAddOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting AddOp to D2M: " << addOp << "\n";
    auto loc = addOp.getLoc();

    auto tensorTy = cast<RankedTensorType>(addOp.getType());

    // TODO:
    // 1. insert d2m.to_layout for the lhs and rhs
    // 2. allocate the output and d2m.to_layout it
    // 3. introduce d2m.generic which takes the to_layout outputs as inputs
    // 4. add d2m.tile_add into the d2m.generic
    // 5. replace op with d2m.generic output(s)

#if 1
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

      RankedTensorType tiledTensorTy = getTiledType(tensorType);
      auto emptyOp = rewriter.create<d2m::EmptyOp>(v.getLoc(), tiledTensorTy);
      return rewriter.create<d2m::ToLayoutOp>(v.getLoc(), v, emptyOp)
          ->getResult(0);
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
#if 0
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, bbArgs[2]);
#else
              yield = bbBuilder.create<tt::d2m::TileAddOp>(
                  loc,
                  /* resultTypes */ bbArgs.take_back(numOutputs).getTypes(),
                  /* operands */ bbArgs.take_front(numInputs));

              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
#endif
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

#else

    SmallVector<mlir::AffineMap> indexingMap = {
        rewriter.getMultiDimIdentityMap(tensorTy.getRank())};
    mlir::Attribute iteratorType = tt::ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), tt::ttcore::IteratorType::Parallel);

    auto typeConverter = getTypeConverter();
    auto retType =
        cast<RankedTensorType>(typeConverter->convertType(addOp.getType()));

    Value result = rewriter.create<tt::d2m::EmptyOp>(loc, retType);

    llvm::errs() << "new lhs: " << adaptor.getLhs() << "\n";
    llvm::errs() << "new rhs: " << adaptor.getRhs() << "\n";
    llvm::errs() << "new result: " << result << "\n";

    SmallVector<Value> inputs = {adaptor.getLhs(), adaptor.getRhs()};

    // create the generic region wrapping the compute ops
    auto generic = rewriter.create<tt::d2m::GenericOp>(
        loc, inputs, ValueRange{result},
        rewriter.getAffineMapArrayAttr(indexingMap),
        rewriter.getArrayAttr({iteratorType}));
    // Create one bb in 'generic''s region and set its arguments.

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      auto getShardShape = [](RankedTensorType type) {
        llvm::errs() << "get shard shape for " << type << "\n";
        auto layout = cast<tt::ttcore::MetalLayoutAttr>(type.getEncoding());
        return layout.getShardShape(type);
      };

      // Populate 'block'.
      {
        llvm::for_each(inputs, [&](Value v) {
          auto type = cast<RankedTensorType>(v.getType());
          auto shardShape = getShardShape(type);
          llvm::errs() << "shard shape: ";
          for (auto s : shardShape)
            llvm::errs() << s << " ";
          llvm::errs() << "\n";
          block->addArgument(
              RankedTensorType::get(shardShape, type.getElementType()), loc);
        });
        block->addArgument(
            RankedTensorType::get(
                getShardShape(cast<RankedTensorType>(retType)),
                cast<RankedTensorType>(retType).getElementType()),
            loc);
        auto blockArgs = block->getArguments();

        const unsigned numInputs = inputs.size();
        const unsigned numOutputs = 1;

        const size_t outputGridRank = retType.getRank() / 2;
        auto grid = tt::ttcore::GridAttr::get(
            rewriter.getContext(),
            paddedAndSquaredInputGridShape(outputGridRank));
        const std::size_t rank = grid.getShape().size();

        auto linalgIndexingMaps = SmallVector<mlir::AffineMap>(
            numInputs + numOutputs, rewriter.getMultiDimIdentityMap(rank));
        auto linalgIteratorTypes = SmallVector<mlir::utils::IteratorType>(
            rank, mlir::utils::IteratorType::parallel);

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

#if 1
    rewriter.replaceOp(addOp, generic->getResult(0));
#else
    rewriter.replaceOpWithNewOp<tt::d2m::TileAddOp>(
        addOp, retType, adaptor.getLhs(), adaptor.getRhs());
#endif
#endif
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

struct ConvertMathTOD2MPass
    : public impl::ConvertMathToD2MBase<ConvertMathTOD2MPass> {
  using impl::ConvertMathToD2MBase<ConvertMathTOD2MPass>::ConvertMathToD2MBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::ConversionTarget target{*context};
    target.addIllegalOp<arith::AddFOp>();
    target.addLegalDialect<tt::d2m::D2MDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) {
#if 1
      // make all tensors 2D
      SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
      if (shape.size() == 1) {
        shape.push_back(1);
      }
      return RankedTensorType::get(shape, type.getElementType(),
                                   type.getEncoding());
#else
      // changes the encoding from triton -> tt
      tt::ttcore::MemorySpace memSpace = tt::ttcore::MemorySpace::DeviceL1;
      tt::ttcore::TensorMemoryLayout memLayout =
          tt::ttcore::TensorMemoryLayout::Sharded;
      SmallVector<int64_t> deviceGridShape = {
          32,
          32}; // TODO: get from module attributes, need to populate device info

      auto i64Ty = IntegerType::get(type.getContext(), 64);
      auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
      DenseIntElementsAttr collapsedIntervals = DenseIntElementsAttr::get(
          intervalTy, llvm::ArrayRef<int64_t>({0, -1}));
      llvm::SmallVector<int64_t> dimAlignments{32, 32};

      SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
      if (shape.size() == 1) {
        shape.push_back(1);
      }
      assert(shape.size() >= 2 && "tt-mlir expects rank 2+ tensors");
      auto ttLayout = tt::ttcore::MetalLayoutAttr::get(
          type.getContext(), shape, deviceGridShape, tt::ttcore::OOBVal::Undef,
          memSpace, memLayout, collapsedIntervals, dimAlignments);
      return RankedTensorType::get(shape, type.getElementType(), ttLayout);
#endif
    });
    typeConverter.addSourceMaterialization([](OpBuilder &builder,
                                              RankedTensorType tensorType,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, tensorType, inputs)
          .getResult(0);
    });
#if 1
    typeConverter.addTargetMaterialization(
        [](OpBuilder &builder, RankedTensorType toType, ValueRange inputs,
           Location loc) -> Value {
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
#endif

    RewritePatternSet patterns(context);
    patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
