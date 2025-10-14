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

struct ConvertAddOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting AddOp to D2M: " << addOp << "\n";
    auto loc = addOp.getLoc();

    auto tensorTy = cast<RankedTensorType>(addOp.getType());
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
        auto layout = cast<tt::ttcore::MetalLayoutAttr>(type.getEncoding());
        return layout.getShardShape(type);
      };

      // Populate 'block'.
      {
        llvm::for_each(inputs, [&](Value v) {
          auto type = cast<RankedTensorType>(v.getType());
          auto shardShape = getShardShape(type);
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

  const SmallVector<size_t> targetSquareGridShape = {32, 32};
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

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) {
      // changes the encoding from triton -> tt
      tt::ttcore::MemorySpace memSpace = tt::ttcore::MemorySpace::DeviceL1;
      tt::ttcore::TensorMemoryLayout memLayout =
          tt::ttcore::TensorMemoryLayout::Sharded;
      SmallVector<int64_t> deviceGridShape = {
          32,
          32}; // TODO: get from module attributes, need to populate device info

      // compute alignments and intervals - TODO is probably only correct for 1D
      // tensors
      auto i64Ty = IntegerType::get(type.getContext(), 64);
      auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
      DenseIntElementsAttr collapsedIntervals = DenseIntElementsAttr::get(
          intervalTy, llvm::ArrayRef<int64_t>({0, -1}));
      llvm::SmallVector<int64_t> dimAlignments{32};

      SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
      if (shape.size() == 1) {
        shape.push_back(1);
      }
      assert(shape.size() >= 2 && "tt-mlir expects rank 2+ tensors");
      auto ttLayout = tt::ttcore::MetalLayoutAttr::get(
          type.getContext(), shape, deviceGridShape, tt::ttcore::OOBVal::Undef,
          memSpace, memLayout, collapsedIntervals, dimAlignments);
      return RankedTensorType::get(shape, type.getElementType(), ttLayout);
    });

    RewritePatternSet patterns(context);
    patterns.add<ConvertAddOp>(typeConverter, patterns.getContext());

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
