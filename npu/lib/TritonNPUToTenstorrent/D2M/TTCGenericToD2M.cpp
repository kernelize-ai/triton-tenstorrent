#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/TritonNPUToD2M/Passes.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h" // ttc.GenericOp

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/TypeSwitch.h"

#include "SPMDArgs.h"
#include "TTCGenericPlan.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_CONVERTTTCGENERICTOD2M
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "convert-ttc-generic-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct OperandTypes {
  ttcore::MetalLayoutAttr layout;
  RankedTensorType funcArg;
  RankedTensorType cast;
  RankedTensorType view;
  RankedTensorType shard;
};

// TODO: de-dupe with FuncOpToD2MGeneric
static ttnn::TTNNLayoutAttr getTTNNLayoutTensor(MLIRContext *context,
                                                ArrayRef<int64_t> scalarShape,
                                                Type elementType) {
  auto memSpaceAttr =
      ttcore::MemorySpaceAttr::get(context, ttcore::MemorySpace::DeviceDRAM);
  // TODO: is there ever a case where this would be L1?
  ttnn::BufferType bufferType =
      memSpaceAttr.getValue() == ttcore::MemorySpace::DeviceL1
          ? ttnn::BufferType::L1
          : ttnn::BufferType::DRAM;

  return ttnn::TTNNLayoutAttr::Builder(context, scalarShape,
                                       elementType)
      .setBufferType(bufferType)
      .setMemoryLayout(
          ttnn::TensorMemoryLayout::Interleaved) // support sharded?
      .build();
}

// TODO: remove gridShape (now unused)
static OperandTypes computeOperandTypes(GenericPlan::Operand &operand,
                                        triton::FuncOp tritonFunc,
                                        ArrayRef<int64_t> gridShape,
                                        IRRewriter &rewriter) {
  OperandTypes ret;

  auto tileShape = ttcore::TileType::getDefaultShape();
  auto tileTy = ttcore::TileType::get(operand.elementType, tileShape);

  ret.layout = ttcore::MetalLayoutAttr::get(
      rewriter.getContext(), operand.logicalShape,
      ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Interleaved);
  auto ttnnLayout =
      getTTNNLayoutTensor(rewriter.getContext(), operand.logicalShape, tileTy);
  // TODO: we have the logical shape, we should be able to use it now instead of
  // plugging in a dynamic shape tensor arg
  SmallVector<int64_t> dynShape(operand.tensorTiles.size(),
                                ShapedType::kDynamic);
  ret.funcArg = RankedTensorType::get(operand.logicalShape, operand.elementType,
                                      ttnnLayout);
  ret.cast = RankedTensorType::get(ret.layout.getDeviceShape({1, 1}, tileShape),
                                   tileTy, ret.layout);
  ret.view = cast<RankedTensorType>(
      d2m::utils::reblockShapedType(ret.cast, operand.tensorTiles));
  ret.shard = RankedTensorType::get(ret.layout.getShardShape(ret.view), tileTy);

  return ret;
}

static FailureOr<func::FuncOp>
emitSignature(triton::FuncOp tritonFunc, const GenericPlan &plan,
              ArrayRef<OperandTypes> operandTypes, IRRewriter &rewriter) {
  MLIRContext *context = rewriter.getContext();
  Block *entry = &tritonFunc.getBody().front();

  // map the rewritten operands to their original arg indices. Because we
  // maintain the signature of the existing triton kernel, we need to work from
  // the existing arg indices.
  DenseMap<unsigned, unsigned> operandForArg;
  for (auto [i, operand] : llvm::enumerate(plan.operands))
    operandForArg.try_emplace(operand.funcArg.getArgNumber(), i);

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  std::optional<unsigned> resultIndex;

  for (BlockArgument arg : entry->getArguments()) {
    // TODO: TensorDescType
    if (isa<triton::PointerType>(arg.getType())) {
      auto it = operandForArg.find(arg.getArgNumber());
      if (it == operandForArg.end())
        return tritonFunc.emitOpError()
               << "kernel argument #" << arg.getArgNumber()
               << " is a pointer that no tt.load or tt.store in the "
                  "ttc.generic touches, so its layout cannot be inferred";
      argTypes.push_back(operandTypes[it->second].funcArg);
      if (it->second >= plan.numInputs)
        resultIndex = argTypes.size() - 1; // the DPS output is the result
    } else {
      argTypes.push_back(arg.getType());
    }
    argLocs.push_back(arg.getLoc());
  }

  // TODO: if we let d2m.generic build its own grid, we can drop the triton
  // provided grid params
  for (unsigned i = 0; i < (unsigned)SpmdArg::Count; ++i) {
    argTypes.push_back(rewriter.getI32Type());
    argLocs.push_back(
        NameLoc::get(StringAttr::get(context, spmdArgName((SpmdArg)i))));
  }

  assert(resultIndex && "populateOperands guarantees exactly one tt.store");
  auto funcTy = rewriter.getFunctionType(argTypes, argTypes[*resultIndex]);
  auto newFunc = func::FuncOp::create(rewriter, tritonFunc.getLoc(),
                                      tritonFunc.getName(), funcTy);
  ttmlir::utils::setFunctionType(newFunc,
                                 ttmlir::utils::FunctionType::ForwardDevice);
  rewriter.createBlock(&newFunc.getBody(), newFunc.getBody().end(), argTypes,
                       argLocs);

  return newFunc;
}

// Converts kernel function arguments into the form the implicit form of the
// d2m.generic op consumes
static Value materializeOperand(BlockArgument funcArg,
                                const OperandTypes &types,
                                IRRewriter &rewriter) {
  Location loc = funcArg.getLoc();
  Value cast =
      ttir::TTNNMetalLayoutCastOp::create(rewriter, loc, types.cast, funcArg);
  return d2m::ViewLayoutOp::create(rewriter, loc, types.view, cast);
}

static d2m::GenericOp emitGeneric(Location loc, const GenericPlan &plan,
                                  ArrayRef<Value> operands,
                                  IRRewriter &rewriter) {
  ArrayRef<Value> ins = operands.take_front(plan.numInputs);
  ArrayRef<Value> outs = operands.drop_front(plan.numInputs);

  SmallVector<AffineMap> indexingMaps;
  indexingMaps.reserve(plan.operands.size());
  for (const GenericPlan::Operand &operand : plan.operands)
    indexingMaps.push_back(operand.indexingMap);

  return d2m::GenericOp::create(
      rewriter, loc, ins, outs, /*additionalArgs=*/ValueRange(),
      rewriter.getAffineMapArrayAttr(indexingMaps),
      rewriter.getArrayAttr(plan.iteratorTypes), d2m::ThreadType::Unified,
      ttcore::GridAttr::get(rewriter.getContext(), plan.gridShape),
      plan.blockFactors);
}

static Value emitTileOp(Operation *op, ValueRange operands, Location loc,
                        OpBuilder &builder) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<arith::AddFOp>([&](auto) {
        return d2m::TileAddOp::create(builder, loc, operands[0].getType(),
                                      operands[0], operands[1]);
      })
      .Default([](Operation *) { return Value(); });
}

static bool isTranslatableDataOp(Operation *op) {
  return isa<arith::AddFOp>(op);
}

static LogicalResult emitGenericRegion(const GenericPlan &plan,
                                       d2m::GenericOp generic,
                                       ArrayRef<OperandTypes> operandTypes,
                                       IRRewriter &rewriter) {
  // validate maps in the data plane have a supported d2m translation target
  for (Operation *op : plan.dataOps)
    if (!isTranslatableDataOp(op))
      return op->emitOpError("has no d2m tile-op equivalent");

  Block *entry = rewriter.createBlock(&generic.getRegion(0));
  rewriter.setInsertionPointToStart(entry);

  const unsigned numInputs = plan.numInputs;
  SmallVector<Value> linalgIns, linalgOuts;
  for (unsigned i = 0; i < plan.operands.size(); ++i) {
    Location loc = plan.operands[i].funcArg.getLoc();
    RankedTensorType shardTy = operandTypes[i].shard;
    Value buffer = tensor::EmptyOp::create(rewriter, loc, shardTy.getShape(),
                                           shardTy.getElementType());
    if (i >= numInputs) {
      linalgOuts.push_back(buffer);
      continue;
    }
    SmallVector<Value> indices =
        d2m::utils::buildGridIndices(rewriter, loc, generic.getIndexingMap(i));
    linalgIns.push_back(
        d2m::RemoteLoadOp::create(rewriter, loc, shardTy, buffer,
                                  generic->getOperand(i), indices)
            .getResult());
  }

  SmallVector<AffineMap> linalgMaps;
  for (const GenericPlan::Operand &operand : plan.operands)
    linalgMaps.push_back(operand.indexingMap);

  SmallVector<mlir::utils::IteratorType> linalgIterators;
  for (Attribute attr : plan.iteratorTypes)
    linalgIterators.push_back(cast<ttcore::IteratorTypeAttr>(attr).getValue() ==
                                      ttcore::IteratorType::Parallel
                                  ? mlir::utils::IteratorType::parallel
                                  : mlir::utils::IteratorType::reduction);

  auto buildBody = [&](OpBuilder &b, Location bodyLoc, ValueRange bbArgs) {
    // Seed from the boundary ops: each tt.load's result becomes the matching
    // linalg block argument
    IRMapping mapping;
    for (unsigned i = 0; i < numInputs; ++i)
      mapping.map(plan.operands[i].boundaryOp->getResult(0), bbArgs[i]);

    for (Operation *op : plan.dataOps) {
      SmallVector<Value> tileOperands;
      for (Value operand : op->getOperands()) {
        Value mapped = mapping.lookupOrNull(operand);
        // classify() rejects any op that is in neither the data nor the control
        // plane, so a data op's operands are always load results or the results
        // of earlier data ops -- both already mapped.
        assert(mapped && "data op operand escaped plane classification");
        tileOperands.push_back(mapped);
      }
      mapping.map(op->getResult(0), emitTileOp(op, tileOperands, bodyLoc, b));
    }

    auto store = cast<triton::StoreOp>(plan.operands.back().boundaryOp);
    linalg::YieldOp::create(b, bodyLoc, mapping.lookup(store.getValue()));
  };

  auto linalgOp = linalg::GenericOp::create(
      rewriter, generic.getLoc(), /*resultTypes=*/TypeRange(linalgOuts),
      linalgIns, linalgOuts, linalgMaps, linalgIterators, buildBody);

  SmallVector<Value> storeResults;
  for (unsigned i = numInputs; i < plan.operands.size(); ++i) {
    Location loc = plan.operands[i].funcArg.getLoc();
    SmallVector<Value> indices =
        d2m::utils::buildGridIndices(rewriter, loc, generic.getIndexingMap(i));
    Value view = generic->getOperand(i);
    storeResults.push_back(
        d2m::RemoteStoreOp::create(rewriter, loc, view.getType(), view, indices,
                                   linalgOp->getResult(i - numInputs))
            .getResult());
  }
  d2m::YieldOp::create(rewriter, generic.getLoc(), storeResults);

  return success();
}

} // namespace

struct ConvertTTCGenericToD2MPass
    : public impl::ConvertTTCGenericToD2MBase<ConvertTTCGenericToD2MPass> {

  LogicalResult emitFunction(triton::FuncOp tritonFunc,
                             MutableArrayRef<GenericPlan> plans,
                             IRRewriter &rewriter) {
    rewriter.setInsertionPoint(tritonFunc);

    if (plans.size() != 1) {
      return tritonFunc.emitError("expected only one generic plan for "
                                  "ttc.generic -> D2M.generic lowering");
    }

    auto &plan = plans.front();

    SmallVector<OperandTypes> operandTypes;
    for (auto &operand : plan.operands) {
      operandTypes.push_back(
          computeOperandTypes(operand, tritonFunc, plan.gridShape, rewriter));
    }

    auto funcSigResult =
        emitSignature(tritonFunc, plan, operandTypes, rewriter);
    if (failed(funcSigResult))
      return funcSigResult;
    func::FuncOp newFunc = *funcSigResult;

    rewriter.setInsertionPointToStart(&newFunc.getBody().front());
    SmallVector<Value> operandViews;
    for (auto [operand, types] : llvm::zip(plan.operands, operandTypes)) {
      BlockArgument newArg =
          newFunc.getArgument(operand.funcArg.getArgNumber());
      operandViews.push_back(materializeOperand(newArg, types, rewriter));
    }

    // emit the generic op and fill the generic op region
    auto d2mGeneric =
        emitGeneric(newFunc.getLoc(), plan, operandViews, rewriter);

    if (failed(emitGenericRegion(plan, d2mGeneric, operandTypes, rewriter)))
      return failure();

    rewriter.setInsertionPointAfter(d2mGeneric);
    Value result = ttir::TTNNMetalLayoutCastOp::create(
        rewriter, newFunc.getLoc(), newFunc.getFunctionType().getResult(0),
        d2mGeneric->getResult(0));
    func::ReturnOp::create(rewriter, newFunc.getLoc(), result);

    return success();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    auto gridAttr = tt::TritonTenstorrentDialect::getGridAttr(mod);
    SmallVector<int64_t> gridShape = llvm::to_vector(gridAttr.getShape());

    SmallVector<GenericPlan, 4> plans;
    triton::FuncOp tritonFunc;
    mod.walk([&](triton::FuncOp func) {
      // if there are already generic plans from a previous function, bail
      // - we can't cross function boundaries yet
      if (!plans.empty()) {
        func.emitError(
            "expected one parent func for all generic ops in triton kernel");
        signalPassFailure();
      }
      func.walk([&](cpu::GenericOp generic) {
        auto planResult = GenericPlan::build(generic);
        if (failed(planResult)) {
          LDBG("Failed to build generic plan for generic "
               << generic.getHeader());
          signalPassFailure();
          return;
        }

        plans.push_back(*planResult);
        LDBG("plan for " << generic.getHeader() << "\n" << *planResult);
      });
      tritonFunc = func;
    });

    // TODO: using this as a failure signal, but it might be better to signal
    // the walk was interrupted above (or separate the analyze and rewrite steps
    // into functions here)
    if (plans.empty())
      return;

    IRRewriter rewriter(context);
    if (failed(emitFunction(tritonFunc, plans, rewriter)))
      signalPassFailure();

    rewriter.eraseOp(tritonFunc);
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
