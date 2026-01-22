#include <algorithm>
#include <limits>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/RegAlias.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/RegAlloc.h"

#define DEBUG_TYPE "tritontenstorrent-register-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tttt = mlir::triton::npu::tt;

using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTREGALLOC
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register Allocation Analysis
//===----------------------------------------------------------------------===//
class RegAllocationAnalysis {
public:
  RegAllocationAnalysis(Operation *operation,
                        RegAllocation::FuncAllocMapT *funcAllocMap,
                        RegAllocation *allocation)
      : operation(operation), funcAllocMap(funcAllocMap),
        allocation(allocation) {
    run();
  }

private:
  using BufferT = RegAllocation::BufferT;

  /// Value -> Liveness Range
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  static constexpr int TT_TILE_DIM_SIZE = 32;

  static size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

  /// Computes the number of tiles required for a given type.
  int getNumTiles(Type type) {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      auto shape = rankedType.getShape();
      if (shape.size() == 1) {
        // 1D tensor, use 1024 size tiles
        return cdiv(shape[0], TT_TILE_DIM_SIZE * TT_TILE_DIM_SIZE);
      } else if (shape.size() == 2) {
        // 2D tensor, use 32x32 size tiles
        return cdiv(shape[0], TT_TILE_DIM_SIZE) *
               cdiv(shape[1], TT_TILE_DIM_SIZE);
      } else {
        assert(false && "Unsupported tensor rank");
      }
    }
    return 0;
  }

  /// Initializes explicitly defined register values for a given operation.
  void getExplicitValueSize(Operation *op) {
    if (op->getNumResults() != 1)
      return;
    Value result = op->getResult(0);
    // ComputeOps, tt.dot,
    if (isa<tt::BinaryComputeOp, arith::TruncIOp, arith::TruncFOp>(op)) {
      for (auto operand : op->getOperands()) {
        int tiles = getNumTiles(operand.getType());
        LDBG("AddExpBuffers: " << op->getName() << " operand (" << tiles
                               << "): " << operand);
        allocation->addBuffer<BufferT::BufferKind::Explicit>(operand, tiles);
      }
      int tiles = getNumTiles(result.getType());
      LDBG("AddExpBuffers: " << op->getName() << " result (" << tiles
                             << "): " << result);
      allocation->addBuffer<BufferT::BufferKind::Explicit>(result, tiles);
    } else if (isa<DotOp>(op)) {
      // Accumulator will be aliased to the result, so we need to allocate the
      // entire result
      int tiles = getNumTiles(result.getType());
      LDBG("AddExpBuffers: " << op->getName() << " result (" << tiles
                             << "): " << result);
      allocation->addBuffer<BufferT::BufferKind::Explicit>(result, tiles);
    }
  }

  /// Initializes aliased registers for a given value.
  void getValueAlias(Value value, RegAliasAnalysis &analysis) {
    dataflow::Lattice<AliasInfo> *latticeElement =
        analysis.getLatticeElement(value);
    if (latticeElement) {
      AliasInfo &info = latticeElement->getValue();
      if (!info.getAllocs().empty()) {
        for (auto alloc : info.getAllocs()) {
          LDBG("getValueAlias: Alias value " << value);
          LDBG("getValueAlias: Alloc value " << alloc);
          allocation->addAlias(value, alloc);
        }
      }
    }
  }

  /// Extract all register values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { getExplicitValueSize(op); });
    // Get the alias values
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    RegAliasAnalysis *aliasAnalysis = solver->load<RegAliasAnalysis>();
    // Run the analysis rooted at every isolated from above operation, including
    // the top-level function but also any nested regions.
    operation->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
          failed(solver->initializeAndRun(op))) {
        // TODO: return error instead of bailing out..
        llvm_unreachable("failed to run RegAliasAnalysis");
      }
    });
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        getValueAlias(operand, *aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, *aliasAnalysis);
      }
    });
  }

  /// Computes the liveness range of the allocated value.
  /// Each buffer is allocated only once.
  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
      LDBG("resolveExplicitBufferLiveness: Value "
           << value << " range " << bufferRange[buffer].start() << " to "
           << bufferRange[buffer].end());
    }
  }

  /// Extends the liveness range by unionizing the liveness range of the aliased
  /// values because each allocated buffer could be an alias of others, if block
  /// arguments are involved.
  void resolveAliasBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (const auto &[value, buffers] : allocation->aliasBuffer) {
      auto range = getLiveness(value);
      LDBG("resolveAliasBufferLiveness: Value "
           << value << " range " << range.start() << " to " << range.end());
      for (auto *buffer : buffers) {
        auto minId = range.start();
        auto maxId = range.end();
        if (bufferRange.count(buffer)) {
          // Extend the allocated buffer's range
          minId = std::min(minId, bufferRange[buffer].start());
          maxId = std::max(maxId, bufferRange[buffer].end());
        }
        LDBG("resolveAliasBufferLiveness: Extending buffer "
             << buffer->id << " range from " << bufferRange[buffer].start()
             << " to " << bufferRange[buffer].end() << " to " << minId << " to "
             << maxId);
        bufferRange[buffer] = Interval(minId, maxId);
      }
    }
  }

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // Assign an ID to each operation using post-order traversal.
    // To achieve the correct liveness range, the parent operation's ID
    // should be greater than each of its child operation's ID .
    // Example:
    //     ...
    //     %5 = const 0.0f
    //     %6 = scf.for ... iter_args(%arg0 = %5) -> (f32) {
    //       %accum = triton.dot %a %b %arg0
    //       ...
    //       scf.yield %accum
    //     }
    //     %7 = trunc.f32 %6
    //     %8 = triton.store %7 %out
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      LDBG("getValueLivenessRange: Value " << value);
      auto liveOperations = liveness.resolveLiveness(value);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        LDBG("getValueLivenessRange: Live operation " << liveOp->getName());
        if (operationId[liveOp] < minId) {
          minId = operationId[liveOp];
        }
        if (operationId[liveOp] > maxId) {
          maxId = operationId[liveOp];
        }
      });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
    resolveAliasBufferLiveness(getValueLivenessRange);
  }

  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
    }
  }

  void dumpAllocationSize() const {
    LDBG("Dump register allocation size -----------");
    auto liveBuffers = allocation->getLiveBuffers();
    int64_t analyzedSize = 0;
    for (auto [value, bufferIds] : liveBuffers) {
      auto bufferId = allocation->getBufferId(value);
      if (bufferId != RegAllocation::InvalidBufferId) {
        int64_t bufferSize = allocation->getAllocatedSize(bufferId);
        analyzedSize = std::max(analyzedSize, bufferSize);
      }
    }
    llvm::dbgs() << "Allocated: " << allocation->registerSize
                 << ", analyzed: " << analyzedSize << "\n";
  }

  void dumpInterferenceGraph(const GraphT &interference) const {
    LDBG("\n");
    LDBG("Dump interference graph: \n");
    for (auto edges : interference) {
      // llvm::dbgs() << "-- from " << edges.first->id << " to ";
      llvm::dbgs() << "-- from " << edges.first << " to ";
      for (auto node : edges.second) {
        llvm::dbgs() << node->id << "; ";
      }
      llvm::dbgs() << "\n";
    }
  }

  /// Computes the register offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://dl.acm.org/doi/pdf/10.5555/314500.315082)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    // Sort buffers by size in descending order to reduce the fragmentation
    // on big buffers caused by smaller buffers. Big buffers have a higher
    // chance to overlap with multiple other buffers, and allocating them first
    // (by calculateStarts) ensures a higher chance that they will occupy a
    // standalone smem slot.
    llvm::stable_sort(
        buffers, [&](BufferT *A, BufferT *B) { return A->size > B->size; });

    calculateStarts(buffers);

    // NOTE: The original paper doesn't consider interference between
    // the bumped ranges. Buffers that previously do not interfere with
    // could interfere after offset bumping if their liveness ranges overlap.
    // Therefore, we rerun the interference graph algorithm after bumping so
    // that we regroup the buffers and color them again. Since we always
    // increase the buffer offset and keep reducing conflicts, we will
    // eventually reach a fixed point.
    GraphT interference;
    buildInterferenceGraph(buffers, interference);
    do {
      allocate(buffers, interference);
      buildInterferenceGraph(buffers, interference);
    } while (!interference.empty());

    LLVM_DEBUG(dumpAllocationSize());
  }

  /// Computes the initial register offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers) {
    //  v = values in registers
    //  t = triplet of (size, start, end)
    //  register space
    //  -
    //  |         *******t4
    //  | /|\ v2 inserts t4, t5, and t6
    //  |  |
    //  | ******t5         ************t6
    //  | ^^^^^v2^^^^^^
    //  |  |      *********************t2
    //  | \|/ v2 erases t1
    //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
    //  |---------------------------------------------| liveness range
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
    // If the available triple's range is less than a given buffer range,
    // we won't know if there has been an overlap without using graph coloring.
    // Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, Interval<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Interval<size_t>()));
    SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto offset = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt =
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (const auto &val : tripleMap)
              res = res &&
                    !val.second.intersects(xRange); // only one buffer intersect
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        // TODO(Keren): A buffer's size shouldn't be determined here, have to
        // clean it up
        size_t alignOffset = buffer->setOffsetAligned(offset);
        tripleMap.insert({alignOffset + xSize,
                          Interval{std::max(range.start(), xRange.start()),
                                   std::min(range.end(), xRange.end())}});
        // We could either insert (range.start, xRange.start) or (range.start,
        // xRange.end), both are correct and determine the potential buffer
        // offset, and the graph coloring algorithm will solve the interference,
        // if any
        if (range.start() < xRange.start())
          tripleMap.insert({offset, Interval{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({offset, Interval{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
    LLVM_DEBUG(dumpBuffers());
  }

  /// Builds a graph of all register values. Edges are created between
  /// register values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = x->offset;
        auto yStart = y->offset;
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);

        // Buffers interfere if their allocation offsets overlap and they are
        // live at the same time.
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }
      }
    }

    LLVM_DEBUG(dumpInterferenceGraph(interference));
  }

  /// Finalizes register offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const GraphT &interference) {
    // Reset register size
    allocation->registerSize = 0;
    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
      LLVM_DEBUG({
        llvm::dbgs() << "-- color " << x->id << " " << colors[x] << "\n";
      });
    }
    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : buffers) {
      size_t newOffset = 0;
      for (auto y : interference.lookup(x)) {
        newOffset = std::max(newOffset, y->offset + y->size);
      }
      if (colors.lookup(x) != 0)
        x->setOffsetAligned(newOffset);
      allocation->registerSize =
          std::max(allocation->registerSize, x->offset + x->size);
    }
    LLVM_DEBUG(dumpBuffers());
  }

private:
  Operation *operation;
  RegAllocation::FuncAllocMapT *funcAllocMap;
  RegAllocation *allocation;
  BufferRangeMapT bufferRange;
};

void RegAllocation::run(FuncAllocMapT &funcAllocMap) {
  RegAllocationAnalysis(getOperation(), &funcAllocMap, this);
}

RegAllocation::ValueBufferIdT RegAllocation::getLiveBuffers() {
  RegAllocation::ValueBufferIdT liveBuffers;

  Operation *rootOperation = getOperation();
  Liveness liveness(rootOperation);
  auto analyzeOperation = [&](Operation *op) -> void {
    for (Value result : op->getOpResults()) {
      auto bufferId = getBufferId(result);
      if (bufferId == InvalidBufferId)
        continue;
      LDBG("getLiveBuffers: Result (" << bufferId << "): " << result);
      liveBuffers[result].push_back(bufferId);
      for (auto [alias, buffers] : aliasBuffer) {
        if (alias == result)
          continue;
        for (auto buffer : buffers) {
          if (buffer->id == bufferId) {
            LDBG("getLiveBuffers: Alias " << alias);
            liveBuffers[alias].push_back(bufferId);
            break;
          }
        }
      }
    }
  };
  rootOperation->walk(analyzeOperation);
  return liveBuffers;
}

////////////////////////////////////////////////////////////
// TritonTenstorrentRegAllocPass
////////////////////////////////////////////////////////////
class TritonTenstorrentRegAllocPass
    : public triton::npu::impl::TritonTenstorrentRegAllocBase<
          TritonTenstorrentRegAllocPass> {

  void assignRegisterOffset(Value result, int offset, int size) {
    if (auto op = result.getDefiningOp()) {
      auto dialect = tt::TritonTenstorrentDialect::getLoaded(op);
      dialect->getAllocOffsetAttrHelper().setAttr(
          op, IntegerAttr::get(IntegerType::get(op->getContext(), 32), offset));
      dialect->getAllocSizeAttrHelper().setAttr(
          op, IntegerAttr::get(IntegerType::get(op->getContext(), 32), size));
      return;
    } else if (auto blockArg = dyn_cast<BlockArgument>(result)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
        if (auto init = loopOp.getTiedLoopInit(blockArg)) {
          assignRegisterOffset(init->get(), offset, size);
          return;
        }
      }
    }
    assert(false && "No defining op found for result ");
  }

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    ModuleRegAllocation moduleRegAllocation(m);

    m.walk([&](FuncOp funcOp) {
      auto funcName = funcOp.getName();
      LDBG("RegAllocPass: Processing compute function " << funcName);
      if (funcName.ends_with("__compute")) {
        auto *funcAllocation = moduleRegAllocation.getFuncData(funcOp);
        assert(funcAllocation && "Function allocation not found");
        RegAllocation::ValueBufferIdT liveBuffers =
            funcAllocation->getLiveBuffers();
        for (auto &[result, buffers] : liveBuffers) {
          for (auto bufferId : buffers) {
            LDBG("RegAllocPass: Buffer id " << bufferId << " for result "
                                            << result);
            int offset = funcAllocation->getOffset(bufferId);
            int size = funcAllocation->getAllocatedSize(bufferId);
            LDBG("RegAllocPass: Setting allocation.offset = "
                 << offset << " with size " << size);
            assignRegisterOffset(result, offset, size);
          }
        }
      }
    });
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
