#ifndef TRITON_TENSTORRENT_ANALYSIS_REG_ALLOC_H
#define TRITON_TENSTORRENT_ANALYSIS_REG_ALLOC_H

#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir {
namespace triton {
namespace npu {

class RegAllocationAnalysis;

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, specified using a start and an end
/// values: [Start, End).
template <typename T> class Interval {
public:
  Interval() {}
  Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }
  T start() const { return Start; }
  T end() const { return End; }
  T size() const { return End - Start; }
  bool contains(T Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const Interval &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const Interval &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const Interval &R) const { return !(*this == R); }
  bool operator<(const Interval &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  T Start = std::numeric_limits<T>::min();
  T End = std::numeric_limits<T>::max();
};

template <class T> Interval(T, T) -> Interval<T>;

class RegAllocation {
public:
  /// A unique identifier for register buffers
  using BufferId = size_t;
  using BufferIdSetT = DenseSet<BufferId>;
  using FuncAllocMapT = CallGraph<RegAllocation>::FuncDataMapT;

  static constexpr BufferId InvalidBufferId =
      std::numeric_limits<BufferId>::max();

  RegAllocation() = default;
  /// Creates a new RegAllocation analysis that computes the register
  /// information for all associated register values.
  explicit RegAllocation(Operation *operation) : operation(operation) {}

  /// Runs allocation analysis on the given top-level operation.
  void run(FuncAllocMapT &funcAllocMap);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Returns the offset of the given buffer in the registers.
  size_t getOffset(BufferId bufferId) const {
    return bufferSet.at(bufferId).offset;
  }

  /// Returns the size of the given buffer in the registers.
  size_t getAllocatedSize(BufferId bufferId) const {
    return bufferSet.at(bufferId).size;
  }

  /// Returns the allocated interval of the given buffer.
  Interval<size_t> getAllocatedInterval(BufferId bufferId) const {
    auto &buffer = bufferSet.at(bufferId);
    return Interval<size_t>(buffer.offset, buffer.offset + buffer.size);
  }

  /// Returns the buffer id of the given value.
  /// This interface only returns the allocated buffer id.
  /// If you want to get all the buffer ids that are associated with the given
  /// value, including alias buffers, use getBufferIds.
  BufferId getBufferId(Value value) const {
    if (valueBuffer.count(value)) {
      return valueBuffer.lookup(value)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns all the buffer ids of the given value, including alias buffers.
  BufferIdSetT getBufferIds(Value value) const {
    BufferIdSetT bufferIds;
    auto allocBufferId = getBufferId(value);
    if (allocBufferId != InvalidBufferId)
      bufferIds.insert(allocBufferId);
    for (auto *buffer : aliasBuffer.lookup(value)) {
      if (buffer->id != InvalidBufferId)
        bufferIds.insert(buffer->id);
    }
    return bufferIds;
  }

  /// Returns the scratch buffer id of the given value.
  BufferId getBufferId(Operation *operation) const {
    if (operation->getNumResults() == 1) {
      Value value = operation->getResult(0);
      if (valueBuffer.count(value)) {
        return valueBuffer.lookup(value)->id;
      }
    }
    return InvalidBufferId;
  }

  /// Returns the size of total registers allocated
  size_t getRegisterSize() const { return registerSize; }

  /// Returns mapping from operation to list of live registers
  using ValueBufferIdT = llvm::MapVector<Value, SmallVector<BufferId>>;
  ValueBufferIdT getLiveBuffers();

public:
  /// A class that represents a register buffer
  struct BufferT {
    /// Explicit: ttg.local_alloc
    enum class BufferKind { Explicit };

    BufferKind kind;
    BufferId id;
    Value value;
    size_t size;
    size_t alignment;
    size_t offset;

    bool operator==(const BufferT &other) const { return id == other.id; }
    bool operator<(const BufferT &other) const { return id < other.id; }

    BufferT(BufferKind kind, BufferId id, Value value, size_t size,
            size_t alignment = 1, size_t offset = 0)
        : kind(kind), id(id), value(value), size(size), alignment(alignment),
          offset(offset) {}

    Operation *getOwner() const { return value.getDefiningOp(); }

    size_t setOffsetAligned(size_t newOffset) {
      return offset = llvm::alignTo(newOffset, alignment);
    }
  };

  /// Value -> Explicit Buffer
  using ValueBufferMapT = llvm::MapVector<Value, BufferT *>;
  /// Value -> Alias Buffer
  using AliasBufferMapT = llvm::MapVector<Value, llvm::SetVector<BufferT *>>;
  /// BufferId -> Buffer
  using BufferSetT = std::map<BufferId, BufferT>;

private:
  template <BufferT::BufferKind Kind, typename KeyType, typename... Args>
  void addBuffer(KeyType &key, Args &&...args) {
    if (valueBuffer.count(key)) {
      return;
    }
    BufferId nextId = bufferIdCounter++;
    auto [it, inserted] = bufferSet.insert_or_assign(
        nextId, BufferT(Kind, nextId, key, std::forward<Args>(args)...));
    BufferT *buffer = &it->second;
    if constexpr (Kind == BufferT::BufferKind::Explicit) {
      valueBuffer[key] = buffer;
    } else {
      assert(false && "Invalid buffer kind");
    }
  }

  void addAlias(Value value, Value alloc) {
    if (value == alloc) {
      return;
    }
    if (valueBuffer.count(value)) {
      // If the value was already allocated, we need to remove it from the value
      // buffer
      valueBuffer.erase(value);
    }
    if (!valueBuffer.count(alloc)) {
      for (auto *buffer : aliasBuffer.lookup(alloc)) {
        aliasBuffer[value].insert(buffer);
      }
      return;
    }
    aliasBuffer[value].insert(valueBuffer[alloc]);
  }

private:
  Operation *operation = nullptr;
  ValueBufferMapT valueBuffer;
  AliasBufferMapT aliasBuffer;
  BufferSetT bufferSet;
  size_t registerSize = 0;

  size_t bufferIdCounter = 0;

  friend class RegAllocationAnalysis;
};

/// Static analysis that computes the allocation of register buffers
/// of the entire call graph.
/// The allocation is performed in a post-order walk of the call graph.
class ModuleRegAllocation : public CallGraph<RegAllocation> {
public:
  using FuncOffsetMapT = DenseMap<FunctionOpInterface, Value>;

  ModuleRegAllocation(ModuleOp moduleOp) : CallGraph<RegAllocation>(moduleOp) {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order edge walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order node walk callback
        [&](FunctionOpInterface funcOp) {
          auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
          if (inserted)
            iter->second.run(funcMap);
        });
  }

  size_t getRegisterSize() {
    size_t size = 0;
    for (auto funcOp : getRoots()) {
      auto *alloc = getFuncData(funcOp);
      size = std::max(size, alloc->getRegisterSize());
    }
    return size;
  }

  size_t getRegisterSize(FunctionOpInterface funcOp) {
    return getFuncData(funcOp)->getRegisterSize();
  }

  void setFunctionRegisterValue(FunctionOpInterface funcOp, Value value) {
    registerValue[funcOp] = value;
  }

  Value getFunctionRegisterBase(FunctionOpInterface funcOp) {
    return registerValue[funcOp];
  }

private:
  FuncOffsetMapT registerValue;
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_TENSTORRENT_ANALYSIS_REG_ALLOC_H
