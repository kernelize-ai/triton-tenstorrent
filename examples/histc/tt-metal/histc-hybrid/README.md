# histc-hybrid — compute-engine locally, `noc_semaphore_inc` to merge

Third variant of the `torch.histc` example, sitting between the other two:

| | `../histc` | `../histc-atomic` | `histc-hybrid` (this one) |
| --- | --- | --- | --- |
| cores | 1 | 1 | 4 |
| per-bin counting | compute engine, compare-and-reduce over the *whole* tensor | scalar RISC-V loop, one element at a time | compute engine, compare-and-reduce over *this core's shard* |
| atomics | none | one `noc_semaphore_inc` per **element** | one `noc_semaphore_inc` per **(core, bin)**, incrementing by the whole local count |
| passes over data | O(bins × n_tiles) | O(n_elements), scalar | O(bins × n_tiles / num_cores) per core, in parallel |

The idea: keep the part the Tensix compute engine is actually fast at (SIMD
compare + reduce over tiles) instead of moving it to a scalar RISC-V loop
like `../histc-atomic` did, and use the NOC atomic-increment primitive only
for what it's cheap at -- merging a handful of per-core partial sums,
rather than scattering every individual element.

## how it works

1. Input is split into `kNumCores` (4) equal contiguous tile shards, one per
   core, padded up to a multiple of `kNumCores * TILE_ELEMS` with the same
   out-of-range sentinel `../histc` uses.
2. Each core's `reader.cpp` reads only its own shard.
3. Each core's `histc_compute.cpp` runs `../histc`'s exact compare-and-reduce
   algorithm, unmodified, just scoped to that shard -- so it produces a
   local per-bin count, not a global one.
4. Each core's `aggregate.cpp`:
   - the leftmost core (the "aggregator") zeroes a shared L1 counters
     buffer (`cb_hist`) and signals the other cores it's safe to start;
   - every core (aggregator included) then loops over its `bins` local
     counts and does one `noc_semaphore_inc(hist_addr + bin*4, count)` per
     bin against the aggregator's `cb_hist` -- the atomic increment amount
     is the local count itself, not a fixed 1, so this is `bins` atomic ops
     per core rather than `bins * (tiles worth of elements)`;
   - non-aggregator cores signal a "done" semaphore when finished; the
     aggregator waits for all of them, then streams `cb_hist` to DRAM.

## why this is the version worth actually considering

`../histc-atomic`'s README concludes its per-element scatter probably isn't
a win because it moves all the work onto a RISC-V core with no hardware FPU
and pays a NOC atomic transaction per element. This version doesn't have
that problem: the per-element work (comparisons, summation) never leaves
the SIMD compute engine, atomics only ever carry a handful of `bins`-sized
partial sums, and the compute-engine work itself is now split `kNumCores`
ways instead of running serially. This is the shape you'd actually reach
for if you wanted `noc_semaphore_inc` to help rather than hurt.

## known simplifications / API risk areas

Not built or run against a real tree, same as the other two examples. The
multi-core semaphore handshake in `aggregate.cpp` is the least-verified
part of any of the three histc examples -- specifically:

- `CreateSemaphore(program, core_range, initial_value)` returning an ID
  used as a compile-time arg, and `get_semaphore(id)` resolving it to a
  local L1 address inside the kernel.
- `noc_semaphore_wait(local_ptr, value)` (busy-poll a local L1 word until
  it reaches `value`) and `noc_semaphore_inc(noc_addr, incr)` with an
  `incr` other than 1 (adding a whole count in one atomic transaction, not
  just bumping by one).
- The assumption that `CreateCircularBuffer(program, core_range, config)`
  assigns a CB index the *same* L1 offset on every core in `core_range` --
  relied on so every core can locate the aggregator's `cb_hist` by reading
  its own local copy of the address, without a third handshake just to
  hand that address around.
- `my_x[0]` / `my_y[0]` for this core's own coordinates, and
  `noc_async_atomic_barrier()` to make sure increments have landed before
  the next semaphore signal is trusted -- both carried over from
  `../histc-atomic`, same caveat there applies here.
- The single-row, aggregator-is-the-leftmost-core layout is hardcoded
  (`agg_x + 1 + c` in `aggregate.cpp`); a real multi-row or larger grid
  would need real core coordinates instead of that assumption.

Before trusting any of this, the fastest sanity check is to diff the
semaphore/handshake shape here against an existing multi-core tt-metal
example that does a producer/consumer or gather handoff (multicast matmul,
sharded reduce, etc.) -- that pattern is common enough in the real
`tt_metal/programming_examples` tree that it's a better reference than
this file's memory of it.

Building: same instructions as `../histc/README.md`.
