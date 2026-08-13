# The `histc.py` challenge

## Goal

Get `examples/histc.py` (a Triton kernel implementing `torch.histc`) running end to end on this
repo's Tenstorrent NPU backend (branch `atomics`), by fixing whatever is broken in the lowering
from Triton IR to TTKernel.

## Why it's hard: the kernel's shape doesn't fit the backend's atomic model

`histc.py`'s kernel loads a `BLOCK_SIZE=1024` block of input elements, computes a per-lane bin
index, and calls `tl.atomic_add(hist_ptr + bin_idx, 1.0, mask=in_range)` — one atomic increment
per lane, each targeting a **different, data-dependent address** (whichever bin that lane's value
falls into). In Triton IR this is a single `tt.atomic_rmw` whose pointer operand is
**tensor-shaped**: a per-lane scatter.

This backend's atomic lowering (`npu/lib/TritonNPUToTenstorrent/AtomicOpToTTKernel.cpp`) only
supports **scalar atomics on a uniform address** — a global counter, or a split-K-style
accumulation into one shared output element. This is a hardware reality, not an oversight: the
NoC's only real atomic primitive is a fetch-and-add against a single 4-byte-aligned L1 address
(`noc_semaphore_inc`); everything else (add/cas/max/min, DRAM targets) is built in software as a
spinlock-guarded read/modify/write. There is no hardware scatter/indirect-addressing primitive to
build a *per-lane* version of that on top of. Tensor-shaped `tt.atomic_rmw` is explicitly rejected
by `ConvertAtomicRMWOp`, and there's a regression test
(`test/Tenstorrent/atomic_op_to_ttkernel_scatter.mlir`) asserting that rejection is intentional.

So `histc.py` as written needs either:
- **Rewritten to avoid scatter** — e.g. `BLOCK_SIZE=1` (one scalar atomic per grid iteration,
  matching the "one static atomic call site executed repeatedly across grid iterations" pattern
  the existing lowering is built for), or
- **A new mutex + whole-tile accumulate design** — build a local per-bin histogram in vectorized
  compute (no scatter), then merge it into the global histogram under a lock as one bulk
  read-modify-write of the whole output tile, using `tl.atomic_cas`/`tl.atomic_xchg` as a
  hand-rolled spinlock (both already scalar-legal) plus a `tl.reduce`-based local count.

Neither path is implemented in `histc.py` yet. The second path is architecturally the more
faithful one (keeps the kernel vectorized) and is what this session's work was building toward.

## Bugs found and fixed this session

Investigating why `histc.py` failed surfaced five independent, real compiler bugs — each one a
genuine gap, not something specific to this kernel:

1. **`TritonFuncOpToFuncOp.cpp`** — scalar `f32` kernel arguments (e.g. `min_val`, `bin_width`)
   were read as raw `i32` common-arg words and never bitcast to `f32`, causing an
   "unresolved materialization from i32 to f32" failure. This was the *original*, first-hit bug
   blocking `histc.py`. **Fixed**: added a `Float32Type` branch that bitcasts.

2. **`CoreSpecialize.cpp` — atomics duplicated across threads.** Atomic ops
   (`tt.atomic_rmw`/`tt.atomic_cas`) weren't classified as load-like or store-like, so
   `CoreSpecialize` cloned them unmodified into *all three* specialized thread functions
   (reader/compute/writer) instead of confining them to one. On compute this crashed
   (`createTensorAccessor`'s assertion — the compute thread never gets a `TensorAccessorArgs`
   chain built, since it never issues NoC ops); on reader+writer it would have silently run the
   same lock/read/write sequence twice, redundantly. **Fixed**: atomics are now erased from
   reader and compute, kept only on writer.

3. **`CoreSpecialize.cpp` — scalar stores crashed `createSharedBuffer`.** Any plain scalar
   (non-tensor) `tt.store` — e.g. writing back a reduced or atomic scalar result — crashed
   `createSharedBuffer`'s `cast<RankedTensorType>`, which unconditionally assumed every store's
   value was tensor-shaped. Confirmed with a minimal isolated repro (`tl.store(out_ptr, 42)`
   alone, no reduce/atomics involved). **Fixed**: scalar stores now skip the CB/shared-buffer path
   entirely and, like atomics, are confined to the writer thread only.

4. **`MemoryOpToTTKernel.cpp` — scalar store lowering didn't exist.** `ConvertStoreOp` only ever
   handled "store a tensor tile loaded through a CB" — a plain scalar value store hit
   `assert(false && "expected store from a local load op")`. **Fixed**: added a scalar branch that
   stages the value through a dedicated scratch L1 slot (a freshly reserved compile-time
   semaphore, mirroring how the atomic lowering reserves its own lock/scratch semaphores) and
   NoC-writes it out as a single element.

5. **`TritonNPUToTTKernel.cpp` — tile-register bookkeeping crashed on an empty compute thread.**
   Once atomics/scalar-stores stopped being duplicated into the compute thread (fixes #2/#3), a
   compute thread with *no* tile computation at all became a real, reachable case — and the
   tile-regs-acquire/commit/release insertion logic unconditionally assumed at least one
   `PackTileOp` was present. **Fixed**: guarded the whole tile-regs bookkeeping step on the
   compute function actually having pack-tile ops.

All five were validated by rebuilding `triton.0` (which consumes this repo as a Triton backend
plugin via `TRITON_PLUGIN_DIRS`) and re-running small isolated probe kernels — not `histc.py`
itself — after each fix.

## New feature added: direct `tt.reduce` lowering

Added `npu/lib/TritonNPUToTenstorrent/ReduceOpToTTKernel.cpp`: a `tt.reduce` → TTKernel pattern
for the case needed by the mutex/whole-tile-accumulate design above — rank-1, axis-0 (full)
reduce, int32 element type, single Sum or Max combiner. It mirrors the existing (but currently
unreachable, D2M-only) `D2M/ReduceOpToD2M.cpp` in *strategy* — classify the combiner, use the
int32 Sum/Max SFPU path — but targets TTKernel ops directly, the same way `DotOpToTTKernel.cpp`
converts `tt.dot` straight to TTKernel rather than routing through D2M.

Mechanically: `tt.reduce` over a 1D tensor produces a genuine **scalar** SSA value in Triton IR
(confirmed by direct inspection), but `ttkernel.sfpu_reduce` reduces a DST tile *in place*,
leaving the result trapped in a compute-engine register — not something a scalar consumer (a
store, an atomic) can use directly. The lowering bridges this by packing the reduced tile out to a
**dedicated scratch CB reserved per call site** (mirroring the atomic lowering's per-call-site
semaphore reservation) and reading the first element back over L1, the same
`CastToL1PtrOp`/`LoadFromL1Op` idiom the atomic lowering already uses for its own return value.
Register allocation support for `tt.reduce` was added to `RegAlloc.cpp` alongside it.

This compiles cleanly but **is not yet validated end-to-end** — see below.

## Where it stopped: a runtime crash, not a compiler bug

The natural test for the reduce lowering (`tl.sum(...)` then `tl.store`) depends on the scalar
store fix (#4 above). With all five MLIR-level fixes in place, that probe now compiles cleanly
through the *entire* pipeline — but crashes **at runtime**, on-device, with `free(): invalid
pointer` deep inside the tt-metal/knexus native runtime (`libtt_plugin.so`), not in any of this
repo's MLIR code. Likely cause: a buffer/accessor sizing mismatch in how the new scalar-store
lowering treats a small, non-tile-aligned output buffer via `createTensorAccessor`. Debugging this
needs native tooling (gdb backtraces, possibly ASAN) rather than MLIR-level reasoning, and was
intentionally left for a follow-up session.

## Status summary

| Piece | Status |
|---|---|
| Original `histc.py` materialization bug (f32 args) | Fixed |
| Atomic-op thread duplication crash | Fixed |
| Scalar-store `CoreSpecialize` crash | Fixed |
| Scalar-store lowering (didn't exist) | Fixed |
| Empty-compute-thread tile-regs crash | Fixed |
| Direct `tt.reduce` lowering | Added, compiles, **not runtime-validated** |
| Scalar-store runtime crash (native) | **Open** |
| `histc.py` kernel itself | **Still doesn't run** — needs rewrite (scalar-per-iteration atomics, or the mutex+reduce whole-tile design) once the runtime crash is resolved |

## Files touched

- `npu/lib/TritonNPUToTenstorrent/TritonFuncOpToFuncOp.cpp`
- `npu/lib/Dialect/TritonTenstorrent/Transforms/CoreSpecialize.cpp`
- `npu/lib/TritonNPUToTenstorrent/MemoryOpToTTKernel.cpp`
- `npu/lib/TritonNPUToTenstorrent/TritonNPUToTTKernel.cpp`
- `npu/lib/Dialect/TritonTenstorrent/Transforms/RegAlloc.cpp`
- `npu/lib/TritonNPUToTenstorrent/ReduceOpToTTKernel.cpp` (new)
- `npu/lib/TritonNPUToTenstorrent/PatternTritonNPUToTenstorrent.h`
- `npu/lib/TritonNPUToTenstorrent/CMakeLists.txt`

None of these changes are committed — they're sitting as working-tree modifications on the
`atomics` branch.
