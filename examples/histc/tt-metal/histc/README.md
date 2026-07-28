# histc — TT-Metal port of `triton/histc.py`

A single-core TT-Metal kernel implementing `torch.histc`: bucket a 1D
floating-point tensor into `bins` equal-width bins over `[min, max]` and
return per-bin counts. See `../../triton/histc.py` for the Triton original
and the exact `torch.histc` semantics being matched (range auto-detection
when `min == max == 0`, degenerate-range widening, inclusive top edge).

## why this isn't a literal port

Triton's kernel computes a bin index per element and does one pass with
`tl.atomic_add` scattering into the histogram. Tensix's compute engine has
no comparable random-access scatter/atomic-add, so this kernel instead does
**compare-and-reduce, one bin at a time**:

```
for each bin b, edges [lo, hi):
    mask = (x >= lo) & (x < hi)   # per-element, SFPU unary compares
    count[b] = sum(mask)          # tile reduce engine
```

That's `O(bins * n_tiles)` passes over the data instead of Triton's single
`O(n)` pass — the trade for hardware that can't scatter-write. It's also
exactly the kind of op this trade-off matters for: `tt/readme.md` notes that
GLM-5's MoE routing breaks on tt-xla/tt-mlir precisely because
`histc(int) → cumsum(int32) → _grouped_mm` has no TTNN lowering. This is
what a from-scratch, hand-written implementation of that missing op looks
like at the Metalium level.

## files

| file | role |
| --- | --- |
| `histc.cpp` | Host program: builds the padded input, uploads it, creates the circular buffers and the three kernels below, runs the program, and checks the result against a CPU reference (mirrors the `if __name__ == "__main__"` block in `triton/histc.py`). |
| `kernels/dataflow/reader.cpp` | Reads the whole (padded) input into a resident circular buffer once, and fills an all-ones "scaler" tile for the reduce engine. |
| `kernels/compute/histc_compute.cpp` | The actual histogram: per bin, per tile, build a 0/1 mask and reduce it. See the comment at the top of that file for the two-phase structure and why it's split that way. |
| `kernels/dataflow/writer.cpp` | Streams one output tile per bin back to DRAM. |

## known simplifications (reference-kernel, not production op)

- **Single core, whole input resident in L1.** The reader loads every tile
  once and the compute kernel re-addresses it by index for every bin — no
  re-streaming from DRAM, but the whole (padded) tensor must fit in one
  core's L1. Fine at the scale of the Triton demo (10k elements); a
  production version would tile across the grid and/or re-stream per bin.
- **bfloat16 data.** Tiles are native bf16 (~7-bit mantissa), like the other
  `tt_metal/programming_examples`. `torch.histc` normally runs in fp32;
  expect occasional off-by-one bin assignment right at a bin edge from the
  rounding, which is why `histc.cpp`'s self-check uses a tolerance instead
  of an exact match.
- **One tile per bin in the output.** `reduce_tile` with `REDUCE_SCALAR`
  produces its sum inside a full 32×32 tile (element 0 is the real count);
  this kernel doesn't bother packing `bins` scalars densely into fewer
  tiles, so the DRAM output buffer is `bins` tiles and the host only reads
  element 0 of each.

## building

This wasn't built or run against hardware in this session (by request —
see the commit that added it). To build it against a real checkout, either:

- copy `histc/` under `tt_metal/programming_examples/` in a tt-metal source
  tree and add `add_subdirectory(histc)` to that directory's
  `CMakeLists.txt` (same pattern as every other example there), or
- build it standalone against an installed Metalium package:
  `find_package(TT-Metalium)` (see `CMakeLists.txt`) needs
  `CMAKE_PREFIX_PATH`/`TT_METAL_HOME` pointed at the built tree.

The kernel source paths in `histc.cpp` (`"kernels/dataflow/reader.cpp"`,
etc.) are relative to this directory; if `CreateKernel`'s path resolution
in your tree expects paths relative to some other root (as the
`tt_metal/programming_examples` examples sometimes do), adjust them
accordingly.

Run with `TT_METAL_DPRINT_CORES=0,0` set if you want any `DPRINT` output
from the kernels (none is added by default).
