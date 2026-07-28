# histc-atomic — L1 scatter/atomic-increment variant of `../histc`

Same op as `../histc` (`torch.histc` via a single-core TT-Metal kernel), but
where `../histc` works around Tensix's lack of a scatter/atomic-add by doing
compare-and-reduce one bin at a time, this version does the thing that
workaround was avoiding: a genuine one-pass "compute a bin index per
element, atomically bump its counter" scatter, matching
`triton/histc.py`'s `tl.atomic_add` shape. It exists to answer a direct
question: *can a TT-Metal kernel write to an arbitrary L1 address instead of
a whole tile, and is there an atomic op for L1?* Yes to both, with caveats
below.

## how it works

The scatter can't happen on the Tensix **compute** engine — that one only
moves data through `pack_tile`/`copy_tile` into circular-buffer tile slots,
which is the real restriction `../histc`'s README is describing. It's not a
whole-chip restriction, though: the **data-movement** kernels (the RISC-V
cores running `reader`/`writer`-style code) have L1 memory-mapped directly,
so they can dereference any address, and the NOC hardware supports an
atomic-increment transaction against any 4-byte-aligned L1 word — that's
what semaphores are built on, but the primitive isn't restricted to
addresses obtained from `CreateSemaphore()`.

So this version drops the compute kernel entirely and puts the whole
histogram on two data-movement kernels:

- `kernels/dataflow/counter.cpp` reads the padded input into L1 (same as
  `../histc`'s reader), zero-initializes a shared L1 buffer of `bins`
  `uint32_t` counters (`cb_hist`, a plain circular-buffer page — nothing
  semaphore-specific about it), then scans the input **scalar, element by
  element**: expand each bf16 value to float, compute its bin index, and
  issue `noc_semaphore_inc()` against that bin's counter address.
- `kernels/dataflow/writer.cpp` waits for the counting to finish and streams
  the raw counter buffer to DRAM with `noc_async_write` — an arbitrary
  byte-range write, not `noc_async_write_tile`, since this isn't tile data.

## why this probably isn't actually better

It trades `../histc`'s `O(bins * n_tiles)` tile-wide SFPU/reduce passes for
`O(n_elements)` *scalar* work on a RISC-V core with (generally) no hardware
FPU — so the float divide per element is software-emulated — plus one NOC
atomic transaction per element instead of per tile. Fewer total passes over
the data, but the per-element cost is much higher and none of it uses the
SIMD-width compute engine at all. Whether this wins depends entirely on
`bins` vs `n_elements` at your actual scale; `../histc`'s README makes the
same point about its own tradeoff. Nothing here is measured — see "not
built" below.

## known simplifications / API risk areas

This was written from memory of the TT-Metal API, the same way `../histc`
was, and **wasn't built or run against a real tree in this session**. The
parts most likely to need adjustment if a checkout disagrees:

- `noc_semaphore_inc()` on a plain circular-buffer address rather than one
  from `CreateSemaphore()` — the mechanism should be address-based, not
  semaphore-ID-based, but confirm your version doesn't gate it some other
  way.
- `my_x[0]` / `my_y[0]` for this core's own NOC coordinates (needed because
  even a same-core "loopback" atomic increment is issued as a NOC
  transaction with an explicit target) and `noc_async_atomic_barrier()` to
  make sure all increments have landed before the writer kernel reads the
  counters — both used from memory of the pattern, not verified here.
- `InterleavedAddrGen<true>` (the non-tile-shaped page generator, for the
  writer's single small page) alongside `InterleavedAddrGenFast<true>` (the
  tile-shaped one `../histc` and the reader here both use) — check both
  still exist with these names/signatures.
- `DataFormat::UInt32` for the `cb_hist` circular buffer.
- Same bfloat16/single-core/whole-input-resident-in-L1 scope limits as
  `../histc` — see that README for those.

Building: same instructions as `../histc/README.md` (copy under
`tt_metal/programming_examples/`, or build standalone against an installed
Metalium package with `TT_METAL_HOME` set).
