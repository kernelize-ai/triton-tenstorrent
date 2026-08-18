# ttnn.generic kernel wire contract

`FuncOpToTTNNGeneric.cpp` (`--convert-triton-npu-to-ttnn-generic`) does not
generate kernel bodies. It only builds the shape of a `ttnn.generic` op —
three kernel descriptors (reader / writer / compute) whose `source` is a
placeholder:

```
#define READER_KERNEL
#include "<kernelName>.h"
```

(`<kernelName>` = the Triton kernel function's name.) A real kernel body
must be supplied out of band as a header named `<kernelName>.h`, discoverable
on the compiler's include path (`backend/driver.py` sets `CPATH` to the repo
root for this — put kernel headers there, e.g. `sin_kernel.h` for a Triton
kernel named `sin_kernel`).

This doc is the wire contract that header must satisfy to match what the
pass actually emits, so a hand-written or AI-generated kernel source lines
up with the wrapper without needing to read the pass's C++.

## File shape

One file, three `#ifdef`-gated `kernel_main()` bodies:

```cpp
#ifdef COMPUTE_KERNEL
void kernel_main() { ... }
#endif

#ifdef READER_KERNEL
void kernel_main() { ... }
#endif

#ifdef WRITER_KERNEL
void kernel_main() { ... }
#endif
```

Only one of the three macros is defined per compile, so each block is
compiled as an independent translation unit against tt-metal's kernel API
(`api/...` headers — resolved automatically, no user include-path setup
needed for those).

## io_tensors index space

Every tensor argument of the Triton kernel (in signature order) becomes an
`io_tensors` entry, **inputs first, then the single output last** —
regardless of where the output pointer falls in the original Triton
signature. Only one output tensor is currently supported.

```
io_tensors = [input_0, input_1, ..., input_{N-1}, output]
```

`N` = number of input tensors; `N+1` = total io tensor count. This index
space is what `operand_index` in `ct_args`/`common_rt_args` below refers to.

## Circular buffers / ct_args[0 .. N]: CB indices

Each io tensor gets one double-buffered CB (2x the tile footprint of a
single load/store on that tensor). The CB's compile-time buffer index is
appended to `ct_args` in io_tensors order:

```
ct_args[i] = CB index for io_tensors[i]        for i in [0, N]
```

Kernel code opens these with `get_compile_time_arg_val(i)`:

```cpp
CircularBuffer cb_in(get_compile_time_arg_val(0));   // io_tensors[0] (first input)
CircularBuffer cb_out(get_compile_time_arg_val(N));  // io_tensors[N] (the output)
```

## ct_args[N+1 ..]: TensorAccessorArgs markers

Immediately after the CB indices, one `TensorAccessorArgs` marker is
appended per io tensor, same order:

```
ct_args[N+1+i] = TensorAccessorArgs marker for io_tensors[i]
```

Each marker expands **at launch time**, against the live buffer, into a
variable-length uint32 sequence (2 uint32s for a plain interleaved/DRAM
tensor: an `ArgsConfig` word and an aligned-page-size word — more if
sharded). Kernel code must consume it with tt-metal's
`TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>` template, chaining the offset
across tensors:

```cpp
auto acc0 = TensorAccessorArgs<N + 1, 0>();
auto acc1 = TensorAccessorArgs<acc0.next_compile_time_args_offset(),
                                acc0.next_common_runtime_args_offset()>();
// ... one per io tensor, in the same io_tensors order
TensorAccessor in_accessor(acc0, /*base_address=*/..., /*page_size=*/...);
```

`CRTA_OFFSET` (0 above) only matters for sharded tensors (runtime-resolved
rank/shape/bank-coords); for a plain interleaved tensor it is unused and any
value works. **The kernel author must hardcode `N+1` as the first
`CTA_OFFSET`** — it is fixed by the Triton kernel's own input count, known
at kernel-authoring time (it's `len(input tensor args)`).

Kernel headers do not get separate `ct_args` per reader/writer/compute —
the pass emits the *same* `ct_args` array to all three kernel descriptors,
so any of the three can reference any io tensor's CB index or accessor
args.

## common_rt_args: shared-across-cores runtime args

`common_rt_args` (read via `get_common_arg_val<T>(i)`) is resolved **once**
per launch and shared by every core. Built by walking the Triton kernel's
converted argument list, in original signature order:

- a tensor arg emits `KernelArgAddressOfTensor(io_tensors index)` →
  resolves to that tensor's buffer base address
- any non-tensor arg emits `KernelArgScalar` → resolves to that scalar's
  runtime value

Two trailing scalars are always synthesized and appended last, after all of
the kernel's own scalar args (the SPMD convention, `SPMDArgs.h`):
`x_grid`, `y_grid` — the **runtime** launch grid size (the actual
`(gridX, gridY)` the caller passed in), *not* a compile-time constant.

So for a kernel `(x_ptr, output_ptr, n_elements, BLOCK_SIZE: constexpr)`:

```
common_rt_args[0] = address of x_ptr        (get_common_arg_val<uint32_t>(0))
common_rt_args[1] = address of output_ptr   (get_common_arg_val<uint32_t>(1))
common_rt_args[2] = n_elements               (get_common_arg_val<uint32_t>(2))
common_rt_args[3] = x_grid                   (get_common_arg_val<uint32_t>(3))
common_rt_args[4] = y_grid                   (get_common_arg_val<uint32_t>(4))
```

(`constexpr` args like `BLOCK_SIZE` never reach this list — they're folded
into the specialized kernel at Triton JIT-compile time and don't need a
runtime slot.)

Note tensor addresses use `io_tensors` index positions (0..N for inputs,
N for output) while everything after them is offset by `N+1` (total
io-tensor count) — this is because `common_rt_args`, at the runtime layer,
resolves against one combined array `io_tensors ++ additional_scalar_args`,
not two separate index spaces.

## rt_args: per-*physical-core* runtime args

`rt_args` (read via `get_arg_val<T>(i)`, no `common_`) is baked **per
physical core**, at compile time — one `CoreRuntimeArgsAttr` per core in
the *device's full core grid* (from the system descriptor, independent of
the Triton launch grid). Currently the pass assigns each core exactly one
linear work-item id, row-major over the grid:

```
core (x, y), linear id = y * num_cols + x
rt_args["tile_start"] = id
rt_args["tile_end"]   = id + 1
```

exposed as named args, retrieved positionally:

```cpp
uint32_t tile_start = get_arg_val<uint32_t>(0);  // "tile_start"
uint32_t tile_end   = get_arg_val<uint32_t>(1);  // "tile_end"
for (uint32_t tile = tile_start; tile < tile_end; ++tile) { ... }
```

**This is a fixed one-tile-per-physical-core assignment**, not a real
work-distribution scheme — it's correct only when the actual runtime launch
grid (`x_grid * y_grid`, i.e. total Triton "programs") equals the physical
core count. It is *not* derived from `x_grid`/`y_grid` (those are only
available via `common_rt_args`, at runtime, and `rt_args` values are baked
before the launch's grid size is known). A kernel that needs a real
grid-stride / persistent-kernel loop over a runtime-sized grid must read
`x_grid`/`y_grid` itself via `get_common_arg_val` and is not yet supported
by anything this pass bakes automatically — that's a follow-up, not
something a hand-written kernel can currently opt into via `rt_args` alone.

## Fixed compute-kernel config

The pass hardcodes `HiFi4` math fidelity, `fp32_dest_acc_en=false`,
`dst_full_sync_en=false`, no unpack-to-dest modes, no bfp8 precise pack, no
math-approx mode. Kernel authors cannot currently vary these per kernel.

## Checklist for a new kernel header

1. Name the file `<triton_fn_name>.h`, place it on `CPATH`.
2. Gate reader/writer/compute bodies behind `READER_KERNEL` /
   `WRITER_KERNEL` / `COMPUTE_KERNEL`.
3. Know `N` = number of Triton **tensor** input args (not counting the
   output, not counting scalars/constexprs).
4. CB indices: `get_compile_time_arg_val(i)` for `i` in `[0, N]`
   (`N` itself is the output's CB).
5. Tensor accessors: `TensorAccessorArgs<N + 1, ...>()`, chained via
   `next_compile_time_args_offset()` per subsequent io tensor.
6. Tensor addresses / scalars: `get_common_arg_val<uint32_t>(i)` — tensors
   at their `io_tensors` index, kernel scalars right after (index `N+1`
   onward, in original signature order), `x_grid`/`y_grid` last.
7. Per-core loop bounds: `get_arg_val<uint32_t>(0)` / `(1)` as
   `[tile_start, tile_end)` — valid only when the launch grid size equals
   the physical core count.
