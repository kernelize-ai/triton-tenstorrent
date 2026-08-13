"""Triton kernel implementing torch.histc.

torch.histc(input, bins=100, min=0, max=0) buckets the elements of a
floating-point tensor into `bins` equal-width bins over [min, max] and
returns per-bin counts (as a float tensor). Elements outside [min, max]
are dropped; if min == max == 0, the range is taken from the data itself.

The kernel below does the bucketing + counting in one pass with a global
histogram in float32, accumulated via tl.atomic_add across blocks.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _histc_kernel(
    x_ptr,
    hist_ptr,
    n_elements,
    min_val,
    max_val,
    bin_width,
    n_bins,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Sentinel for out-of-bounds lanes: guaranteed to fail the range check
    # below regardless of min_val, so it never needs its own branch.
    x = tl.load(x_ptr + offsets, mask=mask, other=min_val - 1.0)

    in_range = mask & (x >= min_val) & (x <= max_val)

    bin_idx = tl.floor((x - min_val) / bin_width).to(tl.int32)
    # x == max_val lands one past the last bin before clamping; torch.histc
    # treats the top edge as inclusive to the last bin.
    bin_idx = tl.minimum(bin_idx, n_bins - 1)
    bin_idx = tl.maximum(bin_idx, 0)

    tl.atomic_add(hist_ptr + bin_idx, 1.0, mask=in_range)


def histc(input: torch.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0) -> torch.Tensor:
    """Drop-in Triton replacement for torch.histc."""
    if not input.is_floating_point():
        raise RuntimeError("histc only supports floating point tensors")

    x = input.contiguous().view(-1)
    n = x.numel()

    min_val, max_val = float(min), float(max)
    if min_val == 0.0 and max_val == 0.0:
        if n == 0:
            min_val, max_val = 0.0, 1.0
        else:
            min_val, max_val = x.min().item(), x.max().item()

    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    hist = torch.zeros(bins, dtype=torch.float32, device=input.device)
    if n == 0:
        return hist.to(input.dtype)

    bin_width = (max_val - min_val) / bins
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _histc_kernel[grid](
        x, hist, n,
        min_val, max_val, bin_width, bins,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return hist.to(input.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cases = [
        dict(bins=10, min=0.0, max=0.0),
        dict(bins=4, min=0.0, max=3.0),
        dict(bins=50, min=-2.0, max=2.0),
        dict(bins=1, min=0.0, max=0.0),
    ]
    for i, kwargs in enumerate(cases):
        x = torch.randn(10_000, device=device)
        got = histc(x, **kwargs)
        want = torch.histc(x, **kwargs)
        torch.testing.assert_close(got, want)
        print(f"case {i} {kwargs}: OK (sum={got.sum().item():.0f})")

    empty = torch.empty(0, device=device)
    got = histc(empty, bins=5, min=0.0, max=1.0)
    want = torch.histc(empty, bins=5, min=0.0, max=1.0)
    torch.testing.assert_close(got, want)
    print("empty-input case: OK")

    print("all cases match torch.histc")
