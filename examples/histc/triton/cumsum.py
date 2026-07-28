"""Triton kernel implementing torch.cumsum.

torch.cumsum(input, dim) returns the inclusive running sum of `input`
along `dim`, same shape and (mostly) same dtype as the input.

The tensor is viewed as (M, N, K): M = product of dims before `dim`,
N = size of `dim`, K = product of dims after `dim` (K == 1 when `dim`
is the last one). Each Triton program owns one (m, k) "line" of N
elements strided by K, and scans it in BLOCK_SIZE-sized chunks with a
carried running total -- so N can exceed BLOCK_SIZE without needing
any cross-program communication.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cumsum_kernel(
    x_ptr,
    out_ptr,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // K
    k = pid % K
    base = m * N * K + k

    running = tl.zeros((), dtype=x_ptr.dtype.element_ty)
    for start in tl.range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        ptrs = x_ptr + base + offs * K
        x = tl.load(ptrs, mask=mask, other=0)
        scanned = tl.cumsum(x, axis=0) + running
        tl.store(out_ptr + base + offs * K, scanned, mask=mask)
        running += tl.sum(x, axis=0)


def cumsum(input: torch.Tensor, dim: int) -> torch.Tensor:
    """Drop-in Triton replacement for torch.cumsum."""
    ndim = input.ndim
    if dim < 0:
        dim += ndim
    if not 0 <= dim < ndim:
        raise IndexError(f"dim {dim} out of range for tensor of rank {ndim}")

    # torch.cumsum on bool upcasts to int64; mirror that rather than
    # summing into a 1-bit accumulator.
    compute_dtype = torch.int64 if input.dtype == torch.bool else input.dtype
    x = input.contiguous().to(compute_dtype)
    shape = x.shape

    N = shape[dim]
    M = 1
    for s in shape[:dim]:
        M *= s
    K = 1
    for s in shape[dim + 1:]:
        K *= s

    out = torch.empty_like(x)
    if N == 0 or M * K == 0:
        return out

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    grid = (M * K,)
    _cumsum_kernel[grid](x, out, N, K, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cases = [
        dict(shape=(4096,), dim=0),          # 1D, N > BLOCK_SIZE
        dict(shape=(8, 16), dim=1),          # last dim
        dict(shape=(8, 16), dim=0),          # first dim
        dict(shape=(4, 5000, 3), dim=1),     # middle dim, N > BLOCK_SIZE, K > 1
        dict(shape=(4, 5000, 3), dim=-2),    # negative dim indexing
        dict(shape=(1,), dim=0),             # single element
    ]
    for i, c in enumerate(cases):
        x = torch.randn(*c["shape"], device=device)
        got = cumsum(x, c["dim"])
        want = torch.cumsum(x, c["dim"])
        # Chunked accumulation sums in a different order than torch's
        # sequential scan, so long running sums pick up float rounding
        # drift; loosen tolerance accordingly instead of expecting bitwise
        # equality.
        torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)
        print(f"case {i} shape={c['shape']} dim={c['dim']}: OK")

    # integer and bool dtypes
    xi = torch.randint(-5, 5, (4, 2000), device=device, dtype=torch.int64)
    torch.testing.assert_close(cumsum(xi, 1), torch.cumsum(xi, 1))
    print("int64 case: OK")

    xb = torch.randint(0, 2, (4, 100), device=device, dtype=torch.bool)
    torch.testing.assert_close(cumsum(xb, 1), torch.cumsum(xb, 1))
    print("bool case: OK")

    print("all cases match torch.cumsum")
