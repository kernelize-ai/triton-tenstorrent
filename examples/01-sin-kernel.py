"""
Sin Kernel
==========
This script demonstrates an elementwise `sin` kernel using Triton, run through
the Tenstorrent D2M/ttnn lowering path.

.. code-block:: bash

    TRITON_TTMLIR_TARGET="d2m" TT_SYSTEM_DESC_PATH=path-to-sys-desc-from-ttrt.ttsys python 01-sin-kernel.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sin_kernel(x_ptr,  # *Pointer* to the input vector.
               output_ptr,  # *Pointer* to the output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def sin(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def torch_pcc(golden, result):
    x = golden.detach().reshape(-1).float()
    y = result.detach().reshape(-1).float()
    m = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[m], y[m]
    # torch.corrcoef treats rows as variables, cols as observations; the
    # off-diagonal of the 2x2 is the Pearson correlation between the two vectors.
    if x.numel() < 2 or x.std() == 0 or y.std() == 0:
        return 1.0 if torch.allclose(x, y) else 0.0
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def validate(size, dtype):
    print(f"{size=}, verification torch vs triton: ")
    x = torch.randn((size, ), device="cpu", dtype=torch.float32).to(dtype)

    torch_result = torch.sin(x.to(torch.float32))
    triton_result = sin(x)
    print(f"triton_result: {triton_result}")
    print(f"torch_result: {torch_result}")
    pcc = torch_pcc(torch_result, triton_result.to(torch.float32))
    print(f"PCC = {pcc:.6f}")
    assert pcc > 0.97, f"PCC too low: {pcc:.6f} (expected > 0.97)"
    print("Success!")
    print()


if __name__ == "__main__":
    torch.manual_seed(0)

    validate(32, torch.float32)
    validate(1024, torch.float32)
    validate(98432, torch.float32)
