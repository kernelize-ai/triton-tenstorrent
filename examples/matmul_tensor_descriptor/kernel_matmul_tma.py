"""
Persistent Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.

To build the flatbuffer call:

    scripts/build-d2m-kernel.sh matmul_tma.py \
        --kernel-name matmul_kernel_tma \
        --num-warps 1 --num-stages 2 \
        --grid "(M/32)*(N/32),1,1" \
        --signature "tensordesc<bf16[32,32]>, tensordesc<bf16[32,32]>, tensordesc<bf16[32,32]>, 32, 32, 32, 8" \
        --out-dir kernel_matmul_tma

"""

import argparse
import itertools

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from contextlib import contextmanager

from typing import Optional

def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 1}, num_stages=s,
                      num_warps=w, pre_hook=pre_hook)
        for BM in [32]
        for BN in [32]
        for BK in [32]
        for s in ([1])
        for w in [1]
    ]

def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_tma(a_desc, b_desc, c_desc,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      ):
    M, K = a_desc.shape
    K, N = b_desc.shape
    BLOCK_SIZE_M: tl.constexpr = a_desc.block_shape[0]
    BLOCK_SIZE_K: tl.constexpr = a_desc.block_shape[1]
    BLOCK_SIZE_N: tl.constexpr = b_desc.block_shape[1]

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles):
        offs_k = k * a_desc.block_shape[1]
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator) # removed transpose 

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], accumulator)


## Get kernel function from autotuner
matmul_tma_jit = matmul_kernel_tma.fn


def matmul_tma(a, b):
    # Check constraints.
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    M, K = a.shape
    K, N = b.shape 
    dtype = a.dtype

    c = torch.zeros((M, N), device=a.device, dtype=dtype)

    # A dummy block value that will [maybe?] be overwritten when we have the real block size
    dummy_block = [32, 32]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)
    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel_tma[grid](
        a_desc, b_desc, c_desc,  #
    )
    return c

