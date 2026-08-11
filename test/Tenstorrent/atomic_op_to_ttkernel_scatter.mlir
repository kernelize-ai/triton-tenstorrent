// RUN: not triton-opt %s --convert-triton-npu-to-ttkernel 2>&1 | FileCheck %s

// Tensor-shaped (per-lane scatter) atomics are out of scope for this
// backend today (no general indirect addressing) and must fail to convert
// rather than silently mis-lower.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @atomic_add_scatter_kernel__reader(%ptr: !tt.ptr<i32> {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %val = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    // CHECK: failed to legalize operation 'tt.atomic_rmw' that was explicitly marked illegal
    %old = tt.atomic_rmw add, relaxed, gpu, %ptrs, %val : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }
}
