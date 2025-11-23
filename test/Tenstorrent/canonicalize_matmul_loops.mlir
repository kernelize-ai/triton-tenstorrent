// RUN: triton-opt %s -split-input-file --tritontenstorrent-canonicalize-matmul-loops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @matmul_kernel
  tt.func public @matmul_kernel__compute(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32} , %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32} , %M: i32 {tt.divisibility = 8 : i32} , %N: i32 {tt.divisibility = 8 : i32} , %K: i32 {tt.divisibility = 8 : i32} , %stride_am: i32 {tt.divisibility = 8 : i32} , %stride_bk: i32 {tt.divisibility = 8 : i32} , %stride_cm: i32 {tt.divisibility = 8 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x32xi32, #blocked2>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32

    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>

    %a_ptrs = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked2>
    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>

    // CHECK: %[[ACCUMULATOR:.*]] = scf.for
    %accumulator:3 = scf.for %accumulator_34 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%accumulator_35 = %cst, %a_ptrs_36 = %a_ptrs, %b_ptrs_37 = %b_ptrs) -> (tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %a_38 = ttg.local_load %a : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_39 = ttg.local_load %b : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      // CHECK: %[[LOOP_ACCUMULATOR:.*]] = tt.dot
      %accumulator_40 = tt.dot %a_38, %b_39, %accumulator_35 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
      %a_ptrs_41 = tt.addptr %a_ptrs_36, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
      %b_ptrs_42 = arith.muli %stride_bk, %c32_i32 : i32
      %b_ptrs_43 = tt.splat %b_ptrs_42 : i32 -> tensor<32x32xi32, #blocked1>
      %b_ptrs_44 = tt.addptr %b_ptrs_37, %b_ptrs_43 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      // CHECK: scf.yield %[[LOOP_ACCUMULATOR]] : tensor<32x32xf32, #blocked>
      scf.yield %accumulator_40, %a_ptrs_41, %b_ptrs_44 : tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    // CHECK: arith.truncf %[[ACCUMULATOR]]
    %c = arith.truncf %accumulator#0 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    ttg.local_store %c, %0 : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}
