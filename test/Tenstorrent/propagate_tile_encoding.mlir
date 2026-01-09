// RUN: triton-opt %s -split-input-file --tritontenstorrent-propagate-tile-encoding | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @add_kernel
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[X_PTR:.*]] = tt.splat %arg0
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[X_PTR_OFFSET:.*]] = tt.addptr %[[X_PTR]],
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[Y_PTR:.*]] = tt.splat %arg1
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[Y_PTR_OFFSET:.*]] = tt.addptr %[[Y_PTR]],
    %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[X_CVT:.*]] = ttg.convert_layout %[[X_PTR_OFFSET]] : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: %[[X:.*]] = tt.load %[[X_CVT]] : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %5 = tt.load %2 : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[Y_CVT:.*]] = ttg.convert_layout %[[Y_PTR_OFFSET]] : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: %[[Y:.*]] = tt.load %[[Y_CVT]] : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %6 = tt.load %4 : tensor<1024x!tt.ptr<f32>, #blocked>
    // COM: Make sure the binary_compute op uses the new load ops and not an intermediate cvt
    // CHECK: %[[OUTPUT:.*]] = triton_tenstorrent.binary_compute["arith.addf"] %[[X]], %[[Y]] : {{.*}} -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %7 = triton_tenstorrent.binary_compute["arith.addf"] %5, %6 : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COM: Make sure the store type is propagate to the binary compute op without a cvt by checking for direct use of %OUTPUT
    // CHECK: tt.store {{.*}} %[[OUTPUT]]
    tt.store %9, %7 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @matmul_kernel
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) {
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked2>
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 0 : i32

    %a_ptrs = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>

    %accumulator:3 = scf.for %accumulator_52 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%accumulator_53 = %cst_2, %a_ptrs_1 = %a_ptrs, %b_ptrs_1 = %b_ptrs) -> (tensor<32x32xf32, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %a = arith.muli %accumulator_52, %c32_i32 : i32
      %a_56 = arith.subi %K, %a : i32
      %a_60 = tt.load %a_ptrs_1 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %b = tt.splat %a_56 : i32 -> tensor<32x1xi32, #blocked1>
      %b_63 = tt.load %b_ptrs_1 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      %a_64 = ttg.convert_layout %a_60 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
      %b_65 = ttg.convert_layout %b_63 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
      %accumulator_66 = tt.dot %a_64, %b_65, %accumulator_53 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<32x32xf32, #blocked2>
      %a_ptrs_67 = tt.addptr %a_ptrs_1, %cst : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %b_ptrs_68 = arith.muli %stride_bk, %c32_i32 : i32
      %b_ptrs_69 = tt.splat %b_ptrs_68 : i32 -> tensor<32x32xi32, #blocked1>
      %b_ptrs_70 = tt.addptr %b_ptrs_1, %b_ptrs_69 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %accumulator_66, %a_ptrs_67, %b_ptrs_70 : tensor<32x32xf32, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }

    // CHECK: %[[OUTPUT:.*]] = arith.truncf
    %c = arith.truncf %accumulator#0 : tensor<32x32xf32, #blocked2> to tensor<32x32xf16, #blocked2>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %7 = ttg.convert_layout %c : tensor<32x32xf16, #blocked2> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_1:.*]] = ttg.convert_layout %[[OUTPUT]] : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_TILE:.*]] = ttg.convert_layout %[[OUTPUT_CVT_1]] : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: tt.store {{.*}}, %[[OUTPUT_CVT_TILE]] : tensor<32x32x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    tt.store %c_ptrs, %7 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
 }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @matmul_kernel
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) {
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked2>
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 0 : i32

    %a_ptrs = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>

    %a_60 = tt.load %a_ptrs : tensor<32x32x!tt.ptr<f16>, #blocked>
    %b = tt.splat %K : i32 -> tensor<32x1xi32, #blocked1>
    %b_63 = tt.load %b_ptrs : tensor<32x32x!tt.ptr<f16>, #blocked1>
    %a_64 = ttg.convert_layout %a_60 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %b_65 = ttg.convert_layout %b_63 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %accumulator = tt.dot %a_64, %b_65, %cst_2 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<32x32xf32, #blocked2>

    // CHECK: %[[OUTPUT:.*]] = arith.truncf
    %c = arith.truncf %accumulator : tensor<32x32xf32, #blocked2> to tensor<32x32xf16, #blocked2>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %7 = ttg.convert_layout %c : tensor<32x32xf16, #blocked2> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_1:.*]] = ttg.convert_layout %[[OUTPUT]] : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_TILE:.*]] = ttg.convert_layout %[[OUTPUT_CVT_1]] : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: tt.store {{.*}}, %[[OUTPUT_CVT_TILE]] : tensor<32x32x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    tt.store %c_ptrs, %7 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
 }
}
