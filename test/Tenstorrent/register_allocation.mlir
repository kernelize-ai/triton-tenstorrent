// RUN: triton-opt %s -split-input-file --tritontenstorrent-register-allocation | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @add_kernel
  tt.func public @add_kernel__compute(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[X_PTR:.*]] = tt.splat %arg0
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[X_PTR_OFFSET:.*]] = tt.addptr %[[X_PTR]],
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[Y_PTR:.*]] = tt.splat %arg1
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[Y_PTR_OFFSET:.*]] = tt.addptr %[[Y_PTR]],
    %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[X:.*]] = tt.load %[[X_PTR_OFFSET]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
    %5 = tt.load %2 : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[Y:.*]] = tt.load %[[Y_PTR_OFFSET]] {triton_tenstorrent.alloc_offset = 1 : i32, triton_tenstorrent.alloc_size = 1 : i32}
    %6 = tt.load %4 : tensor<1024x!tt.ptr<f32>, #blocked>
    // COM: Make sure the binary_compute op uses the new load ops and not an intermediate cvt
    // CHECK: %[[OUTPUT:.*]] = triton_tenstorrent.binary_compute["arith.addf"] %[[X]], %[[Y]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
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
  tt.func public @matmul_kernel__compute(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) {
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

    // CHECK: %[[OUTPUT:.*]] = arith.truncf {{.*}} {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
    %c = arith.truncf %accumulator#0 : tensor<32x32xf32, #blocked2> to tensor<32x32xf16, #blocked2>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %7 = ttg.convert_layout %c : tensor<32x32xf16, #blocked2> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_1:.*]] = ttg.convert_layout %[[OUTPUT]] : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked>
    // CHECK: tt.store {{.*}}, %[[OUTPUT_CVT_1]] : tensor
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
  tt.func public @matmul_kernel__compute(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) {
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

    // CHECK: %[[OUTPUT:.*]] = arith.truncf {{.*}} {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
    %c = arith.truncf %accumulator : tensor<32x32xf32, #blocked2> to tensor<32x32xf16, #blocked2>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %7 = ttg.convert_layout %c : tensor<32x32xf16, #blocked2> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[OUTPUT_CVT_1:.*]] = ttg.convert_layout %[[OUTPUT]] : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked>
    // CHECK: tt.store {{.*}}, %[[OUTPUT_CVT_1]] : tensor
    tt.store %c_ptrs, %7 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
 }
}


// -----

// CHECK-DAG: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @matmul_kernel_tma__compute
    tt.func public @matmul_kernel_tma__compute(%arg10: !tt.tensordesc<tensor<32x64xf16>>, %offs_am: i32, %offs_bn: i32, %a_7: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b_8: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg20: tensor<32x64xf32, #blocked>) {
      %accumulator = tt.dot %a_7, %b_8, %arg20 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
      // CHECK: %[[OUTPUT:.*]] = arith.truncf {{.*}} {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32}
      %c = arith.truncf %accumulator : tensor<32x64xf32, #blocked> to tensor<32x64xf16, #blocked>
      %3 = ttg.convert_layout %c : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #blocked1>
      tt.descriptor_store %arg10[%offs_am, %offs_bn], %3 : !tt.tensordesc<tensor<32x64xf16>>, tensor<32x64xf16, #blocked1>
      tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @muladd_kernel__compute
  tt.func public @muladd_kernel__compute(%arg0: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc {alloc_idx = 3 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %z = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %1 = ttc.block_end
    %2 = ttc.block_start
    scf.for %arg5 = %2 to %1 step %c1_i32  : i32 {
      %3 = ttc.current_block %arg5 : i32
      // CHECK: %[[X:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %x_0 = ttg.local_load %x : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: %[[Y:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 1 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %y_1 = ttg.local_load %y : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: %[[Z:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 2 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %z_2 = ttg.local_load %z : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: %[[OUT1:.*]] = triton_tenstorrent.binary_compute["arith.mulf"] %[[X]], %[[Y]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %output = triton_tenstorrent.binary_compute["arith.mulf"] %x_0, %y_1 : (tensor<1024xbf16, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
      // CHECK: triton_tenstorrent.binary_compute["arith.addf"] %[[OUT1]], %[[Z]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %output_3 = triton_tenstorrent.binary_compute["arith.addf"] %output, %z_2 : (tensor<1024xbf16, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
      ttg.local_store %output_3, %0 : tensor<1024xbf16, #blocked> -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @muladd_kernel__compute
  tt.func public @muladd_kernel__compute(%arg0: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc {alloc_idx = 3 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %z = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    %1 = ttc.block_end
    %2 = ttc.block_start
    scf.for %arg5 = %2 to %1 step %c1_i32  : i32 {
      %3 = ttc.current_block %arg5 : i32
      // CHECK: %[[X:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %x_0 = ttg.local_load %x : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: %[[Y:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 1 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %y_1 = ttg.local_load %y : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: %[[OUT1:.*]] = triton_tenstorrent.binary_compute["arith.mulf"] %[[X]], %[[Y]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %output = triton_tenstorrent.binary_compute["arith.mulf"] %x_0, %y_1 : (tensor<1024xbf16, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
      // CHECK: %[[Z:.*]] = ttg.local_load %{{.*}} {triton_tenstorrent.alloc_offset = 1 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %z_2 = ttg.local_load %z : !ttg.memdesc<1024xbf16, #shared, #smem, mutable> -> tensor<1024xbf16, #blocked>
      // CHECK: triton_tenstorrent.binary_compute["arith.addf"] %[[OUT1]], %[[Z]] {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32}
      %output_3 = triton_tenstorrent.binary_compute["arith.addf"] %output, %z_2 : (tensor<1024xbf16, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
      ttg.local_store %output_3, %0 : tensor<1024xbf16, #blocked> -> !ttg.memdesc<1024xbf16, #shared, #smem, mutable>
    }
    tt.return
  }
}
