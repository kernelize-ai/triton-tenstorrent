// RUN: triton-opt %s -split-input-file --core-specialize -symbol-dce -cse -canonicalize | FileCheck %s

// COM: TODO: add layout checks
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK-LABEL: @add_kernel__reader
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
    // CHECK: ttg.local_store
    // CHECK: ttg.local_store
    // CHECK-NOT: triton_tenstorrent.binary_compute

    // CHECK-LABEL: @add_kernel__compute
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 2 : i32}
    // CHECK: ttg.local_load
    // CHECK: ttg.local_load
    // CHECK: triton_tenstorrent.binary_compute
    // CHECK: ttg.local_store

    // CHECK-LABEL: @add_kernel__writer
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 2 : i32}
    // CHECK: ttg.local_load
    // CHECK: tt.store
    // CHECK-NOT: triton_tenstorrent.binary_compute

    tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
         %c1024_i32 = arith.constant 1024 : i32
        %pid = tt.get_program_id x : i32
        %block_start = arith.muli %pid, %c1024_i32 : i32
        %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %offsets_1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
        %offsets_2 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %offsets_3 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %offsets_4 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
        %offsets_5 = arith.addi %offsets_2, %offsets : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %offsets_6 = arith.addi %offsets_3, %offsets_0 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %offsets_7 = arith.addi %offsets_4, %offsets_1 : tensor<1024xi32, #blocked>
        %mask = tt.splat %n_elements : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %mask_8 = tt.splat %n_elements : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %mask_9 = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked>
        %mask_10 = arith.cmpi slt, %offsets_5, %mask : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %mask_11 = arith.cmpi slt, %offsets_6, %mask_8 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %mask_12 = arith.cmpi slt, %offsets_7, %mask_9 : tensor<1024xi32, #blocked>
        %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %x_13 = tt.addptr %x, %offsets_5 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %x_14 = tt.load %x_13, %mask_10 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
        %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %y_15 = tt.addptr %y, %offsets_6 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %y_16 = tt.load %y_15, %mask_11 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
        %output = triton_tenstorrent.binary_compute["arith.addf"] %x_14, %y_16 : (tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>) -> tensor<1024xf32, #blocked>
        %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %1 = tt.addptr %0, %offsets_7 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        tt.store %1, %output, %mask_12 : tensor<1024x!tt.ptr<f32>, #blocked>
        tt.return
    }
}

// -----

// CHECK-LABEL: @matmul_kernel_tma__reader
// CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
// CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}

// CHECK: scf.for
// CHECK: ttc.current_block
// COM: dot loop
// CHECK: scf.for
// CHECK: %[[A:.*]] = tt.descriptor_load
// CHECK: ttg.local_store %[[A]]

// CHECK: %[[B:.*]] = tt.descriptor_load
// CHECK: ttg.local_store %[[B]]
// CHECK-NOT: tt.dot
// CHECK: tt.return

// CHECK-LABEL: @matmul_kernel_tma__compute
// CHECK-DAG: ttg.local_alloc {alloc_idx = 2 : i32}
// CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
// CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}

// CHECK: scf.for
// CHECK: ttc.current_block
// COM: dot loop
// CHECK: scf.for
// CHECK-DAG: %[[A_LOAD:.*]] = ttg.local_load
// CHECK-DAG: %[[B_LOAD:.*]] = ttg.local_load
// CHECK: %[[ACCUM:.*]] = tt.dot %[[A_LOAD]], %[[B_LOAD]]
// CHECK: scf.yield %[[ACCUM]]
// CHECK: ttg.local_store
// CHECK-NOT: tt.store
// CHECK: tt.return

// CHECK-LABEL: @matmul_kernel_tma__writer
// CHECK-DAG: ttg.local_alloc {alloc_idx = 2 : i32}

// CHECK: scf.for
// CHECK: ttc.current_block
// CHECK: %[[LOAD:.*]] = ttg.local_load
// CHECK: tt.descriptor_store {{.*}}, %[[LOAD]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_tma(%arg0: !tt.tensordesc<tensor<32x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<64x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<32x64xf16>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #tiled>
    %0 = ttc.block_end
    %1 = ttc.block_start
    %num_pid_m = arith.addi %arg15, %c31_i32 : i32
    %num_pid_m_0 = arith.divsi %num_pid_m, %c32_i32 : i32
    %num_pid_n = arith.addi %arg16, %c63_i32 : i32
    %num_pid_n_1 = arith.divsi %num_pid_n, %c64_i32 : i32
    %k_tiles = arith.addi %arg17, %c63_i32 : i32
    %k_tiles_2 = arith.divsi %k_tiles, %c64_i32 : i32
    scf.for %arg18 = %1 to %0 step %c1_i32  : i32 {
      %2 = ttc.current_block %arg18 : i32
      %group_id = arith.divsi %2, %num_pid_n_1 : i32
      %group_size_m = arith.subi %num_pid_m_0, %group_id : i32
      %group_size_m_3 = arith.minsi %group_size_m, %c1_i32 : i32
      %pid_m = arith.remsi %2, %group_size_m_3 : i32
      %pid_m_4 = arith.addi %group_id, %pid_m : i32
      %pid_n = arith.remsi %2, %num_pid_n_1 : i32
      %pid_n_5 = arith.divsi %pid_n, %group_size_m_3 : i32
      %offs_am = arith.muli %pid_m_4, %c32_i32 : i32
      %offs_bn = arith.muli %pid_n_5, %c64_i32 : i32
      %accumulator = scf.for %accumulator_6 = %c0_i32 to %k_tiles_2 step %c1_i32 iter_args(%arg20 = %cst) -> (tensor<32x64xf32, #tiled>)  : i32 {
        %offs_k = arith.muli %accumulator_6, %c64_i32 : i32
        %a = tt.descriptor_load %arg0[%offs_am, %offs_k] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked>
        %b = tt.descriptor_load %arg5[%offs_k, %offs_bn] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
        %a_7 = ttg.convert_layout %a : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
        %b_8 = ttg.convert_layout %b : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>>
        %accumulator_9 = tt.dot %a_7, %b_8, %arg20 : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> * tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>> -> tensor<32x64xf32, #tiled>
        scf.yield %accumulator_9 : tensor<32x64xf32, #tiled>
      }
      %c = arith.truncf %accumulator : tensor<32x64xf32, #tiled> to tensor<32x64xf16, #tiled>
      %3 = ttg.convert_layout %c : tensor<32x64xf16, #tiled> -> tensor<32x64xf16, #blocked>
      tt.descriptor_store %arg10[%offs_am, %offs_bn], %3 : !tt.tensordesc<tensor<32x64xf16>>, tensor<32x64xf16, #blocked>
    }
    tt.return
  }
}
