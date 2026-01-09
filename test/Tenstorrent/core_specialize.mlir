// RUN: triton-opt %s -split-input-file --core-specialize -symbol-dce -cse | FileCheck %s

// COM: TODO: add layout checks
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @add_kernel__reader
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
    // CHECK: ttg.local_store
    // CHECK: ttg.local_store
    // CHECK-NOT: triton_tenstorrent.binary_compute

    // CHECK: @add_kernel__compute
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 0 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 1 : i32}
    // CHECK-DAG: ttg.local_alloc {alloc_idx = 2 : i32}
    // CHECK: ttg.local_load
    // CHECK: ttg.local_load
    // CHECK: triton_tenstorrent.binary_compute
    // CHECK: ttg.local_store

    // CHECK: @add_kernel__writer
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
