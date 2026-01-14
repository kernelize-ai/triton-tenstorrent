// RUN: triton-opt %s -split-input-file --tritontenstorrent-remove-redundant-masks | FileCheck %s

// CHECK-DAG: #[[REG0:.+]] = #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>
// CHECK-DAG: #[[REG1:.+]] = #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>
// CHECK-DAG: #[[REG2:.+]] = #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %offsets_1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %offsets_2 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_3 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %offsets_4 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %offsets_5 = arith.addi %offsets_2, %offsets : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_6 = arith.addi %offsets_3, %offsets_0 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %offsets_7 = arith.addi %offsets_4, %offsets_1 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %mask_8 = tt.splat %n_elements : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %mask_9 = tt.splat %n_elements : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %mask_10 = arith.cmpi slt, %offsets_5, %mask : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %mask_11 = arith.cmpi slt, %offsets_6, %mask_8 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %mask_12 = arith.cmpi slt, %offsets_7, %mask_9 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: %[[X:.*]] = tt.addptr
    %x_13 = tt.addptr %x, %offsets_5 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: tt.load %[[X]] : tensor<1024x!tt.ptr<f32>, #[[REG0]]>
    %x_14 = tt.load %x_13, %mask_10 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: %[[Y:.*]] = tt.addptr
    %y_15 = tt.addptr %y, %offsets_6 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: tt.load %[[Y]] : tensor<1024x!tt.ptr<f32>, #[[REG1]]>
    %y_16 = tt.load %y_15, %mask_11 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: %[[OUTPUT:.*]] = triton_tenstorrent.binary_compute["arith.addf"]
    %output = triton_tenstorrent.binary_compute["arith.addf"] %x_14, %y_16 : (tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>) -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    // CHECK: %[[OUTPUT_PTR:.*]] = tt.addptr
    %1 = tt.addptr %0, %offsets_7 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    // CHECK: tt.store %[[OUTPUT_PTR]], %[[OUTPUT]] : tensor<1024x!tt.ptr<f32>, #[[REG2]]>
    tt.store %1, %output, %mask_12 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    tt.return
  }
}
