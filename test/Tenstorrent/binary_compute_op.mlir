// RUN: triton-opt %s -split-input-file --tritontenstorrent-convert-compute-ops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @add_kernel
    tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {

        %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
        %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %x_ptrs_offset = tt.addptr %x_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %y_ptrs = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %y_ptrs_offset = tt.addptr %y_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: %[[X:.*]] = tt.load
        %x = tt.load %x_ptrs_offset : tensor<1024x!tt.ptr<f32>, #blocked>
        // CHECK: %[[Y:.*]] = tt.load
        %y = tt.load %y_ptrs_offset : tensor<1024x!tt.ptr<f32>, #blocked>
        // CHECK: triton_tenstorrent.binary_compute["arith.addf"] %[[X]], %[[Y]] : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %output = arith.addf %x, %y : tensor<1024xf32, #blocked>
        %output_ptrs = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %output_ptrs_offset = tt.addptr %output_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: tt.store
        tt.store %output_ptrs_offset, %output : tensor<1024x!tt.ptr<f32>, #blocked>
        tt.return
    }
}
