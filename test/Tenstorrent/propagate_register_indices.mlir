// RUN: triton-opt %s -split-input-file --tritontenstorrent-propagate-register-indices | FileCheck %s

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
    // CHECK: %[[X_CVT:.*]] = ttg.convert_layout %[[X_PTR_OFFSET]] : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.tile_encoding<{index = 0, parent = #blocked}>>
    // CHECK: %[[X:.*]] = tt.load %[[X_CVT]] : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.tile_encoding<{index = 0, parent = #blocked}>>
    %5 = tt.load %2 : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[Y_CVT:.*]] = ttg.convert_layout %[[Y_PTR_OFFSET]] : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.tile_encoding<{index = 1, parent = #blocked}>>
    // CHECK: %[[Y:.*]] = tt.load %[[Y_CVT]] : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.tile_encoding<{index = 1, parent = #blocked}>>
    %6 = tt.load %4 : tensor<1024x!tt.ptr<f32>, #blocked>
    // COM: Make sure the binary_compute op uses the new load ops and not an intermediate cvt
    // CHECK: %[[OUTPUT:.*]] = triton_tenstorrent.binary_compute["arith.addf"] %[[X]], %[[Y]] : {{.*}} -> tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 2, parent = #blocked}>>
    %7 = triton_tenstorrent.binary_compute["arith.addf"] %5, %6 : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COM: Make sure the store type is propagate to the binary compute op without a cvt by checking for direct use of %OUTPUT
    // CHECK: tt.store {{.*}} %[[OUTPUT]]
    tt.store %9, %7 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
