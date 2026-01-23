// RUN: triton-opt %s -split-input-file --tritontenstorrent-remove-dot-load-layout-conversions --tritongpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
// CHECK-DAG: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[TILED:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
// CHECK-DAG: #[[TILED1:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
 // CHECK: tt.func public @mma(%[[ARG0:.*]]: tensor<32x64x!tt.ptr<f16>, #[[BLOCKED]]>, %[[ARG1:.*]]: tensor<64x64x!tt.ptr<f16>, #[[BLOCKED]]>, %arg2: tensor<32x64xf32, #blocked1>, %arg3: tensor<32x64x!tt.ptr<f16>, #[[BLOCKED]]>)
  tt.func public @mma(%arg0: tensor<32x64x!tt.ptr<f16>, #blocked>, %arg1: tensor<64x64x!tt.ptr<f16>, #blocked>, %arg2: tensor<32x64xf32, #blocked1>, %arg3: tensor<32x64x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[ARG0_TILED:.*]] = ttg.convert_layout %[[ARG0]] : tensor<32x64x!tt.ptr<f16>, #[[BLOCKED]]> -> tensor<32x64x!tt.ptr<f16>, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #[[TILED]]}>>
    // CHECK: tt.load %[[ARG0_TILED]] : tensor<32x64x!tt.ptr<f16>, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #[[TILED]]}>>
    // CHECK: %[[ARG1_TILED:.*]] = ttg.convert_layout %[[ARG1]] : tensor<64x64x!tt.ptr<f16>, #[[BLOCKED]]> -> tensor<64x64x!tt.ptr<f16>, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #[[TILED1]]}>>
    // CHECK: tt.load %[[ARG1_TILED]] : tensor<64x64x!tt.ptr<f16>, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #[[TILED1]]}>>
    %0 = tt.load %arg0 : tensor<32x64x!tt.ptr<f16>, #blocked>
    %1 = tt.load %arg1 : tensor<64x64x!tt.ptr<f16>, #blocked>
    %2 = ttg.convert_layout %0 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
    %3 = ttg.convert_layout %1 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>>
    %4 = ttg.convert_layout %arg2 : tensor<32x64xf32, #blocked1> -> tensor<32x64xf32, #tiled>
    %5 = tt.dot %2, %3, %4 : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> * tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>> -> tensor<32x64xf32, #tiled>
    %6 = arith.truncf %5 : tensor<32x64xf32, #tiled> to tensor<32x64xf16, #tiled>
    %7 = ttg.convert_layout %6 : tensor<32x64xf16, #tiled> -> tensor<32x64xf16, #blocked>
    tt.store %arg3, %7 : tensor<32x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-NOT: {{.*}} = #ttg.blocked
// CHECK-DAG: #[[TILED:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
// CHECK-DAG: #[[TILED1:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
   tt.func public @matmul_kernel_tma(%arg0: !tt.tensordesc<tensor<32x64xf16>>, %arg5: !tt.tensordesc<tensor<64x64xf16>>, %offs_am:  i32, %offs_k: i32, %offs_bn: i32, %arg20: tensor<32x64xf32, #tiled>) -> tensor<32x64xf32, #tiled> {
      // CHECK-DAG: %[[A:.*]] = tt.descriptor_load %{{.*}} : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #[[TILED]]}>>
      // CHECK-DAG: %[[B:.*]] = tt.descriptor_load %{{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #[[TILED1]]}>>
      // CHECK: tt.dot %[[A]], %[[B]]
      %a = tt.descriptor_load %arg0[%offs_am, %offs_k] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked>
      %b = tt.descriptor_load %arg5[%offs_k, %offs_bn] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
      %a_7 = ttg.convert_layout %a : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
      %b_8 = ttg.convert_layout %b : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>>
      %accumulator_9 = tt.dot %a_7, %b_8, %arg20 : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> * tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>> -> tensor<32x64xf32, #tiled>
      tt.return %accumulator_9 : tensor<32x64xf32, #tiled>
   }
}
