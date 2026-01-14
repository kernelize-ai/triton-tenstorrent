// RUN: triton-opt %s -split-input-file --tritontenstorrent-accelerate-matmul --tritongpu-remove-layout-conversions | FileCheck %s

// CHECK-DAG: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[TILED:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
// CHECK-DAG: #[[TILED1:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
// CHECK-DAG: #[[REG:.+]] = #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: tt.func public @mma(%[[A_PTR:.*]]: tensor<32x64x!tt.ptr<f16>, #[[BLOCKED1]]>, %[[B_PTR:.*]]: tensor<64x64x!tt.ptr<f16>, #[[BLOCKED1]]>, %[[C:.*]]: tensor<32x64xf32, #[[BLOCKED]]>, %[[D_PTR:.*]]: tensor<32x64x!tt.ptr<f16>, #[[REG]]>) {
    tt.func public @mma(%a_58: tensor<32x64x!tt.ptr<f16>, #blocked1>, %b_70: tensor<64x64x!tt.ptr<f16>, #blocked1>, %arg35: tensor<32x64xf32, #blocked>, %arg36: tensor<32x64x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked1}>>)  {
    // CHECK-DAG: %[[A:.*]] = tt.load %[[A_PTR]]
    // CHECK-DAG: %[[B:.*]] = tt.load %[[B_PTR]]
    %a_59 = tt.load %a_58 : tensor<32x64x!tt.ptr<f16>, #blocked1>
    %b_71 = tt.load %b_70 : tensor<64x64x!tt.ptr<f16>, #blocked1>
    // CHECK-DAG: %[[A_TILED:.*]] = ttg.convert_layout %[[A]] : tensor<32x64xf16, #[[BLOCKED1]]> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #[[TILED]]}>>
    // CHECK-DAG: %[[B_TILED:.*]] = ttg.convert_layout %[[B]] : tensor<64x64xf16, #[[BLOCKED1]]> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #[[TILED1]]}>>
    %a_72 = ttg.convert_layout %a_59 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_73 = ttg.convert_layout %b_71 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK-DAG: %[[C_TILED:.*]] = ttg.convert_layout %[[C]] : tensor<32x64xf32, #[[BLOCKED]]> -> tensor<32x64xf32, #[[TILED]]>
    // CHECK: %[[ACCUM:.*]] = tt.dot %[[A_TILED]], %[[B_TILED]], %[[C_TILED]] : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #[[TILED]]}>> * tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #[[TILED1]]}>> -> tensor<32x64xf32, #[[TILED]]>
    %accumulator = tt.dot %a_72, %b_73, %arg35 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
    %c = arith.truncf %accumulator : tensor<32x64xf32, #blocked> to tensor<32x64xf16, #blocked>
    // CHECK: %[[TRUNCD:.*]] = arith.truncf %[[ACCUM]]
    // CHECK: %[[CONVERTED_OUT:.*]] = ttg.convert_layout %[[TRUNCD]] : tensor<32x64xf16, #[[TILED]]> -> tensor<32x64xf16, #[[REG]]>
    // CHECK: tt.store %[[D_PTR]], %[[CONVERTED_OUT]] : tensor<32x64x!tt.ptr<f16>, #[[REG]]>
     %13 = ttg.convert_layout %c : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked1}>>
    tt.store %arg36, %13 : tensor<32x64x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked1}>>
    tt.return
    }
}
