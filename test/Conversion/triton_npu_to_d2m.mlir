// RUN: triton-opt %s --convert-triton-npu-to-d2m | FileCheck %s

// Full matmul pipeline: covers arith.constant, tt.descriptor_load, tt.dot,
// arith.truncf, and tt.descriptor_store in one realistic function so that all
// intermediate tensors are live (no dead-code issues with the conversion pass).

#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
#dot_a = #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>
#dot_b = #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled}>

// CHECK: #l1 = #ttcore.memory_space<l1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: func.func @matmul
  tt.func public @matmul(
      %a_desc: !tt.tensordesc<tensor<64x64xf16>>, %a_row: i32, %a_col: i32,
      %b_desc: !tt.tensordesc<tensor<64x64xf16>>, %b_row: i32, %b_col: i32,
      %c_desc: !tt.tensordesc<tensor<64x64xf16>>, %c_row: i32, %c_col: i32)
      attributes {noinline = false} {
    // arith.constant → memref.alloc (accumulator in L1)
    // CHECK: %[[ACC:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #tiled>

    // tt.descriptor_load → CB alloc + index_cast per dim + d2m.remote_load
    // CHECK: memref.alloc()
    // CHECK-SAME: memref<2x2x!ttcore.tile<32x32, f16>
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: %[[A:.*]] = d2m.remote_load
    %a = tt.descriptor_load %a_desc[%a_row, %a_col] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #dot_a>

    // CHECK: memref.alloc()
    // CHECK-SAME: memref<2x2x!ttcore.tile<32x32, f16>
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: %[[B:.*]] = d2m.remote_load
    %b = tt.descriptor_load %b_desc[%b_row, %b_col] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #dot_b>

    // tt.dot → linalg.generic over (M, N, K) with K=reduction, using d2m.tile_matmul
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: ins(%[[A]], %[[B]] :
    // CHECK-SAME: outs(%[[ACC]] :
    // CHECK: "d2m.tile_matmul"
    // CHECK: linalg.yield
    %result = tt.dot %a, %b, %acc : tensor<64x64xf16, #dot_a> * tensor<64x64xf16, #dot_b> -> tensor<64x64xf32, #tiled>

    // arith.truncf → alloc + linalg.generic with d2m.tile_typecast
    // CHECK: memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_typecast"(%{{.*}}) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
    // CHECK-NEXT: linalg.yield
    %c = arith.truncf %result : tensor<64x64xf32, #tiled> to tensor<64x64xf16, #tiled>

    // tt.descriptor_store → index_cast per dim + d2m.remote_store
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: d2m.remote_store
    tt.descriptor_store %c_desc[%c_row, %c_col], %c : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16, #tiled>
    tt.return
  }
}
