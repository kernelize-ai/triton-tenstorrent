// RUN: triton-opt %s --convert-triton-npu-to-d2m | FileCheck %s

// Full matmul pipeline: covers arith.constant, tt.descriptor_load, tt.dot,
// arith.truncf, and tt.descriptor_store in one realistic function so that all
// intermediate tensors are live (no dead-code issues with the conversion pass).

#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
#dot_a = #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>
#dot_b = #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled}>

// CHECK: #l1 = #ttcore.memory_space<l1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<8, 8>} {
  // CHECK-LABEL: func.func @matmul

  // The !tt.tensordesc args are lowered to ttnn-layout tensors, then cast to
  // interleaved DRAM memrefs (one per input/output descriptor).
  // CHECK: %[[ACAST:.*]] = ttir.ttnn_metal_layout_cast
  // CHECK: %[[BCAST:.*]] = ttir.ttnn_metal_layout_cast
  // CHECK: %[[CCAST:.*]] = ttir.ttnn_metal_layout_cast

  // All CB allocs hoisted above the generic: f32 accumulator + f16 A/B/typecast-out.
  // CHECK: %[[ACC:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<8192x4096, 2>, #l1>
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #ttcore.cb_layout<4096x2048, 2>, #l1>
  // CHECK: %[[B:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #ttcore.cb_layout<4096x2048, 2>, #l1>
  // CHECK: %[[COUT:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #ttcore.cb_layout<4096x2048, 2>, #l1>

  // Whole body wrapped in a single d2m.generic over the 8x8 device grid.
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>

  tt.func public @matmul(
      %a_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}, %a_row: i32, %a_col: i32,
      %b_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}, %b_row: i32, %b_col: i32,
      %c_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}, %c_row: i32, %c_col: i32)
      attributes {noinline = false} {

    // arith.constant → the f32 accumulator alloc (hoisted, checked above as %[[ACC]]).
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #tiled>

    // tt.descriptor_load → flat tile-index linearization
    // (flat = (row/32) * ceildiv(cols,32) + (col/32)) then d2m.remote_load.
    // CHECK: arith.ceildivsi
    // CHECK: arith.divsi
    // CHECK: arith.divsi
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: d2m.remote_load %[[A]] %[[ACAST]]
    %a = tt.descriptor_load %a_desc[%a_row, %a_col] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #dot_a>

    // CHECK: arith.ceildivsi
    // CHECK: arith.divsi
    // CHECK: arith.divsi
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: d2m.remote_load %[[B]] %[[BCAST]]
    %b = tt.descriptor_load %b_desc[%b_row, %b_col] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #dot_b>

    // tt.dot → linalg.generic over (M, N, K) with K=reduction, using d2m.tile_matmul
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: ins(%[[A]], %[[B]] :
    // CHECK-SAME: outs(%[[ACC]] :
    // CHECK: "d2m.tile_matmul"
    // CHECK: linalg.yield
    %result = tt.dot %a, %b, %acc : tensor<64x64xf16, #dot_a> * tensor<64x64xf16, #dot_b> -> tensor<64x64xf32, #tiled>

    // arith.truncf → linalg.generic with d2m.tile_typecast (f32 acc → f16 out)
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK-SAME: ins(%[[ACC]] :
    // CHECK-SAME: outs(%[[COUT]] :
    // CHECK: "d2m.tile_typecast"(%{{.*}}) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
    // CHECK-NEXT: linalg.yield
    %c = arith.truncf %result : tensor<64x64xf32, #tiled> to tensor<64x64xf16, #tiled>

    // tt.descriptor_store → flat tile-index linearization + d2m.remote_store
    // CHECK: arith.ceildivsi
    // CHECK: arith.divsi
    // CHECK: arith.divsi
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: d2m.remote_store %[[CCAST]]{{.*}}%[[COUT]]
    tt.descriptor_store %c_desc[%c_row, %c_col], %c : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16, #tiled>
    tt.return
  }
}
