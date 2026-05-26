// RUN: triton-opt %s --convert-triton-npu-to-d2m | FileCheck %s

// Check the lowerings for `tt.reduce` to Tenstorrent's `d2m.tile_reduce_*`.

#tiled_1x64 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 64], order = [1, 0], tileShape = [32, 32]}>
#tiled_2x2  = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2],  order = [1, 0], tileShape = [32, 32]}>
#tiled_4x4  = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [4, 4],  order = [1, 0], tileShape = [32, 32]}>
#tiled_1d   = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1],     order = [0],    tileShape = [32, 32]}>
// CHECK: #l1 = #ttcore.memory_space<l1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {

  // -----------------------------------------------------------------------------------------------
  // Reduce a 1x64 tile grid to a single tile by reducing rows (axis=1, R dim).
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_sum_f16
  tt.func public @reduce_sum_f16(
      %in_desc: !tt.tensordesc<tensor<32x2048xf16>>, %row: i32, %col: i32)
      attributes {noinline = false} {
    // CHECK-DAG: %[[SRC:.*]] = memref.alloc() : memref<1x64x!ttcore.tile<32x32, f16>
    // CHECK-DAG: %[[DST:.*]] = memref.alloc() : memref<1x!ttcore.tile<32x32, f16>
    %in = tt.descriptor_load %in_desc[%row, %col] : !tt.tensordesc<tensor<32x2048xf16>> -> tensor<32x2048xf16, #tiled_1x64>
    %red = "tt.reduce"(%in) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f16, %rhs: f16):
      %s = arith.addf %lhs, %rhs : f16
      tt.reduce.return %s : f16
    }) : (tensor<32x2048xf16, #tiled_1x64>) -> tensor<32xf16, #ttg.slice<{dim = 1, parent = #tiled_1x64}>>
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "reduction"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_sum
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
    // CHECK: linalg.yield
    tt.return
  }

  // -----------------------------------------------------------------------------------------------
  // Reduce a 2x2 tile grid to two tiles by reducing rows (axis=1, R dim); we expect the values to
  // be laid out down column 0 of both output tiles.
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_sum_f32
  tt.func public @reduce_sum_f32(
      %in_desc: !tt.tensordesc<tensor<64x64xf32>>, %row: i32, %col: i32)
      attributes {noinline = false} {
    // CHECK-DAG: %[[SRC:.*]] = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>
    // CHECK-DAG: %[[DST:.*]] = memref.alloc() : memref<2x!ttcore.tile<32x32, f32>
    %in = tt.descriptor_load %in_desc[%row, %col] : !tt.tensordesc<tensor<64x64xf32>> -> tensor<64x64xf32, #tiled_2x2>
    %red = "tt.reduce"(%in) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %s = arith.addf %lhs, %rhs : f32
      tt.reduce.return %s : f32
    }) : (tensor<64x64xf32, #tiled_2x2>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #tiled_2x2}>>
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "reduction"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_sum
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
    // CHECK: linalg.yield
    tt.return
  }

  // -----------------------------------------------------------------------------------------------
  // Reduce a 4x4 tile grid to 4 tiles by reducing columns (axis=0, C dim); we expect the values to
  // be laid out across row 0 of the output tiles.
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_max_f32
  tt.func public @reduce_max_f32(
      %in_desc: !tt.tensordesc<tensor<128x128xf32>>, %row: i32, %col: i32)
      attributes {noinline = false} {
    // CHECK-DAG: %[[SRC:.*]] = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>
    // CHECK-DAG: %[[DST:.*]] = memref.alloc() : memref<4x!ttcore.tile<32x32, f32>
    %in = tt.descriptor_load %in_desc[%row, %col] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32, #tiled_4x4>
    %red = "tt.reduce"(%in) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %m = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %m : f32
    }) : (tensor<128x128xf32, #tiled_4x4>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #tiled_4x4}>>
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["reduction", "parallel"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_max
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim C>
    // CHECK: linalg.yield
    tt.return
  }

  // -----------------------------------------------------------------------------------------------
  // Reduce a 1D tensor (that fits in a single tile) to a single cell in that tile (axis=0, RC dim).
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_sum_bf16_1d
  tt.func public @reduce_sum_bf16_1d(
      %in_desc: !tt.tensordesc<tensor<1024xbf16>>, %row: i32)
      attributes {noinline = false} {
    // CHECK: %[[SRC:.*]] = memref.alloc() : memref<1x!ttcore.tile<32x32, bf16>
    // CHECK: %[[DST:.*]] = memref.alloc() : memref<1x!ttcore.tile<32x32, bf16>
    %in = tt.descriptor_load %in_desc[%row] : !tt.tensordesc<tensor<1024xbf16>> -> tensor<1024xbf16, #tiled_1d>
    %red = "tt.reduce"(%in) <{axis = 0 : i32}> ({
    ^bb0(%lhs: bf16, %rhs: bf16):
      %s = arith.addf %lhs, %rhs : bf16
      tt.reduce.return %s : bf16
    }) : (tensor<1024xbf16, #tiled_1d>) -> bf16
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_sum
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim RC>
    // CHECK: linalg.yield
    tt.return
  }

}
