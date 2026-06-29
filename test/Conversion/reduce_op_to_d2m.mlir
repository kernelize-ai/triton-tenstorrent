// RUN: triton-opt %s --convert-triton-npu-to-d2m | FileCheck %s

// Check the lowerings for `tt.reduce` to Tenstorrent's `d2m.tile_reduce_*`.

#blocked    = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#tiled_2x2  = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2],  order = [1, 0], tileShape = [32, 32]}>
#tiled_4x4  = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [4, 4],  order = [1, 0], tileShape = [32, 32]}>
#tiled_2d   = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 1],     order = [0],    tileShape = [32, 32]}>
// CHECK: #l1 = #ttcore.memory_space<l1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // -----------------------------------------------------------------------------------------------
  // Reduce a 1D tensor-of-pointers (single tile worth) down to a scalar (axis=0).
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_sum_bf16
  tt.func public @reduce_sum_bf16(
      %in_ptr: !tt.ptr<bf16> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>},
      %out_ptr: !tt.ptr<bf16> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>},
      %n_elements: i32)
      attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %n = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n : tensor<1024xi32, #blocked>
    %in_base = tt.splat %in_ptr : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %in_ptrs = tt.addptr %in_base, %offsets : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    %in = tt.load %in_ptrs, %mask : tensor<1024x!tt.ptr<bf16>, #blocked>
    %red = "tt.reduce"(%in) <{axis = 0 : i32}> ({
    ^bb0(%lhs: bf16, %rhs: bf16):
      %s = arith.addf %lhs, %rhs : bf16
      tt.reduce.return %s : bf16
    }) : (tensor<1024xbf16, #blocked>) -> bf16

    // TODO: D2M Generic currently requires an output
    %o = tt.splat %out_ptr : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %o_ptrs = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_ptrs, %in, %mask : tensor<1024x!tt.ptr<bf16>, #blocked>
    // CHECK: d2m.generic
    tt.return
  }

  // -----------------------------------------------------------------------------------------------
  // Reduce a 2x2 tile grid to two tiles by reducing rows (axis=1, R dim); we expect the values to
  // be laid out down column 0 of both output tiles.
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_sum_f32
  tt.func public @reduce_sum_f32(
      %in_desc: !tt.tensordesc<tensor<64x64xf32>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}, %out_desc: !tt.tensordesc<tensor<32x32xbf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}, %row: i32, %col: i32)
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
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: linalg.yield
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "reduction"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_sum
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
    // CHECK: linalg.yield

    // TODO: D2M Generic currently requires an output
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xbf16, #tiled_2d>
    tt.descriptor_store %out_desc[%row, %col], %cst : !tt.tensordesc<tensor<32x32xbf16>>, tensor<32x32xbf16, #tiled_2d>
    tt.return
  }

  // -----------------------------------------------------------------------------------------------
  // Reduce a 4x4 tile grid to 4 tiles by reducing columns (axis=0, C dim); we expect the values to
  // be laid out across row 0 of the output tiles.
  // -----------------------------------------------------------------------------------------------

  // CHECK-LABEL: func.func @reduce_max_f32
  tt.func public @reduce_max_f32(
      %in_desc: !tt.tensordesc<tensor<128x128xf32>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}, %out_desc: !tt.tensordesc<tensor<32x32xbf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}, %row: i32, %col: i32)
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
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: linalg.yield
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["reduction", "parallel"]
    // CHECK-SAME: ins(%[[SRC]]
    // CHECK-SAME: outs(%[[DST]]
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_reduce_max
    // CHECK-SAME: reduce_dim = #d2m<reduce_dim C>
    // CHECK: linalg.yield

    // TODO: D2M Generic currently requires an output
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xbf16, #tiled_2d>
    tt.descriptor_store %out_desc[%row, %col], %cst : !tt.tensordesc<tensor<32x32xbf16>>, tensor<32x32xbf16, #tiled_2d>
    tt.return
  }

}
