// RUN: triton-opt %s --split-input-file --convert-triton-npu-to-d2m | FileCheck %s

// CHECK: #[[TTNN_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<8, 8>} {

  // CHECK: func.func @load_kernel(
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf16, #[[TTNN_LAYOUT]]>
  // CHECK-SAME: %[[ARG6:[a-zA-Z0-9_]+]]: tensor<?x?xf16, #[[TTNN_LAYOUT]]>
  // CHECK-SAME: attributes {tt.function_type = "forward_device"}
  tt.func public @load_kernel(
      %in_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}, %dummy_out_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}, %row: i32, %col: i32)
      attributes {noinline = false} {

    // CHECK: %[[CAST:.*]] = ttir.ttnn_metal_layout_cast %[[ARG0]]
    // CHECK-SAME: : tensor<?x?xf16, #[[TTNN_LAYOUT]]>
    // CHECK-SAME: -> memref<{{.*}}!ttcore.tile<32x32, f16>{{.*}}>

    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<8x8>
    // CHECK: ins(%[[CAST]]
    %a = tt.descriptor_load %in_desc[%row, %col]
        : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #tiled>
    // D2M.Generic currently requires an output
    tt.descriptor_store %dummy_out_desc[%row, %col], %a
        : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16, #tiled>
    tt.return
    // CHECK: return %[[ARG6]] : tensor<?x?xf16, #[[TTNN_LAYOUT]]>
  }
}

// -----

// tt.ptr<f32> tensor arguments (e.g. the vector-add tutorial): each pointer
// argument is paired with a dependent tt.load (INPUT) / tt.store (OUTPUT) used
// to recover the block tensor type, then converted to a dynamic tensor with a
// 1x1 single-tile ttnn layout. TODO: should we update this for the 8x8 grid?
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // The flat 1024-element tensor packs into a single 32x32 tile: <1x1> grid,
  // one tile per core.
  // CHECK: #[[L:.+]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, f32>,{{.*}}>, <interleaved>>

  // Each !tt.ptr<f32> arg becomes (tensor, i32); INPUT/OUTPUT pointers map to
  // tensors, the trailing i32s plus block start/end become additionalArgs.
  // CHECK: func.func @add_kernel(
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?x!tt.ptr<f32>, #[[L]]>,
  // CHECK-SAME: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
  // CHECK-SAME: -> tensor<?x?x!tt.ptr<f32>, #[[L]]>
  // CHECK-SAME: attributes {tt.function_type = "forward_device"}
  tt.func public @add_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %y_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: ttir.ttnn_metal_layout_cast
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1>
    // CHECK: ins(
    // CHECK: outs(
    // CHECK: additionalArgs(
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_p = tt.addptr %y, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_v = tt.load %y_p : tensor<1024x!tt.ptr<f32>, #blocked>
    %sum = triton_tenstorrent.binary_compute["arith.addf"] %x_v, %y_v : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %sum : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
