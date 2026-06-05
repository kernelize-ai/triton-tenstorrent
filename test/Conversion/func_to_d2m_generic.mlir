// RUN: triton-opt %s --convert-triton-npu-to-d2m | FileCheck %s

// CHECK: #[[TTNN_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {

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
    // CHECK-SAME: grid = #ttcore.grid<1x1>
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
