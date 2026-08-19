// RUN: triton-opt %s --split-input-file --convert-triton-npu-to-ttnn-generic | FileCheck %s

// The ttnn.generic path rewrites the triton kernel func straight into a
// func.func holding a single ttnn.generic -- no d2m.generic, and no
// ttir.ttnn_metal_layout_cast, because the generic's io tensors are the
// converted function arguments themselves. Kernel bodies are not generated:
// each of the three kernel descriptors carries an inline `source` string that
// macro-selects a section of an out-of-band <kernelName>.h header (see
// ttnn_generic_kernel_contract.md).

// CHECK: #[[L:.+]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, f32>,{{.*}}>, <interleaved>>
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // The signature conversion is shared with --convert-triton-npu-to-d2m
  // (ArgConversionHelper), so both INPUT and OUTPUT pointers become f32
  // tensors and the output is also returned.
  // CHECK: func.func @add_kernel(
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
  // CHECK-SAME: -> tensor<?x?xf32, #[[L]]>
  tt.func public @add_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %y_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {

    // The two inputs and the single output are the generic's io tensors; the
    // scalars (%n_elements plus the three SPMD args, minus the SPMD arg that
    // is not materialized here) follow as additional_args.
    // CHECK: "ttnn.generic"
    // CHECK-SAME: operandSegmentSizes = array<i32: 3, 3>

    // A single kernel name drives all three descriptors, differing only in
    // the macro that selects the reader/writer/compute body.
    // CHECK-SAME: #ttnn.source_read_kernel<source = "#define READER_KERNEL{{[^"]*}}add_kernel.h
    // The 1x1 grid gives a one-core rectangle.
    // CHECK-SAME: core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>

    // ct_args are the CB buffer indices (one per io tensor) followed by one
    // TensorAccessorArgs marker per io tensor.
    // CHECK-SAME: ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>, #ttnn.kernel_arg_tensor_accessor_args<0>, #ttnn.kernel_arg_tensor_accessor_args<1>, #ttnn.kernel_arg_tensor_accessor_args<2>]

    // common_rt_args index into the runtime's flat io_tensors-then-
    // additional_args list, so the scalars start at numIoTensors (3).
    // CHECK-SAME: common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>, #ttnn.kernel_arg_address_of_tensor<1>, #ttnn.kernel_arg_address_of_tensor<2>, #ttnn.kernel_arg_scalar<3>, #ttnn.kernel_arg_scalar<4>, #ttnn.kernel_arg_scalar<5>]
    // Per-core rt_args (tile_start/tile_end) are not emitted yet.
    // CHECK-SAME: rt_args = []

    // CHECK-SAME: #ttnn.source_write_kernel<source = "#define WRITER_KERNEL{{[^"]*}}add_kernel.h
    // CHECK-SAME: #ttnn.source_compute_kernel<source = "#define COMPUTE_KERNEL{{[^"]*}}add_kernel.h
    // CHECK-SAME: math_fidelity = hifi4

    // One double-buffered CB per io tensor: a 32x32 f32 tile is 4096 bytes and
    // each tensor is one tile, so 2 stages x 1 tile x 4096 = 8192 bytes.
    // CHECK-SAME: cbs = [<total_size = 8192, {{.*}}formats = [<buffer_index = 0, dtype = f32, page_size = 4096>]>, <total_size = 8192, {{.*}}formats = [<buffer_index = 1, dtype = f32, page_size = 4096>]>, <total_size = 8192, {{.*}}formats = [<buffer_index = 2, dtype = f32, page_size = 4096>]>]
    // CHECK-SAME: semaphores = []

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
    // The kernel body is dropped -- the generic is the whole function.
    // CHECK-NOT: d2m.generic
    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    // CHECK: return
  }
}

// -----

// Tensor descriptor arguments, a non-square grid, and an explicit
// tt.num_stages: each descriptor expands to (tensor, shape/stride/padding
// scalars...), which shifts every kernel_arg_scalar index, and the CB size
// scales with both the stage count and the per-core tile count.
// CHECK: #[[L:.+]] = #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, f16>,{{.*}}>, <interleaved>>
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<8, 4>} {

  // CHECK: func.func @desc_kernel(
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf16, #[[L]]>
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: tensor<?x?xf16, #[[L]]>
  tt.func public @desc_kernel(
      %in_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>},
      %out_desc: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>},
      %row: i32, %col: i32)
      attributes {noinline = false, tt.num_stages = 3 : i64} {

    // The io tensors are collected inputs-first, so the output tensor moves
    // ahead of the descriptor scalars that precede it in the raw signature.
    // CHECK: "ttnn.generic"(%[[ARG0]], %[[OUT]],
    // CHECK-SAME: operandSegmentSizes = array<i32: 2, 14>

    // A #triton_tenstorrent.grid<8, 4> is rows x cols, while a core_range end
    // coord is (x, y) -- so the rectangle is (0,0) to (cols-1, rows-1).
    // CHECK-SAME: core_ranges = <[#ttnn.core_range<(0,0), (3,7)>]>

    // Two io tensors, so the expanded descriptor scalars start at index 2.
    // CHECK-SAME: common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>, #ttnn.kernel_arg_scalar<2>, {{.*}}#ttnn.kernel_arg_address_of_tensor<1>, #ttnn.kernel_arg_scalar<7>,

    // 3 stages x (2x2) tiles x 2048 bytes per 32x32 f16 tile = 24576.
    // CHECK-SAME: cbs = [<total_size = 24576, {{.*}}page_size = 2048>]>, <total_size = 24576,

    %a = tt.descriptor_load %in_desc[%row, %col] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #tiled>
    tt.descriptor_store %out_desc[%row, %col], %a : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16, #tiled>
    tt.return
    // CHECK: return %[[OUT]] : tensor<?x?xf16, #[[L]]>
  }
}
