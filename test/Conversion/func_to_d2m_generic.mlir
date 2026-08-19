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
  // The OUTPUT pointer's element type comes from the stored value rather than
  // from the tensor-of-pointers being stored through, so it is f32 (not
  // !tt.ptr<f32>) and shares the single #[[L]] layout with the inputs.
  // CHECK: func.func @add_kernel(
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: tensor<?x?xf32, #[[L]]>,
  // CHECK-SAME: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
  // CHECK-SAME: -> tensor<?x?xf32, #[[L]]>
  // CHECK-SAME: attributes {tt.function_type = "forward_device"}
  tt.func public @add_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %y_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: ttir.ttnn_metal_layout_cast
    // The pass hoists a memref.alloc for each operand CB plus the result CB
    // ahead of the d2m.generic op that wraps them (ComputeOpToD2M.cpp); the
    // binary_compute result alloc is the third one.
    // CHECK: memref.alloc()
    // CHECK: memref.alloc()
    // CHECK: %[[SUM:.*]] = memref.alloc()
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
    // binary_compute lowers to a linalg.generic wrapping the per-tile d2m
    // compute op, writing into the result CB allocated above.
    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[SUM]] :
    // CHECK: "d2m.tile_add"
    // CHECK: linalg.yield
    %sum = triton_tenstorrent.binary_compute["arith.addf"] %x_v, %y_v : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %sum : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// arith.maximumf / arith.minimumf binary_compute ops should lower to
// d2m.tile_maximum / d2m.tile_minimum, mirroring the arith.addf → d2m.tile_add
// case above.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // CHECK: func.func @max_kernel(
  tt.func public @max_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %y_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: memref.alloc()
    // CHECK: memref.alloc()
    // CHECK: %[[MAX:.*]] = memref.alloc()
    // CHECK: d2m.generic
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_p = tt.addptr %y, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_v = tt.load %y_p : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[MAX]] :
    // CHECK: "d2m.tile_maximum"
    // CHECK: linalg.yield
    %max = triton_tenstorrent.binary_compute["arith.maximumf"] %x_v, %y_v : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %max : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // CHECK: func.func @min_kernel(
  tt.func public @min_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %y_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: memref.alloc()
    // CHECK: memref.alloc()
    // CHECK: %[[MIN:.*]] = memref.alloc()
    // CHECK: d2m.generic
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_p = tt.addptr %y, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_v = tt.load %y_p : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[MIN]] :
    // CHECK: "d2m.tile_minimum"
    // CHECK: linalg.yield
    %min = triton_tenstorrent.binary_compute["arith.minimumf"] %x_v, %y_v : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %min : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Exercise the unary_compute opcode → d2m tile op mapping (ConvertUnaryComputeOp
// in ComputeOpToD2M.cpp) for every supported unary math op, chaining them so a
// single kernel covers all of them without repeating the load/store boilerplate.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // CHECK: func.func @unary_chain_kernel(
  tt.func public @unary_chain_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: d2m.generic
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<f32>, #blocked>

    // CHECK: "d2m.tile_abs"
    %v0 = triton_tenstorrent.unary_compute["math.absf"] %x_v : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_ceil"
    %v1 = triton_tenstorrent.unary_compute["math.ceil"] %v0 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_floor"
    %v2 = triton_tenstorrent.unary_compute["math.floor"] %v1 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_exp"
    %v3 = triton_tenstorrent.unary_compute["math.exp"] %v2 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_exp2"
    %v4 = triton_tenstorrent.unary_compute["math.exp2"] %v3 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_log"
    %v5 = triton_tenstorrent.unary_compute["math.log"] %v4 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_sqrt"
    %v6 = triton_tenstorrent.unary_compute["math.sqrt"] %v5 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_rsqrt"
    %v7 = triton_tenstorrent.unary_compute["math.rsqrt"] %v6 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_sin"
    %v8 = triton_tenstorrent.unary_compute["math.sin"] %v7 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    // CHECK: "d2m.tile_cos"
    %v9 = triton_tenstorrent.unary_compute["math.cos"] %v8 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>

    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %v9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// arith.truncf / arith.trunci unary_compute ops narrow the element type between
// operand and result (unlike the same-type math ops above), so they must lower
// to d2m.tile_typecast with the output CB allocated using the converted
// (narrower) result type rather than the operand's memref type.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // CHECK: func.func @truncf_kernel(
  tt.func public @truncf_kernel(
      %x_ptr: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<f16> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: %[[OUT:.*]] = memref.alloc() : memref<{{.*}}!ttcore.tile<32x32, f16>{{.*}}>
    // CHECK: d2m.generic
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<f32>, #blocked>

    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[OUT]] :
    // CHECK: "d2m.tile_typecast"
    // CHECK: linalg.yield
    %v0 = triton_tenstorrent.unary_compute["arith.truncf"] %x_v : (tensor<1024xf32, #blocked>) -> tensor<1024xf16, #blocked>

    %o = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %v0 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32, "tt.device-grid" = #triton_tenstorrent.grid<1, 1>} {

  // CHECK: func.func @trunci_kernel(
  tt.func public @trunci_kernel(
      %x_ptr: !tt.ptr<i32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>, tt.divisibility = 8 : i32},
      %output_ptr: !tt.ptr<i16> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>, tt.divisibility = 8 : i32},
      %n_elements: i32 {tt.divisibility = 8 : i32})
      attributes {noinline = false} {
    // CHECK: %[[OUT:.*]] = memref.alloc() : memref<{{.*}}!ttcore.tile<32x32, u16>{{.*}}>
    // CHECK: d2m.generic
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %x_p = tt.addptr %x, %offsets : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %x_v = tt.load %x_p : tensor<1024x!tt.ptr<i32>, #blocked>

    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[OUT]] :
    // CHECK: "d2m.tile_typecast"
    // CHECK: linalg.yield
    %v0 = triton_tenstorrent.unary_compute["arith.trunci"] %x_v : (tensor<1024xi32, #blocked>) -> tensor<1024xi16, #blocked>

    %o = tt.splat %output_ptr : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
    %o_p = tt.addptr %o, %offsets : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_p, %v0 : tensor<1024x!tt.ptr<i16>, #blocked>
    tt.return
  }
}
