// RUN: triton-opt %s -split-input-file --tritontenstorrent-convert-compute-ops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @unary_kernel
    tt.func public @unary_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {

        %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
        %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %x_ptrs_offset = tt.addptr %x_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: %[[X:.*]] = tt.load
        %x = tt.load %x_ptrs_offset : tensor<1024x!tt.ptr<f32>, #blocked>

        // CHECK: %[[ABS:.*]] = triton_tenstorrent.unary_compute["math.absf"] %[[X]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %abs = math.absf %x : tensor<1024xf32, #blocked>
        // CHECK: %[[CEIL:.*]] = triton_tenstorrent.unary_compute["math.ceil"] %[[ABS]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %ceil = math.ceil %abs : tensor<1024xf32, #blocked>
        // CHECK: %[[FLOOR:.*]] = triton_tenstorrent.unary_compute["math.floor"] %[[CEIL]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %floor = math.floor %ceil : tensor<1024xf32, #blocked>
        // CHECK: %[[EXP:.*]] = triton_tenstorrent.unary_compute["math.exp"] %[[FLOOR]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %exp = math.exp %floor : tensor<1024xf32, #blocked>
        // CHECK: %[[EXP2:.*]] = triton_tenstorrent.unary_compute["math.exp2"] %[[EXP]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %exp2 = math.exp2 %exp : tensor<1024xf32, #blocked>
        // CHECK: %[[LOG:.*]] = triton_tenstorrent.unary_compute["math.log"] %[[EXP2]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %log = math.log %exp2 : tensor<1024xf32, #blocked>
        // CHECK: %[[LOG2:.*]] = triton_tenstorrent.unary_compute["math.log2"] %[[LOG]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %log2 = math.log2 %log : tensor<1024xf32, #blocked>
        // CHECK: %[[SQRT:.*]] = triton_tenstorrent.unary_compute["math.sqrt"] %[[LOG2]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %sqrt = math.sqrt %log2 : tensor<1024xf32, #blocked>
        // CHECK: %[[RSQRT:.*]] = triton_tenstorrent.unary_compute["math.rsqrt"] %[[SQRT]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %rsqrt = math.rsqrt %sqrt : tensor<1024xf32, #blocked>
        // CHECK: %[[SIN:.*]] = triton_tenstorrent.unary_compute["math.sin"] %[[RSQRT]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %sin = math.sin %rsqrt : tensor<1024xf32, #blocked>
        // CHECK: %[[COS:.*]] = triton_tenstorrent.unary_compute["math.cos"] %[[SIN]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
        %cos = math.cos %sin : tensor<1024xf32, #blocked>

        %output_ptrs = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %output_ptrs_offset = tt.addptr %output_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: tt.store
        tt.store %output_ptrs_offset, %cos : tensor<1024x!tt.ptr<f32>, #blocked>
        tt.return
    }
}

// -----

// COM: When the operand is loaded via a DescriptorLoadOp, the unary compute op should
// COM: convert its operand into a tiled encoding via a layout conversion, mirroring the
// COM: behavior of RewriteBinaryComputeOp for tensor-descriptor operands.
// CHECK-DAG: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: #[[TILED:.+]] = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
    tt.func public @unary_descriptor_load(%arg10: !tt.tensordesc<tensor<32x64xf16>>, %arg15: !tt.tensordesc<tensor<32x64xf16>>, %offs_am: i32, %offs_bn: i32) {
        %x = tt.descriptor_load %arg15[%offs_am, %offs_bn] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
        // CHECK: %[[X:.*]] = tt.descriptor_load
        // CHECK: %[[X_TILED:.*]] = ttg.convert_layout %[[X]] : tensor<32x64xf16, #[[BLOCKED1]]> -> tensor<32x64xf16, #[[TILED]]>
        // CHECK: triton_tenstorrent.unary_compute["math.sqrt"] %[[X_TILED]] : (tensor<32x64xf16, #[[TILED]]>)
        %sqrt = math.sqrt %x : tensor<32x64xf16, #blocked1>
        tt.descriptor_store %arg10[%offs_am, %offs_bn], %sqrt : !tt.tensordesc<tensor<32x64xf16>>, tensor<32x64xf16, #blocked1>
        tt.return
    }
}

// -----

// COM: arith.truncf / arith.trunci narrow the element type between operand and
// COM: result, unlike the same-type math unary ops above. The wrapped
// COM: unary_compute op must carry the narrowed result type, not the operand's.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @trunc_kernel
    tt.func public @trunc_kernel(%f_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %i_ptr: !tt.ptr<i32> {tt.divisibility = 8 : i32}, %f_out_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %i_out_ptr: !tt.ptr<i16> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
        %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>

        %f_ptrs = tt.splat %f_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
        %f_ptrs_offset = tt.addptr %f_ptrs, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: %[[F:.*]] = tt.load
        %f = tt.load %f_ptrs_offset : tensor<1024x!tt.ptr<f32>, #blocked>
        // CHECK: %[[TRUNCF:.*]] = triton_tenstorrent.unary_compute["arith.truncf"] %[[F]] : (tensor<1024xf32, #blocked>) -> tensor<1024xf16, #blocked>
        %truncf = arith.truncf %f : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
        %f_out_ptrs = tt.splat %f_out_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
        %f_out_ptrs_offset = tt.addptr %f_out_ptrs, %offsets : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: tt.store
        tt.store %f_out_ptrs_offset, %truncf : tensor<1024x!tt.ptr<f16>, #blocked>

        %i_ptrs = tt.splat %i_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
        %i_ptrs_offset = tt.addptr %i_ptrs, %offsets : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: %[[I:.*]] = tt.load
        %i = tt.load %i_ptrs_offset : tensor<1024x!tt.ptr<i32>, #blocked>
        // CHECK: %[[TRUNCI:.*]] = triton_tenstorrent.unary_compute["arith.trunci"] %[[I]] : (tensor<1024xi32, #blocked>) -> tensor<1024xi16, #blocked>
        %trunci = arith.trunci %i : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
        %i_out_ptrs = tt.splat %i_out_ptr : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
        %i_out_ptrs_offset = tt.addptr %i_out_ptrs, %offsets : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
        // CHECK: tt.store
        tt.store %i_out_ptrs_offset, %trunci : tensor<1024x!tt.ptr<i16>, #blocked>

        tt.return
    }
}
