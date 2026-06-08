// RUN: triton-opt %s -split-input-file --tritontenstorrent-tag-ios -verify-diagnostics

// An argument that is both loaded from and stored to cannot be tagged as both
// an input and an output: the pass must reject it.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // expected-error @+1 {{is used as both an input and an output; conflicting io_type tag}}
  tt.func public @conflict_tensordesc(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %off: i32) {
    %0 = tt.descriptor_load %arg0[%off, %off] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    tt.descriptor_store %arg0[%off, %off], %0 : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16>
    tt.return
  }
}

// -----

// Same conflict, but for a scalar !tt.ptr argument splatted/offset into a tensor
// of pointers and driven by tt.load/tt.store.
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // expected-error @+1 {{is used as both an input and an output; conflicting io_type tag}}
  tt.func public @conflict_scalar_ptr(%ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %t = tt.splat %ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %t, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %v = tt.load %ptrs : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %v : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
