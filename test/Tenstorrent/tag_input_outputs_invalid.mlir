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

// Same conflict, but for a tensor-of-pointers argument driven by tt.load/tt.store.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // expected-error @+1 {{is used as both an input and an output; conflicting io_type tag}}
  tt.func public @conflict_ptr_tensor(%arg0: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    %0 = tt.load %arg0 : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.store %arg0, %0 : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
