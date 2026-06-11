// RUN: triton-opt %s -split-input-file --tritontenstorrent-tag-ios | FileCheck %s

// Arguments consumed by a descriptor_load are tagged as inputs, and arguments
// written by a descriptor_store are tagged as outputs. Untyped scalar arguments
// are left untouched.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @tag_tensordesc(
  // CHECK-SAME: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}
  // CHECK-SAME: !tt.tensordesc<tensor<64x64xf16>> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}
  // CHECK-NOT: %{{.*}}: i32 {triton_tenstorrent.io_type
  tt.func public @tag_tensordesc(%in: !tt.tensordesc<tensor<64x64xf16>>, %out: !tt.tensordesc<tensor<64x64xf16>>, %off: i32) {
    %0 = tt.descriptor_load %in[%off, %off] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    tt.descriptor_store %out[%off, %off], %0 : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16>
    tt.return
  }
}

// -----

// test tt.ptr loads
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @tag_scalar_ptr(
  // CHECK-SAME: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}
  // CHECK-SAME: !tt.ptr<f32> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}
  // CHECK-NOT: %{{.*}}: i32 {triton_tenstorrent.io_type
  tt.func public @tag_scalar_ptr(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %in_ptrs = tt.addptr %in, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %v = tt.load %in_ptrs : tensor<1024x!tt.ptr<f32>, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.addptr %out, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %out_ptrs, %v : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
