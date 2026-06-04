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

// Arguments that are tensors of pointers are handled the same way: a tt.load
// marks the pointer tensor as an input, a tt.store marks it as an output.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @tag_ptr_tensor(
  // CHECK-SAME: tensor<64x64x!tt.ptr<f16>, #blocked> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<input>}
  // CHECK-SAME: tensor<64x64x!tt.ptr<f16>, #blocked> {triton_tenstorrent.io_type = #triton_tenstorrent.io_type<output>}
  tt.func public @tag_ptr_tensor(%in: tensor<64x64x!tt.ptr<f16>, #blocked>, %out: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    %0 = tt.load %in : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.store %out, %0 : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
