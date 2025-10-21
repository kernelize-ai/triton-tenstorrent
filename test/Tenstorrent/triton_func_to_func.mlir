// RUN: triton-opt %s -split-input-file --convert-triton-func-to-func | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
// COM: Test that arguments are copied, attributes are not
// CHECK: func.func public @add_kernel__compute(%arg0: !tt.ptr<f32>
// CHECK-NOT: {noinline = false}
tt.func public @add_kernel__compute(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x_0 = ttg.local_load %x : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    %y_1 = ttg.local_load %y : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    %output = arith.addf %x_0, %y_1 : tensor<1024xf32, #blocked>
    ttg.local_store %output, %0 : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}
