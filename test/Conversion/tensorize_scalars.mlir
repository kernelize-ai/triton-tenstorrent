// RUN: triton-opt %s --tensorize-scalars | FileCheck %s

// Test that TensorizeScalars converts scalar integer types (i32, i64) to
// tensor<ui32> (0D ranked tensor with unsigned 32-bit int and scalar TTNN
// layout), while leaving f32 untouched. The ttnn.generic additional_args
// should reference the new tensorized operands.

#loc = loc(unknown)

// Top-level alias avoids nested-angle-bracket parsing issues with inline attrs.
#prog = #ttnn.program<kernels = [#ttnn.data_movement_kernel<symbol_ref = @kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, processor = riscv0, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_scalar<1>, #ttnn.kernel_arg_scalar<0>], common_rt_args = [], rt_args = []>], cbs = [], semaphores = []>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: func.func @tensorize_scalars_test(
  // CHECK-SAME: tensor<ui32,
  // CHECK-SAME: tensor<ui32,
  // CHECK-SAME: f32
  // CHECK-SAME: tensor<4x4xf32>
  func.func @tensorize_scalars_test(
      %scalar_i32: i32,
      %scalar_i64: i64,
      %scalar_f32: f32,
      %tensor_arg: tensor<4x4xf32>)
      attributes {noinline = false} {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 3>
    // CHECK-SAME: #ttnn.kernel_arg_scalar<1>, #ttnn.kernel_arg_scalar<0>
    // CHECK-SAME: (tensor<4x4xf32>, tensor<ui32,{{.*}}, tensor<ui32,{{.*}}, f32)
    "ttnn.generic"(%tensor_arg, %scalar_i32, %scalar_i64, %scalar_f32) <{operandSegmentSizes = array<i32: 1, 3>, program = #prog}> : (tensor<4x4xf32>, i32, i64, f32) -> () loc(#loc)
    return
  } loc(#loc)
} loc(#loc)
