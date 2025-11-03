// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: @add_kernel__compute()
  func.func public @add_kernel__compute(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
    // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK-DAG: %[[OUTPUT1D:.*]] = ttkernel.cb_reinterpret_shape(%[[OUTPUT]])
    // CHECK-DAG: %[[C1_0:.*]] = arith.constant 1 : i32
    // CHECK-DAG: ttkernel.cb_reserve_back(%[[OUTPUT]], %[[C1_0]])
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x_0 = ttg.local_load %x : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 0, parent = #blocked}>>
    %y_1 = ttg.local_load %y : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 1, parent = #blocked}>>
    // CHECK: ttkernel.cb_wait_front(%[[X]], {{.*}})
    // CHECK: %[[X1D:.*]] = ttkernel.cb_reinterpret_shape(%[[X]])

    // CHECK: ttkernel.cb_wait_front(%[[Y]], {{.*}})
    // CHECK: %[[Y1D:.*]] = ttkernel.cb_reinterpret_shape(%[[Y]])

    // CHECK: ttkernel.init_sfpu(%[[X1D]], %[[OUTPUT1D]])
    // CHECK: ttkernel.tile_regs_acquire

    // CHECK: ttkernel.copy_tile_init(%[[X1D]])
    // CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C0_1:.*]] = arith.constant 0 : index
    // CHECK: ttkernel.copy_tile(%[[X1D]], %[[C0_0]], %[[C0_1]])

    // CHECK: ttkernel.copy_tile_init(%[[Y1D]])
    // CHECK-DAG: %[[C0_2:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1_1:.*]] = arith.constant 1 : index
    // CHECK: ttkernel.copy_tile(%[[Y1D]], %[[C0_2]], %[[C1_1]])

    // CHECK: ttkernel.add_binary_tile_init()
    // CHECK-DAG: %[[C0_3:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1_1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2_0:.*]] = arith.constant 2 : index
    // CHECK: ttkernel.add_binary_tile(%[[C0_3]], %[[C1_1]], %[[C2_0]])
    %output = triton_tenstorrent.binary_compute["arith.addf"] %x_0, %y_1 : (tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 0, parent = #blocked}>>, tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 1, parent = #blocked}>>) -> tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 2, parent = #blocked}>>
    // CHECK-DAG: %[[C2_1:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[C0_4:.*]] = arith.constant 0 : index
    // CHECK: ttkernel.pack_tile(%[[C2_1]], %[[OUTPUT1D]], %[[C0_4]], true)
    ttg.local_store %output, %0 : tensor<1024xf32, #triton_tenstorrent.tile_encoding<{index = 2, parent = #blocked}>> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    return
  }
}
