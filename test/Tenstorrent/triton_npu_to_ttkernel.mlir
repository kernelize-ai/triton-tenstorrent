// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__compute() attributes {{{.*}}ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
  tt.func public @add_kernel__compute(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {

    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32

    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2_index:.*]] = arith.constant 2 : index

    // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x_0 = ttg.local_load %x : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %y_1 = ttg.local_load %y : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: ttkernel.cb_wait_front(%[[X]], %[[c1_i32]])

    // CHECK: ttkernel.init_sfpu(%[[X]], %[[OUTPUT]])
    // CHECK: ttkernel.tile_regs_acquire

    // CHECK: ttkernel.copy_tile_init(%[[X]])
    // CHECK: ttkernel.copy_tile(%[[X]], %[[c0_index]], %[[c0_index]])
    // CHECK: ttkernel.cb_pop_front(%[[X]], %[[c1_i32]])

    // CHECK-DAG: ttkernel.cb_wait_front(%[[Y]], %[[c1_i32]])
    // CHECK: ttkernel.copy_tile_init(%[[Y]])
    // CHECK: ttkernel.copy_tile(%[[Y]], %[[c0_index]], %[[c1_index]])
    // CHECK: ttkernel.cb_pop_front(%[[Y]], %[[c1_i32]])

    // CHECK: ttkernel.add_binary_tile_init()
    // CHECK: ttkernel.add_binary_tile(%[[c0_index]], %[[c1_index]], %[[c2_index]])
    %output = triton_tenstorrent.binary_compute["arith.addf"] %x_0, %y_1 : (tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>) -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>

    // CHECK-DAG: ttkernel.cb_reserve_back(%[[OUTPUT]], %[[c1_i32]])
    // CHECK: ttkernel.pack_tile(%[[c2_i32]], %[[OUTPUT]], %[[c0_i32]], false)
    // CHECK: ttkernel.cb_push_back(%[[OUTPUT]], %[[c1_i32]])
    ttg.local_store %output, %0 : tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__reader()
  tt.func public @add_kernel__reader(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32

    // CHECK-DAG: %[[c4096_i32:.*]] = arith.constant 4096 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32

    // CHECK-DAG: %[[true:.*]] = arith.constant true

    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c4_index:.*]] = arith.constant 4 : index

    // CHECK-DAG: %[[X_PTR:.*]] = ttkernel.get_arg_val(%[[c0_index]])
    // CHECK-DAG: %[[Y_PTR:.*]] = ttkernel.get_arg_val(%[[c1_index]])
    // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>

    // CHECK-DAG: %[[X_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[X]])
    // CHECK-DAG: %[[X_DATA_FORMAT:.*]] = ttkernel.get_dataformat(%[[X]])
    // CHECK-DAG: %[[X_ADDR:.*]] = ttkernel.get_interleaved_addr_gen_fast(%[[true]], %[[X_PTR]], %[[X_TILE_SIZE]], %[[X_DATA_FORMAT]])

    // CHECK-DAG: %[[Y_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[Y]])
    // CHECK-DAG: %[[Y_DATA_FORMAT:.*]] = ttkernel.get_dataformat(%[[Y]])
    // CHECK-DAG: %[[Y_ADDR:.*]] = ttkernel.get_interleaved_addr_gen_fast(%[[true]], %[[Y_PTR]], %[[Y_TILE_SIZE]], %[[Y_DATA_FORMAT]])
    // CHECK: %[[PID:.*]] = ttkernel.get_arg_val(%[[c4_index]])
    %pid = tt.get_program_id x : i32

    // CHECK-DAG: %[[TILE_INDEX_X_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32
    // CHECK-DAG: %[[TILE_INDEX_Y_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32
    // CHECK: %[[TILE_INDEX_X:.*]] = arith.divui %[[TILE_INDEX_X_BYTES]], %[[X_TILE_SIZE]]
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %offsets_1 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_2 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %offsets_3 = arith.addi %offsets_1, %offsets : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_4 = arith.addi %offsets_2, %offsets_0 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %x_8 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %x_9 = tt.addptr %x_8, %offsets_3 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    // CHECK: ttkernel.cb_reserve_back(%[[X]], %[[c1_i32]])
    // CHECK: %[[X_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[X]])
    // CHECK: %[[X_NOC_ADDR:.*]] = ttkernel.interleaved_addr_gen_fast.get_noc_addr(%[[X_ADDR]], %[[TILE_INDEX_X]], %[[c0_i32]], )
    %x_10 = tt.load %x_9 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>

    // CHECK: ttkernel.noc_async_read(%[[X_NOC_ADDR]], %[[X_WRITE_PTR]], %[[X_TILE_SIZE]])
    // CHECK: ttkernel.noc_async_read_barrier()
    // CHECK: ttkernel.cb_push_back(%[[X]], %[[c1_i32]])
    ttg.local_store %x_10, %x : tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y_11 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %y_12 = tt.addptr %y_11, %offsets_4 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: %[[TILE_INDEX_Y:.*]] = arith.divui %[[TILE_INDEX_Y_BYTES]], %[[Y_TILE_SIZE]]
    // CHECK: ttkernel.cb_reserve_back(%[[Y]], %[[c1_i32]])
    // CHECK: %[[Y_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[Y]])
    // CHECK: %[[Y_NOC_ADDR:.*]] = ttkernel.interleaved_addr_gen_fast.get_noc_addr(%[[Y_ADDR]], %[[TILE_INDEX_Y]], %[[c0_i32]], ) {{.*}}
    %y_13 = tt.load %y_12 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK: ttkernel.noc_async_read(%[[Y_NOC_ADDR]], %[[Y_WRITE_PTR]], %[[Y_TILE_SIZE]])
    // CHECK: ttkernel.noc_async_read_barrier()
    // CHECK: ttkernel.cb_push_back(%[[Y]], %[[c1_i32]])
    ttg.local_store %y_13, %y : tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__writer()
  tt.func public @add_kernel__writer(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32

    // CHECK-DAG: %[[c4096_i32:.*]] = arith.constant 4096 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32

    // CHECK-DAG: %[[true:.*]] = arith.constant true

    // CHECK-DAG: %[[c2_index:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c4_index:.*]] = arith.constant 4 : index

    // CHECK-DAG: %[[OUTPUT_PTR:.*]] = ttkernel.get_arg_val(%[[c2_index]])
    // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)

    // CHECK-DAG: %[[OUTPUT_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[OUTPUT]])
    // CHECK-DAG: %[[OUTPUT_DATA_FORMAT:.*]] = ttkernel.get_dataformat(%[[OUTPUT]])
    // CHECK-DAG: %[[OUTPUT_ADDR:.*]] = ttkernel.get_interleaved_addr_gen_fast(%[[true]], %[[OUTPUT_PTR]], %[[OUTPUT_TILE_SIZE]], %[[OUTPUT_DATA_FORMAT]])
    // CHECK-NOT: ttkernel.copy_tile_init
    // CHECK-NOT: ttkernel.copy_tile
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %1 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    %2 = tt.addptr %1, %offsets_1 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    // CHECK: %[[PID:.*]] = ttkernel.get_arg_val(%[[c4_index:.*]])
    // CHECK: %[[OUTPUT_TILE_INDEX_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32

    // CHECK: ttkernel.cb_wait_front(%[[OUTPUT]], %[[c1_i32]])

    // CHECK: %[[OUTPUT_TILE_INDEX:.*]] = arith.divui %[[OUTPUT_TILE_INDEX_BYTES]], %[[OUTPUT_TILE_SIZE]]
    // CHECK: %[[OUTPUT_READ_PTR:.*]] = ttkernel.get_read_ptr(%[[OUTPUT]])
    // CHECK: %[[OUTPUT_NOC_ADDR:.*]] = ttkernel.interleaved_addr_gen_fast.get_noc_addr(%[[OUTPUT_ADDR]], %[[OUTPUT_TILE_INDEX]], %[[c0_i32]], )
    // CHECK: ttkernel.noc_async_write(%[[OUTPUT_READ_PTR]], %[[OUTPUT_NOC_ADDR]], %[[OUTPUT_TILE_SIZE]])
    // CHECK: ttkernel.noc_async_write_barrier()
    %3 = ttg.local_load %0 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    tt.store %2, %3 : tensor<1024x!tt.ptr<f32>, #triton_tenstorrent.register_encoding<{index = 2, parent = #blocked}>>
    // CHECK: ttkernel.cb_pop_front(%[[OUTPUT]], %[[c1_i32]])
    // CHECK: return
    tt.return
  }
}

// -----

// COM: Multi-block kernel lowering
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @add_kernel__reader(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %0 = ttc.block_end
    %1 = ttc.block_start
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %mask = tt.splat %arg3 : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %mask_1 = tt.splat %arg3 : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    %x_2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
    %y_3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c4_index:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[c5_index:.*]] = arith.constant 5 : index
    // CHECK-DAG: %[[START:.*]] = ttkernel.get_arg_val(%[[c4_index]])
    // CHECK-DAG: %[[END:.*]] = ttkernel.get_arg_val(%[[c5_index]])
    // CHECK: scf.for %[[arg0:.*]] = %[[START]] to %[[END]] step %[[c1_i32]]
    scf.for %arg4 = %1 to %0 step %c1_i32  : i32 {
      // CHECK: %[[block_start:.*]] = arith.muli %[[arg0]]
      // CHECK-COUNT-2: ttkernel.noc_async_read(
      // CHECK-NOT: ttkernel.noc_async_read(
      %2 = ttc.current_block %arg4 : i32
      %block_start = arith.muli %2, %c1024_i32 : i32
      %offsets_4 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
      %offsets_5 = tt.splat %block_start : i32 -> tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
      %offsets_6 = arith.addi %offsets_4, %offsets : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
      %offsets_7 = arith.addi %offsets_5, %offsets_0 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
      %mask_8 = arith.cmpi slt, %offsets_6, %mask : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
      %mask_9 = arith.cmpi slt, %offsets_7, %mask_1 : tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
      %x_10 = tt.addptr %x_2, %offsets_6 : tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
      %x_11 = tt.load %x_10, %mask_8 : tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>>
      ttg.local_store %x_11, %x : tensor<1024xf16, #triton_tenstorrent.register_encoding<{index = 0, parent = #blocked}>> -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
      %y_12 = tt.addptr %y_3, %offsets_7 : tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>, tensor<1024xi32, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
      %y_13 = tt.load %y_12, %mask_9 : tensor<1024x!tt.ptr<f16>, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>>
      ttg.local_store %y_13, %y : tensor<1024xf16, #triton_tenstorrent.register_encoding<{index = 1, parent = #blocked}>> -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    }
    // CHECK: }
    // CHECK: return
    tt.return
  }
}
