// RUN: triton-opt %s -split-input-file -finalize-cb-transaction | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#l1 = #ttcore.memory_space<l1>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: @add_kernel__compute()
    func.func public @add_kernel__compute() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c1_i32 = arith.constant 1 : i32
        // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
        // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
        // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)
        // CHECK-DAG: ttkernel.cb_reserve_back(%[[OUTPUT]], {{.*}})

        // CHECK: ttkernel.cb_wait_front
        // CHECK: ttkernel.cb_wait_front
        %0 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
        ttkernel.cb_reserve_back(%0, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
        %y = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
        %x = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
        ttkernel.cb_wait_front(%x, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
        ttkernel.cb_wait_front(%y, %c1_i32) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, i32) -> ()
        ttkernel.init_sfpu(%x, %0) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
        ttkernel.tile_regs_acquire() : () -> ()
        ttkernel.copy_tile_init(%x) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
        ttkernel.copy_tile(%x, %c0, %c0) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index, index) -> ()
        ttkernel.copy_tile_init(%y) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
        ttkernel.copy_tile(%y, %c0, %c1) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index, index) -> ()
        ttkernel.add_tiles_init(%x, %y) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
        ttkernel.add_tiles(%x, %y, %c0, %c1, %c2) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
        ttkernel.pack_tile(%c2, %0, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index) -> ()
        // CHECK-DAG: ttkernel.cb_pop_front(%[[X]], {{.*}})
        // CHECK-DAG: ttkernel.cb_pop_front(%[[Y]], {{.*}})
        // CHECK: ttkernel.cb_push_back(%[[OUTPUT]], {{.*}})
        return
    }
}
