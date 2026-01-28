// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel | FileCheck %s

// COM: TODO: figure out why the matmul loop is being optimized away with canonicalize on

#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 64]}>
#shared1 = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
// CHECK: @matmul_kernel_fused__compute
tt.func public @matmul_kernel_fused__compute(%arg23: i32, %k_tiles_0: i32) {
    %0 = ttg.local_alloc {alloc_idx = 3 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>

    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    %bias = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>

    %cst = arith.constant {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32} dense<0.000000e+00> : tensor<32x64xf32, #tiled>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 0 : i32
    // CHECK: ttkernel.mm_init
    // CHECK: ttkernel.mm_init_short

    // CHECK: scf.for
    // COM: now check for M,N, and K loops
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: ttkernel.matmul_tiles
    %accumulator = scf.for %accumulator_3 = %c0_i32 to %k_tiles_0 step %c1_i32 iter_args(%arg25 = %cst) -> (tensor<32x64xf32, #tiled>)  : i32 {
        %a_4 = ttg.local_load %a : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
        %b_5 = ttg.local_load %b : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>>
        %accumulator_6 = tt.dot %a_4, %b_5, %arg25 {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32} : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> * tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>> -> tensor<32x64xf32, #tiled>
        scf.yield %accumulator_6 : tensor<32x64xf32, #tiled>
    } {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32}
    // CHECK-COUNT-3: scf.yield
    // CHECK: ttkernel.copy_tile_init
    // CHECK-COUNT-2: ttkernel.copy_tile
    %c = arith.truncf %accumulator {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32} : tensor<32x64xf32, #tiled> to tensor<32x64xf16, #tiled>
    %bias_1 = ttg.local_load %bias {triton_tenstorrent.alloc_offset = 2 : i32, triton_tenstorrent.alloc_size = 2 : i32} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #tiled>
    // CHECK: ttkernel.add_binary_tile_init
    // CHECK-COUNT-2: ttkernel.add_binary_tile
    %c_2 = triton_tenstorrent.binary_compute["arith.addf"] %c, %bias_1 {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 2 : i32} : (tensor<32x64xf16, #tiled>, tensor<32x64xf16, #tiled>) -> tensor<32x64xf16, #tiled>
    // CHECK: ttkernel.tile_regs_commit()
    // CHECK: ttkernel.tile_regs_wait()
    // CHECK-COUNT-2: ttkernel.pack_tile
    // CHECK: ttkernel.tile_regs_release()
    ttg.local_store %c_2, %0 : tensor<32x64xf16, #tiled> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    tt.return
}

}
