// RUN: triton-opt %s -split-input-file --tritontenstorrent-materialize-multicasts | FileCheck %s

#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    tt.func public @matmul_kernel_tma__writer(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %offs_am: i32, %k_tiles_2: i32) {
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c64_i32 = arith.constant 64 : i32

        %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
        // CHECK: %[[A:.*]] = ttg.local_alloc

        scf.for %accumulator = %c0_i32 to %k_tiles_2 step %c1_i32  : i32 {
        %offs_k = arith.muli %accumulator, %c64_i32 : i32
        // CHECK: %[[MULTICAST_RESULT:.*]] = triton_tenstorrent.multicast %[[A]] {
        // CHECK: %[[LOAD_RESULT:.*]] = tt.descriptor_load
        // CHECK: triton_tenstorrent.yield %[[LOAD_RESULT]], %[[A]]
        // CHECK: }
        // CHECK: ttg.local_store %[[MULTICAST_RESULT]], %[[A]]
        %a_6 = tt.descriptor_load %arg0[%offs_am, %offs_k] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
        ttg.local_store %a_6, %a : tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      }
      tt.return
    }
}
