// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// Compare-and-reduce histogram, one bin at a time:
//
//   for each bin b, edges [lo, hi):
//     mask_tile = (x >= lo) & (x < hi)      -- per-element, SFPU compare
//     count[b]  = sum(mask_tile)            -- reduce engine
//
// (the last bin's upper edge is inclusive, matching torch.histc.)
//
// Triton's version computes a bin index per element and does a single
// tl.atomic_add scatter into the histogram. Tensix's compute engine has no
// equivalent scatter-write, so instead every bin does its own full pass over
// the (L1-resident) input: O(bins * n_tiles) instead of O(n_tiles), trading
// passes over data that never leaves L1 for the atomic this hardware can't
// do. See ../../histc.cpp for why the whole input fits in L1 for this
// reference-scale kernel.
//
// Each bin runs in two non-overlapping phases so the kernel never nests the
// two different DST-register synchronization styles the API offers:
//   Phase A (mask): tile_regs_acquire/commit/wait/release around copy_tile +
//     unary compares + mul_binary_tile, once per input tile -- self
//     contained per tile, no state carried across iterations.
//   Phase B (reduce): acquire_dst/release_dst around a loop of reduce_tile
//     calls that accumulate into one DST slot across all of the bin's mask
//     tiles, exactly the pattern used by ttnn's own reduce_hw.cpp.
// reduce_init/reduce_uninit bracket only Phase B, since Phase A's plain
// pack_tile calls need the packer back in its default (non-reduce) state.

#include <cstdint>
#include <cstring>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "compute_kernel_api/reduce.h"

namespace {

FORCE_INLINE uint32_t float_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

}  // namespace

namespace NAMESPACE {
void MAIN {
    const uint32_t cb_in = get_compile_time_arg_val(0);
    const uint32_t cb_scaler = get_compile_time_arg_val(1);
    const uint32_t cb_mask = get_compile_time_arg_val(2);
    const uint32_t cb_out = get_compile_time_arg_val(3);
    const uint32_t num_tiles = get_compile_time_arg_val(4);
    const uint32_t bins = get_compile_time_arg_val(5);

    float min_val, bin_width, max_val;
    {
        const uint32_t min_val_bits = get_arg_val<uint32_t>(0);
        const uint32_t bin_width_bits = get_arg_val<uint32_t>(1);
        const uint32_t max_val_bits = get_arg_val<uint32_t>(2);
        std::memcpy(&min_val, &min_val_bits, sizeof(min_val));
        std::memcpy(&bin_width, &bin_width_bits, sizeof(bin_width));
        std::memcpy(&max_val, &max_val_bits, sizeof(max_val));
    }

    // The input is loaded once by the reader and never popped: every bin
    // re-addresses the same resident tiles by index.
    cb_wait_front(cb_in, num_tiles);
    cb_wait_front(cb_scaler, 1);

    init_sfpu(cb_in, cb_mask);

    for (uint32_t b = 0; b < bins; ++b) {
        const bool last_bin = (b == bins - 1);
        const float lo = min_val + static_cast<float>(b) * bin_width;
        const float hi = last_bin ? max_val : (lo + bin_width);
        const uint32_t lo_bits = float_bits(lo);
        const uint32_t hi_bits = float_bits(hi);

        // ---- Phase A: mask_t = (x_t >= lo) & (x_t <cmp> hi), all tiles ----
        cb_reserve_back(cb_mask, num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            tile_regs_acquire();

            copy_tile_init(cb_in);
            copy_tile(cb_in, t, 0);
            unary_ge_tile_init();
            unary_ge_tile(0, lo_bits);

            copy_tile_init(cb_in);
            copy_tile(cb_in, t, 1);
            if (last_bin) {
                unary_le_tile_init();
                unary_le_tile(1, hi_bits);
            } else {
                unary_lt_tile_init();
                unary_lt_tile(1, hi_bits);
            }

            mul_binary_tile_init();
            mul_binary_tile(0, 1, 2);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(2, cb_mask);
            tile_regs_release();
        }
        cb_push_back(cb_mask, num_tiles);

        // ---- Phase B: count[b] = sum over all of the bin's mask tiles ----
        cb_wait_front(cb_mask, num_tiles);
        reduce_init(cb_mask, cb_scaler, cb_out);
        constexpr uint32_t count_dst = 0;
        acquire_dst();
        for (uint32_t t = 0; t < num_tiles; ++t) {
            reduce_tile(cb_mask, cb_scaler, t, 0, count_dst);
        }
        cb_reserve_back(cb_out, 1);
        pack_tile(count_dst, cb_out);
        cb_push_back(cb_out, 1);
        release_dst();
        reduce_uninit();

        cb_pop_front(cb_mask, num_tiles);
    }
}
}  // namespace NAMESPACE
