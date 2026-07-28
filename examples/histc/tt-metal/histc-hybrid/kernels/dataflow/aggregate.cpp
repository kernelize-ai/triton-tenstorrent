// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// Runs on every core, after ../compute/histc_compute.cpp has packed this
// core's local per-bin counts into cb_out (one bf16 tile per bin, element 0
// = count). Every core atomically adds its *whole local count* for each
// bin -- one NOC atomic-increment transaction per bin, not per element --
// into a single shared L1 histogram (cb_hist) that lives on the leftmost
// core in the row (the "aggregator", identified by comparing my_x/my_y
// against the agg_x/agg_y compile-time args, not by a separate flag).
//
// This needs two one-shot semaphores, both created on the host as plain
// CreateSemaphore() calls over the whole core range:
//   ready_sem: the aggregator zeroes cb_hist, then increments every other
//     core's *own copy* of ready_sem by 1 (a semaphore is per-core memory;
//     there's no way to "read" another core's copy, only to increment it
//     remotely and poll your own locally) -- so non-aggregator cores must
//     wait on their local copy before touching cb_hist.
//   done_sem: after contributing its bins, each non-aggregator core
//     increments the aggregator's copy of done_sem by 1; the aggregator
//     waits for its local copy to reach num_cores - 1 before it's safe to
//     write cb_hist out to DRAM.
//
// This is the least battle-tested file in this example -- cross-check
// get_semaphore()/noc_semaphore_wait()/noc_semaphore_inc() and the "a
// CircularBuffer gets the same L1 offset on every core in the range it was
// created for" assumption (used below to find cb_hist's address without a
// third handshake) against a real tree or an existing multi-core tt-metal
// example (e.g. a multicast/reduce kernel) before relying on this.

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

namespace {

// Element 0 of the tile is the real (bf16) count; reduce_tile's SUM over
// 0/1 masks always lands on a small non-negative integer, so a bit-widen
// to float32 + round recovers it exactly for any count this reference
// kernel's input sizes can produce.
FORCE_INLINE uint32_t bf16_tile_count_to_u32(uint32_t cb_out) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_out));
    const uint32_t bits = static_cast<uint32_t>(ptr[0]) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return static_cast<uint32_t>(f + 0.5f);
}

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t bins = get_compile_time_arg_val(1);
    constexpr uint32_t agg_x = get_compile_time_arg_val(2);
    constexpr uint32_t agg_y = get_compile_time_arg_val(3);
    constexpr uint32_t ready_sem = get_compile_time_arg_val(4);
    constexpr uint32_t done_sem = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);

    constexpr uint32_t cb_hist = tt::CBIndex::c_24;

    const bool is_aggregator = (my_x[0] == agg_x) && (my_y[0] == agg_y);
    // Every core in the range got cb_hist at the same L1 offset (see the
    // file header comment), so every core can find the aggregator's
    // counters just by reading its own local copy of that address.
    const uint32_t hist_addr = get_write_ptr(cb_hist);
    const uint64_t hist_noc_base = get_noc_addr(agg_x, agg_y, hist_addr);

    auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem));
    auto* done_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sem));

    if (is_aggregator) {
        auto* hist_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_addr);
        for (uint32_t b = 0; b < bins; ++b) {
            hist_ptr[b] = 0;
        }
        // Single-row layout: the other num_cores - 1 cores sit immediately
        // to the right of the aggregator (see histc_hybrid.cpp's CoreRange).
        for (uint32_t c = 0; c < num_cores - 1; ++c) {
            const uint64_t peer_ready = get_noc_addr(agg_x + 1 + c, agg_y, get_semaphore(ready_sem));
            noc_semaphore_inc(peer_ready, 1);
        }
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_wait(ready_ptr, 1);
    }

    // ---- add this core's whole per-bin count in one atomic op per bin ----
    for (uint32_t b = 0; b < bins; ++b) {
        cb_wait_front(cb_out, 1);
        const uint32_t count = bf16_tile_count_to_u32(cb_out);
        if (count > 0) {
            noc_semaphore_inc(hist_noc_base + b * sizeof(uint32_t), count);
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_atomic_barrier();

    if (is_aggregator) {
        noc_semaphore_wait(done_ptr, num_cores - 1);
        const uint32_t bytes = bins * static_cast<uint32_t>(sizeof(uint32_t));
        const InterleavedAddrGen<true> dst = {.bank_base_address = dst_addr, .page_size = bytes};
        const uint64_t dst_noc_addr = get_noc_addr(0, dst);
        noc_async_write(hist_addr, dst_noc_addr, bytes);
        noc_async_write_barrier();
    } else {
        noc_semaphore_inc(get_noc_addr(agg_x, agg_y, get_semaphore(done_sem)), 1);
    }
}
