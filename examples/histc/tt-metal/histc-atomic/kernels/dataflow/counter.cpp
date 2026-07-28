// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads the (padded) input into L1, then scans it element-by-element and
// bumps each element's bin counter with a NOC atomic-increment transaction
// -- the one-pass "index + atomic_add" shape triton/histc.py uses, moved
// onto the data-movement RISC-V core since the Tensix compute engine has
// no scatter-write. See ../../README.md for the API-risk areas (this
// wasn't built against a real tree) and the tradeoffs against
// ../../../histc/kernels/compute/histc_compute.cpp's compare-and-reduce
// approach.

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

namespace {

FORCE_INLINE float bf16_to_float(uint16_t bits) {
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &widened, sizeof(f));
    return f;
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t n_elements = get_arg_val<uint32_t>(2);
    const uint32_t bins = get_arg_val<uint32_t>(3);
    const uint32_t min_val_bits = get_arg_val<uint32_t>(4);
    const uint32_t bin_width_bits = get_arg_val<uint32_t>(5);
    const uint32_t max_val_bits = get_arg_val<uint32_t>(6);

    float min_val, bin_width, max_val;
    std::memcpy(&min_val, &min_val_bits, sizeof(min_val));
    std::memcpy(&bin_width, &bin_width_bits, sizeof(bin_width));
    std::memcpy(&max_val, &max_val_bits, sizeof(max_val));

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_hist = tt::CBIndex::c_1;

    // ---- read the whole (padded) input into L1, once ----
    const uint32_t tile_size_bytes = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> src = {
        .bank_base_address = src_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    cb_reserve_back(cb_in, num_tiles);
    const uint32_t in_base_addr = get_write_ptr(cb_in);
    uint32_t in_write_addr = in_base_addr;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_read_tile(t, src, in_write_addr);
        in_write_addr += tile_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, num_tiles);

    // ---- shared L1 counters: bins uint32s, zero-initialized here ----
    cb_reserve_back(cb_hist, 1);
    const uint32_t hist_addr = get_write_ptr(cb_hist);
    auto* hist_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_addr);
    for (uint32_t b = 0; b < bins; ++b) {
        hist_ptr[b] = 0;
    }

    // Atomic-increment NOC transactions are addressed like any other NOC
    // transaction, (x, y, l1_addr) -- even a same-core "loopback" increment
    // has to name a target, and my_x[0]/my_y[0] are this core's own
    // coordinates.
    const uint64_t hist_noc_base = get_noc_addr(my_x[0], my_y[0], hist_addr);

    // ---- scalar scan: bin index per element, atomic bump of its counter ----
    auto* in_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr);
    for (uint32_t i = 0; i < n_elements; ++i) {
        const float x = bf16_to_float(in_ptr[i]);
        if (x < min_val || x > max_val) {
            continue;
        }
        int32_t idx = static_cast<int32_t>((x - min_val) / bin_width);
        idx = idx < 0 ? 0 : idx;
        idx = idx >= static_cast<int32_t>(bins) ? static_cast<int32_t>(bins) - 1 : idx;
        noc_semaphore_inc(hist_noc_base + static_cast<uint32_t>(idx) * sizeof(uint32_t), 1);
    }
    noc_async_atomic_barrier();

    cb_push_back(cb_hist, 1);
}
