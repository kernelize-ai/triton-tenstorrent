// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// Streams the shared L1 histogram counters straight to DRAM as raw
// uint32s -- an arbitrary byte-range NOC write (noc_async_write), not the
// tile-shaped noc_async_write_tile ../../../histc uses, since these aren't
// tile-formatted data at all.

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t bins = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_hist = tt::CBIndex::c_1;
    const uint32_t bytes = bins * static_cast<uint32_t>(sizeof(uint32_t));

    cb_wait_front(cb_hist, 1);
    const uint32_t hist_addr = get_read_ptr(cb_hist);

    // Single page covering the whole (small) counters buffer -- there's
    // only one of them, so no need for InterleavedAddrGenFast's tile
    // machinery, just the generic non-tile page generator.
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = bytes,
    };
    const uint64_t dst_noc_addr = get_noc_addr(0, dst);
    noc_async_write(hist_addr, dst_noc_addr, bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_hist, 1);
}
