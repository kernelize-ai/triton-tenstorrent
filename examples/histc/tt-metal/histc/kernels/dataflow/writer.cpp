// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t bins = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // One tile per bin: element 0 of tile b holds histc's count for bin b
    // (see kernels/compute/histc_compute.cpp -- reduce_tile packs a whole
    // tile per bin, not a dense array of scalars).
    for (uint32_t b = 0; b < bins; ++b) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(b, dst, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
