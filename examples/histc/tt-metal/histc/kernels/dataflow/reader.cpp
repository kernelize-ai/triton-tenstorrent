// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

namespace {

// Fills a scratch tile with bfloat16(1.0) in every element -- the reduce
// engine's scaling-factor operand for a plain (unscaled) SUM reduction.
// Production kernels use the faster generate_reduce_scaler() helper
// (ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp),
// which only has to fill one row per face via a NOC zero-fill trick; a full
// fill is simpler to follow and just as correct for a single reference-scale
// tile like this one.
FORCE_INLINE void generate_ones_scaler_tile(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    constexpr uint16_t kOneBf16 = 0x3F80;  // bfloat16(1.0f) bit pattern
    constexpr uint32_t kElemsPerTile = 32 * 32;
    for (uint32_t i = 0; i < kElemsPerTile; ++i) {
        ptr[i] = kOneBf16;
    }
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in);
    const InterleavedAddrGenFast<true> src = {
        .bank_base_address = src_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // The whole (padded) input tensor is read once and stays resident in
    // cb_in for the entire kernel: the compute kernel re-scans it once per
    // histogram bin, addressing tiles by index rather than re-streaming them
    // from DRAM on every bin.
    cb_reserve_back(cb_in, num_tiles);
    uint32_t write_addr = get_write_ptr(cb_in);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_read_tile(t, src, write_addr);
        write_addr += tile_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, num_tiles);

    generate_ones_scaler_tile(cb_scaler);
}
