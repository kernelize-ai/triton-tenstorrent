// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// TT-Metal port of ../../triton/histc.py (torch.histc): bucket a 1D float
// tensor into `bins` equal-width bins over [min, max] and return per-bin
// counts.
//
// Tensix has no scatter/atomic-add primitive comparable to Triton's
// tl.atomic_add, so this is not a literal port of the Triton kernel's
// one-pass "compute a bin index per element, atomically bump its bin"
// strategy. Instead it uses a compare-and-reduce strategy that is a natural
// fit for the hardware's tile-wide SFPU + reduce engine: for every bin,
// build a 0/1 mask tile `(x >= lo) & (x < hi)` with the unary SFPU
// comparisons, then sum the mask with the reduce engine. That is
// O(bins * n_tiles) passes over the data instead of Triton's single pass --
// the trade for hardware with no random-access scatter write. See
// kernels/compute/histc_compute.cpp for the per-bin/per-tile detail.
//
// Single core, whole input resident in L1: this is a reference kernel sized
// like the other tt_metal/programming_examples (thousands of elements), not
// a production/multi-core histogram op.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kTileElems = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;  // 32*32 = 1024
constexpr uint32_t kTileBytes = sizeof(bfloat16) * kTileElems;

// Host-side mirror of triton/histc.py's range handling: min==max==0 means
// "derive the range from the data", and a degenerate (min==max) range gets
// widened by +-0.5 so bin_width is never zero.
void resolve_range(const std::vector<float>& x, float& min_val, float& max_val) {
    if (min_val == 0.0f && max_val == 0.0f) {
        if (x.empty()) {
            min_val = 0.0f;
            max_val = 1.0f;
        } else {
            const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
            min_val = *lo;
            max_val = *hi;
        }
    }
    if (min_val == max_val) {
        min_val -= 0.5f;
        max_val += 0.5f;
    }
}

// Uploads `x`, runs the device histogram, and returns the `bins` counts.
std::vector<float> histc(const std::vector<float>& x, uint32_t bins, float min_val, float max_val) {
    std::vector<float> hist(bins, 0.0f);
    if (x.empty()) {
        return hist;
    }

    resolve_range(x, min_val, max_val);
    const float bin_width = (max_val - min_val) / static_cast<float>(bins);

    const uint32_t num_tiles = (static_cast<uint32_t>(x.size()) + kTileElems - 1) / kTileElems;

    // Pad to a whole number of tiles with a sentinel below min_val: it fails
    // "x >= lo" for every bin (lo only increases with b), so it never counts.
    std::vector<bfloat16> padded(num_tiles * kTileElems, bfloat16(min_val - 1.0f));
    for (size_t i = 0; i < x.size(); ++i) {
        padded[i] = bfloat16(x[i]);
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    distributed::DeviceLocalBufferConfig dram_config{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig src_buffer_config{.size = static_cast<size_t>(num_tiles) * kTileBytes};
    auto src_dram_buffer = distributed::MeshBuffer::create(src_buffer_config, dram_config, mesh_device.get());

    distributed::ReplicatedBufferConfig dst_buffer_config{.size = static_cast<size_t>(bins) * kTileBytes};
    auto dst_dram_buffer = distributed::MeshBuffer::create(dst_buffer_config, dram_config, mesh_device.get());

    // cb_in:     the whole (padded) input, resident for the entire run and
    //            never popped -- every bin re-scans it by tile index.
    // cb_scaler: a single all-ones tile, the SUM reduce's scaling operand.
    // cb_mask:   scratch, one bin's worth of 0/1 mask tiles at a time.
    // cb_out:    one tile per bin, streamed straight to the writer.
    auto make_cb = [&](CBIndex index, uint32_t num_cb_tiles) {
        CircularBufferConfig config =
            CircularBufferConfig(static_cast<size_t>(num_cb_tiles) * kTileBytes, {{index, DataFormat::Float16_b}})
                .set_page_size(index, kTileBytes);
        CreateCircularBuffer(program, core, config);
    };
    make_cb(CBIndex::c_0, num_tiles);   // cb_in
    make_cb(CBIndex::c_1, 1);           // cb_scaler
    make_cb(CBIndex::c_2, num_tiles);   // cb_mask
    make_cb(CBIndex::c_16, 1);          // cb_out

    KernelHandle reader_id = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_id = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle compute_id = CreateKernel(
        program,
        "kernels/compute/histc_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args =
                {
                    static_cast<uint32_t>(CBIndex::c_0),
                    static_cast<uint32_t>(CBIndex::c_1),
                    static_cast<uint32_t>(CBIndex::c_2),
                    static_cast<uint32_t>(CBIndex::c_16),
                    num_tiles,
                    bins,
                },
        });

    EnqueueWriteMeshBuffer(cq, src_dram_buffer, padded, false);

    SetRuntimeArgs(program, reader_id, core, {static_cast<uint32_t>(src_dram_buffer->address()), num_tiles});
    SetRuntimeArgs(
        program,
        compute_id,
        core,
        {std::bit_cast<uint32_t>(min_val), std::bit_cast<uint32_t>(bin_width), std::bit_cast<uint32_t>(max_val)});
    SetRuntimeArgs(program, writer_id, core, {static_cast<uint32_t>(dst_dram_buffer->address()), bins});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Each bin's count lands as a whole tile (see histc_compute.cpp); only
    // element 0 of each tile carries the real value.
    std::vector<bfloat16> result_tiles;
    distributed::EnqueueReadMeshBuffer(cq, result_tiles, dst_dram_buffer, true);
    for (uint32_t b = 0; b < bins; ++b) {
        hist[b] = static_cast<float>(result_tiles[static_cast<size_t>(b) * kTileElems]);
    }

    mesh_device->close();
    return hist;
}

std::vector<float> histc_reference(const std::vector<float>& x, uint32_t bins, float min_val, float max_val) {
    resolve_range(x, min_val, max_val);
    const float bin_width = (max_val - min_val) / static_cast<float>(bins);
    std::vector<float> hist(bins, 0.0f);
    for (float v : x) {
        if (v < min_val || v > max_val) {
            continue;
        }
        int32_t idx = static_cast<int32_t>(std::floor((v - min_val) / bin_width));
        idx = std::min<int32_t>(idx, static_cast<int32_t>(bins) - 1);
        idx = std::max<int32_t>(idx, 0);
        hist[idx] += 1.0f;
    }
    return hist;
}

}  // namespace

int main() {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(4096);
    for (auto& v : x) {
        v = dist(rng);
    }

    struct Case {
        uint32_t bins;
        float min_val;
        float max_val;
    };
    const Case cases[] = {
        {10, 0.0f, 0.0f},   // derive range from the data
        {4, 0.0f, 3.0f},
        {50, -2.0f, 2.0f},
    };

    bool all_ok = true;
    for (const auto& c : cases) {
        std::vector<float> got = histc(x, c.bins, c.min_val, c.max_val);
        std::vector<float> want = histc_reference(x, c.bins, c.min_val, c.max_val);

        // bfloat16 rounds both the input values and the bin edges, so allow
        // a little slack instead of requiring an exact count match.
        bool ok = true;
        for (uint32_t b = 0; b < c.bins; ++b) {
            if (std::abs(got[b] - want[b]) > std::max(1.0f, want[b] * 0.05f)) {
                ok = false;
            }
        }
        fmt::print(
            "bins={} min={} max={}: {} (got sum={}, want sum={})\n",
            c.bins,
            c.min_val,
            c.max_val,
            ok ? "OK" : "MISMATCH",
            std::accumulate(got.begin(), got.end(), 0.0f),
            std::accumulate(want.begin(), want.end(), 0.0f));
        all_ok = all_ok && ok;
    }

    if (!all_ok) {
        fmt::print("Error: one or more cases did not match the reference histogram.\n");
        return 1;
    }
    fmt::print("all cases match the reference histogram\n");
    return 0;
}
