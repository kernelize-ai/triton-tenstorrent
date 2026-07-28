// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// TT-Metal port of ../../triton/histc.py, hybrid variant: each core runs
// ../histc's compare-and-reduce (tile-wide SFPU compares + the reduce
// engine) over its own shard of the input -- the part the compute engine is
// actually good at -- then every core atomically adds its *whole per-bin
// count* (not one increment per element) into a single shared L1 histogram
// living on one "aggregator" core, via noc_semaphore_inc. That's
// num_cores * bins atomic transactions total, vs. ../histc-atomic's one per
// *element* -- the bulk data-parallel work stays on the SIMD compute
// engine, and the NOC atomic only pays for cheap cross-core aggregation.
// See README.md for the ready/done semaphore handshake this needs and its
// API risk areas.
//
// Fixed single-row core layout, aggregator = the leftmost core: this is a
// reference kernel, not a general multi-core histogram op -- see
// README.md's "known simplifications".

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
constexpr uint32_t kNumCores = 4;

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

    // Pad to a whole number of tiles *and* a multiple of kNumCores, so every
    // core gets exactly the same tile count (one compiled compute kernel,
    // one compile-time arg, no remainder handling). Same out-of-range
    // sentinel ../histc uses: it fails "x >= lo" for every bin, so padding
    // tiles never contribute to any core's count.
    const uint32_t elem_tiles = (static_cast<uint32_t>(x.size()) + kTileElems - 1) / kTileElems;
    const uint32_t num_tiles = ((elem_tiles + kNumCores - 1) / kNumCores) * kNumCores;
    const uint32_t tiles_per_core = num_tiles / kNumCores;

    std::vector<bfloat16> padded(num_tiles * kTileElems, bfloat16(min_val - 1.0f));
    for (size_t i = 0; i < x.size(); ++i) {
        padded[i] = bfloat16(x[i]);
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    constexpr CoreCoord agg_core = {0, 0};
    const CoreRange core_range(CoreCoord{0, 0}, CoreCoord{kNumCores - 1, 0});

    distributed::DeviceLocalBufferConfig dram_config{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig src_buffer_config{.size = static_cast<size_t>(num_tiles) * kTileBytes};
    auto src_dram_buffer = distributed::MeshBuffer::create(src_buffer_config, dram_config, mesh_device.get());

    const uint32_t hist_bytes = bins * static_cast<uint32_t>(sizeof(uint32_t));
    distributed::DeviceLocalBufferConfig hist_dram_config{.page_size = hist_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig dst_buffer_config{.size = hist_bytes};
    auto dst_dram_buffer = distributed::MeshBuffer::create(dst_buffer_config, hist_dram_config, mesh_device.get());

    // cb_in/cb_scaler/cb_mask/cb_out: same roles as ../histc, just sized
    // per-core (tiles_per_core instead of the whole tensor) and created
    // across the whole core_range so every core gets its own copy at the
    // same L1 offset.
    // cb_hist: bins uint32 counters. Only the aggregator core ever
    // meaningfully uses it, but it's created across the whole core_range
    // too -- CreateCircularBuffer assigns the same L1 offset to a CB index
    // on every core in the range it's created for, which every core relies
    // on below to compute the aggregator's address without a separate
    // handshake for it. Confirm this allocator guarantee against your tree
    // (see README.md).
    auto make_cb = [&](CBIndex index, uint32_t num_cb_tiles, DataFormat fmt, uint32_t page_bytes) {
        CircularBufferConfig config =
            CircularBufferConfig(static_cast<size_t>(num_cb_tiles) * page_bytes, {{index, fmt}})
                .set_page_size(index, page_bytes);
        CreateCircularBuffer(program, core_range, config);
    };
    make_cb(CBIndex::c_0, tiles_per_core, DataFormat::Float16_b, kTileBytes);   // cb_in
    make_cb(CBIndex::c_1, 1, DataFormat::Float16_b, kTileBytes);               // cb_scaler
    make_cb(CBIndex::c_2, tiles_per_core, DataFormat::Float16_b, kTileBytes);  // cb_mask
    make_cb(CBIndex::c_16, 1, DataFormat::Float16_b, kTileBytes);              // cb_out
    make_cb(CBIndex::c_24, 1, DataFormat::UInt32, hist_bytes);                 // cb_hist

    const uint32_t ready_sem = CreateSemaphore(program, core_range, 0);
    const uint32_t done_sem = CreateSemaphore(program, core_range, 0);

    KernelHandle reader_id = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle aggregate_id = CreateKernel(
        program,
        "kernels/dataflow/aggregate.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    static_cast<uint32_t>(CBIndex::c_16),  // cb_out
                    bins,
                    agg_core.x,
                    agg_core.y,
                    ready_sem,
                    done_sem,
                    kNumCores,
                },
        });

    KernelHandle compute_id = CreateKernel(
        program,
        "kernels/compute/histc_compute.cpp",
        core_range,
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
                    tiles_per_core,
                    bins,
                },
        });

    EnqueueWriteMeshBuffer(cq, src_dram_buffer, padded, false);

    // Same range args for every core.
    SetRuntimeArgs(
        program,
        compute_id,
        core_range,
        {std::bit_cast<uint32_t>(min_val), std::bit_cast<uint32_t>(bin_width), std::bit_cast<uint32_t>(max_val)});
    SetRuntimeArgs(program, aggregate_id, core_range, {static_cast<uint32_t>(dst_dram_buffer->address())});

    // Per-core tile offset into the shared src buffer.
    for (uint32_t c = 0; c < kNumCores; ++c) {
        SetRuntimeArgs(
            program,
            reader_id,
            CoreCoord{c, 0},
            {static_cast<uint32_t>(src_dram_buffer->address()), c * tiles_per_core, tiles_per_core});
    }

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> counts;
    distributed::EnqueueReadMeshBuffer(cq, counts, dst_dram_buffer, true);
    for (uint32_t b = 0; b < bins; ++b) {
        hist[b] = static_cast<float>(counts[b]);
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
