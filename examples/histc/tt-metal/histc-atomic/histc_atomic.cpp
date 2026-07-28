// SPDX-FileCopyrightText: © 2026 Kernelize
//
// SPDX-License-Identifier: Apache-2.0
//
// TT-Metal port of ../../triton/histc.py (torch.histc), atomic-scatter
// variant: unlike ../histc (compare-and-reduce on the Tensix compute
// engine), this version follows Triton's actual algorithm shape -- compute
// a bin index per element and atomically bump that bin's counter -- by
// moving the whole histogram off the compute engine and onto the
// data-movement (RISC-V) cores, which have L1 memory-mapped directly and
// can issue NOC atomic-increment transactions against any properly-aligned
// L1 word, not just addresses obtained from CreateSemaphore(). See
// kernels/dataflow/counter.cpp for the mechanism and README.md for the
// tradeoffs against ../histc.

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

    const uint32_t n_elements = static_cast<uint32_t>(x.size());
    const uint32_t num_tiles = (n_elements + kTileElems - 1) / kTileElems;

    // Padding only needs to fill out the last tile page for the tiled DRAM
    // read -- the counting kernel below is bounded by n_elements directly
    // (it can address any element by scalar index), so unlike ../histc,
    // padding doesn't need a sentinel value guaranteed to miss every bin.
    std::vector<bfloat16> padded(num_tiles * kTileElems, bfloat16(0.0f));
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

    const uint32_t hist_bytes = bins * static_cast<uint32_t>(sizeof(uint32_t));
    distributed::DeviceLocalBufferConfig hist_dram_config{.page_size = hist_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig dst_buffer_config{.size = hist_bytes};
    auto dst_dram_buffer = distributed::MeshBuffer::create(dst_buffer_config, hist_dram_config, mesh_device.get());

    // cb_in:   the whole (padded) input, read once and left resident; the
    //          counter kernel addresses it as a flat bf16 array by scalar
    //          index, not through any tile-shaped op, so tile internal
    //          layout is moot -- same "a page is just a contiguous chunk"
    //          assumption ../histc's reader.cpp already relies on.
    // cb_hist: one page, bins uint32 counters -- a plain circular buffer,
    //          not a semaphore, but the atomic-increment NOC command works
    //          on any 4-byte-aligned L1 word regardless of how it was
    //          allocated. Shared between the counter kernel (writes, via
    //          atomic increment) and the writer kernel (reads, once
    //          counting is done).
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(static_cast<size_t>(num_tiles) * kTileBytes, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, kTileBytes);
    CreateCircularBuffer(program, core, cb_in_config);

    CircularBufferConfig cb_hist_config =
        CircularBufferConfig(hist_bytes, {{CBIndex::c_1, DataFormat::UInt32}})
            .set_page_size(CBIndex::c_1, hist_bytes);
    CreateCircularBuffer(program, core, cb_hist_config);

    KernelHandle counter_id = CreateKernel(
        program,
        "kernels/dataflow/counter.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_id = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // No compute (Tensix) kernel: the whole histogram now runs on the two
    // data-movement RISC-V cores, so the PE/SFPU sits idle for this one.

    EnqueueWriteMeshBuffer(cq, src_dram_buffer, padded, false);

    SetRuntimeArgs(
        program,
        counter_id,
        core,
        {
            static_cast<uint32_t>(src_dram_buffer->address()),
            num_tiles,
            n_elements,
            bins,
            std::bit_cast<uint32_t>(min_val),
            std::bit_cast<uint32_t>(bin_width),
            std::bit_cast<uint32_t>(max_val),
        });
    SetRuntimeArgs(program, writer_id, core, {static_cast<uint32_t>(dst_dram_buffer->address()), bins});

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
