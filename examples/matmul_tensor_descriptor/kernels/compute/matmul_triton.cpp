// compute_kernel2
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif

// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H

namespace experimental {

// Stalls the UNPACK thread until the previous PACK cycle's write is
// committed to L1. Call this before unpacking data that was just packed
// by the previous linalg.generic to guarantee L1 read-after-write ordering.
ALWI void unpack_stall_on_pack() {
  PACK(t6_semaphore_post<>(semaphore::PACK_DONE));
  UNPACK(t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE));
  UNPACK(t6_semaphore_get<>(semaphore::PACK_DONE));
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H

void kernel_main() {
  size_t v1 = 1;
  size_t v2 = 2;
  size_t v3 = 0;
  int32_t v4 = 1;
  int32_t v5 = 0;
  int32_t v6 = 4;
  init_sfpu(get_compile_time_arg_val(2), get_compile_time_arg_val(2));
  init_sfpu(get_compile_time_arg_val(1), get_compile_time_arg_val(2));
  int32_t v7 = (int32_t) get_compile_time_arg_val(4);
  experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
  experimental::CircularBuffer cb_ctarg_7(get_compile_time_arg_val(7));
  experimental::CircularBuffer cb_ctarg_8(get_compile_time_arg_val(8));
  experimental::CircularBuffer cb_ctarg_9(get_compile_time_arg_val(9));
  mm_init(get_compile_time_arg_val(1), get_compile_time_arg_val(8), get_compile_time_arg_val(2), v5);
  for (int32_t i8 = v7; i8 < ((int32_t) get_compile_time_arg_val(3)); i8 += v4) {
    mm_init_short(get_compile_time_arg_val(1), get_compile_time_arg_val(8), v5);
    for (int32_t j9 = v5; j9 < ((int32_t) ((uint32_t) ((int32_t) get_compile_time_arg_val(5)) + (uint32_t) 63) / 64); j9 += v4) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_ctarg_9.wait_front(v6);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_ctarg_8.wait_front(v6);
      }
      cb_ctarg_7.reserve_back(v6);
      for (size_t k10 = v3; k10 < v2; k10 += v1) {
        tile_regs_acquire();
        if (i8 == (int32_t) ((uint32_t) v7 + (uint32_t) v4)) {
          PACK((llk_pack_reconfig_l1_acc(v4)));
        }
        size_t v11 = k10 * v2;
        for (size_t l12 = v3; l12 < v2; l12 += v1) {
          for (size_t m13 = v3; m13 < v2; m13 += v1) {
            size_t v14 = v11 + m13;
            size_t v15 = m13 * v2 + l12;
            matmul_tiles(get_compile_time_arg_val(1), get_compile_time_arg_val(8), v14, v15, l12);
          }
        }
        tile_regs_commit();
        {
        DeviceZoneScopedN("tile_regs_wait");
        tile_regs_wait();
        }
        for (size_t l16 = v3; l16 < v2; l16 += v1) {
          size_t v17 = v11 + l16;
          pack_tile<true>(l16, get_compile_time_arg_val(2), v17);
        }
        tile_regs_release();
      }
      cb_ctarg_7.push_back(v6);
      cb_ctarg_8.pop_front(v6);
      cb_ctarg_9.pop_front(v6);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_ctarg_7.wait_front(v6);
    }
    cb_ctarg_6.reserve_back(v6);
    experimental::unpack_stall_on_pack();
    for (size_t j18 = v3; j18 < v2; j18 += v1) {
      tile_regs_acquire();
      size_t v19 = j18 * v2;
      copy_tile_init(get_compile_time_arg_val(2));
      for (size_t k20 = v3; k20 < v2; k20 += v1) {
        size_t v21 = v19 + k20;
        copy_tile(get_compile_time_arg_val(2), v21, k20);
      }
      typecast_tile_init<static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32), static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16)>();
      for (size_t k22 = v3; k22 < v2; k22 += v1) {
        typecast_tile<static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32), static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16)>(k22);
      }
      tile_regs_commit();
      {
      DeviceZoneScopedN("tile_regs_wait");
      tile_regs_wait();
      }
      for (size_t k23 = v3; k23 < v2; k23 += v1) {
        size_t v24 = v19 + k23;
        pack_tile<true>(k23, get_compile_time_arg_val(0), v24);
      }
      tile_regs_release();
    }
    cb_ctarg_6.push_back(v6);
    cb_ctarg_7.pop_front(v6);
  }
  PACK((llk_pack_reconfig_l1_acc(v5)));
  return;
}

