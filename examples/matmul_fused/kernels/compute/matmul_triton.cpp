// matmul_kernel_fused__compute
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "llk_defs.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/activations.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/eltwise_unary/bitwise_not.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/clamp.h"
inline uint32_t float_to_bits(float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "compute_kernel_api/reduce.h"

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H

namespace experimental {

ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index,
                       uint32_t idst, const uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim, uint32_t nt_dim) {

  for (uint32_t i = 0; i < kt_dim; i++) {
    ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index,
                          idst, transpose, ct_dim, rt_dim, kt_dim);
    in0_tile_index++;
    in1_tile_index += nt_dim;
  }
}

} // namespace experimental

#endif

void kernel_main() {
  size_t v1 = 0;
  size_t v2 = 1;
  size_t v3 = 2;
  size_t v4 = 3;
  size_t v5 = 4;
  size_t v6 = 5;
  size_t v7 = 6;
  size_t v8 = 7;
  int32_t v9 = 0;
  int32_t v10 = 1;
  int32_t v11 = 2;
  int32_t v12 = 4;
  int32_t v13 = 3;
  int32_t v14 = get_common_arg_val<uint32_t>(42);
  mm_block_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(3), v9, v11, v11, v11);
  int32_t v15 = get_arg_val<uint32_t>(v2);
  int32_t v16 = get_arg_val<uint32_t>(v1);
  for (int32_t i17 = v16; i17 < v15; i17 += v10) {
    tile_regs_acquire();
    mm_block_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v11, v11, v11);
    for (int32_t j18 = v9; j18 < ((int32_t) ((uint32_t) v14 + (uint32_t) 63) / 64); j18 += v10) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v12);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v12);
      }
      experimental::matmul_block(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v9, v1, v9, v11, v11, v11, v11);
      cb_pop_front(get_compile_time_arg_val(0), v12);
      cb_pop_front(get_compile_time_arg_val(1), v12);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v12);
    }
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), v1, v5);
    copy_tile(get_compile_time_arg_val(2), v2, v6);
    copy_tile(get_compile_time_arg_val(2), v3, v7);
    copy_tile(get_compile_time_arg_val(2), v4, v8);
    cb_pop_front(get_compile_time_arg_val(2), v12);
    add_binary_tile_init();
    add_binary_tile(v1, v5, v1);
    add_binary_tile(v2, v6, v2);
    add_binary_tile(v3, v7, v3);
    add_binary_tile(v4, v8, v4);
    cb_reserve_back(get_compile_time_arg_val(3), v12);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v5, get_compile_time_arg_val(3), v5);
    pack_tile<true>(v6, get_compile_time_arg_val(3), v6);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(3), v12);
  }
  return;
}
