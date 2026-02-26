// matmul_kernel_fused__compute
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "llk_defs.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/logical_not_noti.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/eltwise_unary/bitwise_not.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/clamp.h"
inline uint32_t float_to_bits(float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "api/compute/reduce.h"
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
  int32_t v11 = 16;
  int32_t v12 = 8;
  int32_t v13 = 9;
  int32_t v14 = 2;
  int32_t v15 = 10;
  int32_t v16 = 3;
  int32_t v17 = 11;
  int32_t v18 = 4;
  int32_t v19 = 12;
  int32_t v20 = 5;
  int32_t v21 = 13;
  int32_t v22 = 6;
  int32_t v23 = 14;
  int32_t v24 = 7;
  int32_t v25 = 15;
  int32_t v26 = get_common_arg_val<uint32_t>(42);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(3), v9);
  int32_t v27 = get_arg_val<uint32_t>(v2);
  int32_t v28 = get_arg_val<uint32_t>(v1);
  for (int32_t i29 = v28; i29 < v27; i29 += v10) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9);
    for (int32_t j30 = v9; j30 < ((int32_t) ((uint32_t) v26 + (uint32_t) 255) / 256); j30 += v10) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v11);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v11);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v9, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v12, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v12, v9, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v12, v12, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v10, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v13, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v10, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v13, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v14, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v15, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v14, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v15, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v16, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v17, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v16, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v17, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v18, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v19, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v18, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v19, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v20, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v21, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v20, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v21, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v22, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v23, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v22, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v23, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v24, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v25, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v24, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v25, v4);
      cb_pop_front(get_compile_time_arg_val(0), v11);
      cb_pop_front(get_compile_time_arg_val(1), v11);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v18);
    }
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), v1, v5);
    copy_tile(get_compile_time_arg_val(2), v2, v6);
    copy_tile(get_compile_time_arg_val(2), v3, v7);
    copy_tile(get_compile_time_arg_val(2), v4, v8);
    cb_pop_front(get_compile_time_arg_val(2), v18);
    add_binary_tile_init();
    add_binary_tile(v1, v5, v1);
    add_binary_tile(v2, v6, v2);
    add_binary_tile(v3, v7, v3);
    add_binary_tile(v4, v8, v4);
    cb_reserve_back(get_compile_time_arg_val(3), v18);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v9, get_compile_time_arg_val(3), v9);
    pack_tile<true>(v10, get_compile_time_arg_val(3), v10);
    pack_tile<true>(v14, get_compile_time_arg_val(3), v14);
    pack_tile<true>(v16, get_compile_time_arg_val(3), v16);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(3), v18);
  }
  return;
}
