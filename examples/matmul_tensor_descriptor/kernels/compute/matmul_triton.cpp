// matmul_kernel_tma__compute
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
  int32_t v5 = 0;
  int32_t v6 = 1;
  int32_t v7 = 2;
  int32_t v8 = 8;
  int32_t v9 = 3;
  int32_t v10 = 4;
  int32_t v11 = 5;
  int32_t v12 = 6;
  int32_t v13 = 7;
  int32_t v14 = get_common_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v5);
  int32_t v15 = get_arg_val<uint32_t>(v2);
  int32_t v16 = get_arg_val<uint32_t>(v1);
  for (int32_t i17 = v16; i17 < v15; i17 += v6) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5);
    for (int32_t j18 = v5; j18 < ((int32_t) ((uint32_t) v14 + (uint32_t) 63) / 64); j18 += v6) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v7);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v8);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v5, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v6, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v7, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v9, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v10, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v11, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v12, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v13, v4);
      cb_pop_front(get_compile_time_arg_val(0), v7);
      cb_pop_front(get_compile_time_arg_val(1), v8);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v10);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<false>(v5, get_compile_time_arg_val(2), v5);
    pack_tile<false>(v6, get_compile_time_arg_val(2), v6);
    pack_tile<false>(v7, get_compile_time_arg_val(2), v7);
    pack_tile<false>(v9, get_compile_time_arg_val(2), v9);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v10);
  }
  return;
}
