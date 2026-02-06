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
  int32_t v2 = 0;
  int32_t v3 = 1;
  int32_t v4 = 2;
  int32_t v5 = 4;
  int32_t v6 = 3;
  int32_t v7 = get_common_arg_val<uint32_t>(32);
  mm_block_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v2, v4, v4, v4);
  int32_t v8 = get_arg_val<uint32_t>(1);
  int32_t v9 = get_arg_val<uint32_t>(v1);
  for (int32_t i10 = v9; i10 < v8; i10 += v3) {
    tile_regs_acquire();
    mm_block_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v4, v4, v4);
    for (int32_t j11 = v2; j11 < ((int32_t) ((uint32_t) v7 + (uint32_t) 63) / 64); j11 += v3) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v5);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v5);
      }
      experimental::matmul_block(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v2, v1, v2, v4, v4, v4, v4);
      cb_pop_front(get_compile_time_arg_val(0), v5);
      cb_pop_front(get_compile_time_arg_val(1), v5);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v5);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v2, get_compile_time_arg_val(2), v2);
    pack_tile<true>(v3, get_compile_time_arg_val(2), v3);
    pack_tile<true>(v4, get_compile_time_arg_val(2), v4);
    pack_tile<true>(v6, get_compile_time_arg_val(2), v6);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v5);
  }
  return;
}
