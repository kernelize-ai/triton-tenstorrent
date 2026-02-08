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
namespace NAMESPACE {
void kernel_main() {
  size_t v1 = 0;
  size_t v2 = 1;
  size_t v3 = 2;
  size_t v4 = 3;
  int32_t v5 = 0;
  int32_t v6 = 1;
  int32_t v7 = 2;
  int32_t v8 = 4;
  int32_t v9 = 3;
  int32_t v10 = get_common_arg_val<uint32_t>(42);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(3), v5);
  int32_t v11 = get_arg_val<uint32_t>(v2);
  int32_t v12 = get_arg_val<uint32_t>(v1);
  for (int32_t i13 = v12; i13 < v11; i13 += v6) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5);
    for (int32_t j14 = v5; j14 < ((int32_t) ((uint32_t) v10 + (uint32_t) 63) / 64); j14 += v6) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v7);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v8);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v5, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v7, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v6, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v9, v2);
      cb_pop_front(get_compile_time_arg_val(0), v7);
      cb_pop_front(get_compile_time_arg_val(1), v8);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v7);
    }
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), v1, v3);
    copy_tile(get_compile_time_arg_val(2), v2, v4);
    cb_pop_front(get_compile_time_arg_val(2), v7);
    add_binary_tile_init();
    add_binary_tile(v1, v3, v1);
    add_binary_tile(v2, v4, v2);
    cb_reserve_back(get_compile_time_arg_val(3), v7);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v5, get_compile_time_arg_val(3), v5);
    pack_tile<true>(v6, get_compile_time_arg_val(3), v6);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(3), v7);
  }
  return;
}
void MAIN { kernel_main(); }
}
