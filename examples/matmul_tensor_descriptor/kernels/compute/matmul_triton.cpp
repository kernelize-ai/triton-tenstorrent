// matmul_kernel_tma__compute
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
  int32_t v3 = 0;
  int32_t v4 = 1;
  int32_t v5 = 16;
  int32_t v6 = 128;
  int32_t v7 = 8;
  int32_t v8 = 2;
  int32_t v9 = 3;
  int32_t v10 = 4;
  int32_t v11 = 5;
  int32_t v12 = 6;
  int32_t v13 = 7;
  int32_t v14 = get_common_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v3);
  int32_t v15 = get_arg_val<uint32_t>(v2);
  int32_t v16 = get_arg_val<uint32_t>(v1);
  for (int32_t i17 = v16; i17 < v15; i17 += v4) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v3);
    for (int32_t j18 = v3; j18 < ((int32_t) ((uint32_t) v14 + (uint32_t) 511) / 512); j18 += v4) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v5);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v6);
      }
      size_t v19;
      v19 = v1;
      for (int32_t k20 = v3; k20 < v7; k20 += v4) {
        size_t v21 = v19;
        for (int32_t l22 = v3; l22 < v5; l22 += v4) {
          int32_t v23 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) k20 * (uint32_t) v5)) + (uint32_t) l22);
          matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), l22, v23, v21);
        }
        v19 = v21 + v2;
      }
      cb_pop_front(get_compile_time_arg_val(0), v5);
      cb_pop_front(get_compile_time_arg_val(1), v6);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v7);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<false>(v3, get_compile_time_arg_val(2), v3);
    pack_tile<false>(v4, get_compile_time_arg_val(2), v4);
    pack_tile<false>(v8, get_compile_time_arg_val(2), v8);
    pack_tile<false>(v9, get_compile_time_arg_val(2), v9);
    pack_tile<false>(v10, get_compile_time_arg_val(2), v10);
    pack_tile<false>(v11, get_compile_time_arg_val(2), v11);
    pack_tile<false>(v12, get_compile_time_arg_val(2), v12);
    pack_tile<false>(v13, get_compile_time_arg_val(2), v13);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v7);
  }
  return;
}
void MAIN { kernel_main(); }
}
