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
  int32_t v5 = 32;
  int32_t v6 = 64;
  int32_t v7 = 2;
  int32_t v8 = 4;
  int32_t v9 = 16;
  int32_t v10 = 8;
  int32_t v11 = 3;
  int32_t v12 = 5;
  int32_t v13 = 6;
  int32_t v14 = 7;
  int32_t v15 = get_common_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v3);
  int32_t v16 = get_arg_val<uint32_t>(v2);
  int32_t v17 = get_arg_val<uint32_t>(v1);
  for (int32_t i18 = v17; i18 < v16; i18 += v4) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v3);
    for (int32_t j19 = v3; j19 < ((int32_t) ((uint32_t) v15 + (uint32_t) 511) / 512); j19 += v4) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v5);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v6);
      }
      size_t v20;
      v20 = v1;
      for (int32_t k21 = v3; k21 < v7; k21 += v4) {
        size_t v22 = v20;
        size_t v23;
        v23 = v22;
        for (int32_t l24 = v3; l24 < v8; l24 += v4) {
          size_t v25 = v23;
          for (int32_t m26 = v3; m26 < v9; m26 += v4) {
            int32_t v27 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) k21 * (uint32_t) v9)) + (uint32_t) m26);
            int32_t v28 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) l24 * (uint32_t) v9)) + (uint32_t) m26);
            matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v28, v25);
          }
          v23 = v25 + v2;
        }
        size_t v29 = v23;
        v20 = v29;
      }
      cb_pop_front(get_compile_time_arg_val(0), v5);
      cb_pop_front(get_compile_time_arg_val(1), v6);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v10);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<false>(v3, get_compile_time_arg_val(2), v3);
    pack_tile<false>(v4, get_compile_time_arg_val(2), v4);
    pack_tile<false>(v7, get_compile_time_arg_val(2), v7);
    pack_tile<false>(v11, get_compile_time_arg_val(2), v11);
    pack_tile<false>(v8, get_compile_time_arg_val(2), v8);
    pack_tile<false>(v12, get_compile_time_arg_val(2), v12);
    pack_tile<false>(v13, get_compile_time_arg_val(2), v13);
    pack_tile<false>(v14, get_compile_time_arg_val(2), v14);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v10);
  }
  return;
}
void MAIN { kernel_main(); }
}
