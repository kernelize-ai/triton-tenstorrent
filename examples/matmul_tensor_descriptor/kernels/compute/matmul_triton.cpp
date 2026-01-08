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
inline uint32_t float_to_bits(float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "compute_kernel_api/reduce.h"
namespace NAMESPACE {
void kernel_main() {
  int32_t v1 = 0;
  int32_t v2 = 1;
  int32_t v3 = 2;
  int32_t v4 = 4;
  int32_t v5 = get_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v1);
  int32_t v6 = get_arg_val<uint32_t>(34);
  int32_t v7 = get_arg_val<uint32_t>(33);
  for (int32_t i8 = v7; i8 < v6; i8 += v2) {
    tile_regs_acquire();
    //mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
    for (int32_t j9 = v1; j9 < ((int32_t) ((uint32_t) v5 + (uint32_t) 63) / 64); j9 += v2) {
      {
      DeviceZoneScopedN("cb_wait_front");
      DPRINT << "Waiting for A CB\n";
      cb_wait_front(get_compile_time_arg_val(0), v3);
      DPRINT << "Acquired A CB\n";
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      DPRINT << "Waiting for B CB\n";
      cb_wait_front(get_compile_time_arg_val(1), v4);
      DPRINT << "Acquired B CB\n";
      }
      size_t v10;
      v10 = 0;
      for (int32_t k11 = v1; k11 < v3; k11 += v2) {
        size_t v12 = v10;
        for (int32_t l13 = v1; l13 < v3; l13 += v2) {
          int32_t v14 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) l13 * (uint32_t) v3)) + (uint32_t) k11);
          matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), l13, v14, v12);
        }
        v10 = v12 + 1;
      }
      cb_pop_front(get_compile_time_arg_val(0), v3);
      cb_pop_front(get_compile_time_arg_val(1), v4);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v3);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<false>(v1, get_compile_time_arg_val(2), v1);
    pack_tile<false>(v2, get_compile_time_arg_val(2), v2);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v3);
  }
  DPRINT << "Matmul kernel done\n";
  return;
}
void MAIN { kernel_main(); }
}
