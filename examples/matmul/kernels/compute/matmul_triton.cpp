// matmul_kernel__compute
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
  int32_t v2 = 0;
  int32_t v3 = 1;
  int32_t v4 = get_common_arg_val<uint32_t>(5);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v2);
  int32_t v5 = get_arg_val<uint32_t>(1);
  int32_t v6 = get_arg_val<uint32_t>(v1);
  for (int32_t i7 = v6; i7 < v5; i7 += v3) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2);
    for (int32_t j8 = v2; j8 < ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32); j8 += v3) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v3);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v3);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v2, v1);
      cb_pop_front(get_compile_time_arg_val(0), v3);
      cb_pop_front(get_compile_time_arg_val(1), v3);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v3);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v2, get_compile_time_arg_val(2), v2);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v3);
  }
  return;
}
void MAIN { kernel_main(); }
}
