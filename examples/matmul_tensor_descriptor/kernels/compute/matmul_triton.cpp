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
  size_t v1 = 0;
  size_t v2 = 2;
  int32_t v3 = 0;
  int32_t v4 = 1;
  int32_t v5 = get_arg_val<uint32_t>(v1);
  int32_t v6 = get_arg_val<uint32_t>(1);
  int32_t v7 = get_arg_val<uint32_t>(v2);
  int32_t v8 = get_arg_val<uint32_t>(3);
  int32_t v9 = get_arg_val<uint32_t>(4);
  int32_t v10 = get_arg_val<uint32_t>(5);
  int32_t v11 = get_arg_val<uint32_t>(6);
  int32_t v12 = get_arg_val<uint32_t>(7);
  int32_t v13 = get_arg_val<uint32_t>(8);
  int32_t v14 = get_arg_val<uint32_t>(9);
  int32_t v15 = get_arg_val<uint32_t>(10);
  int32_t v16 = get_arg_val<uint32_t>(11);
  int32_t v17 = get_arg_val<uint32_t>(12);
  int32_t v18 = get_arg_val<uint32_t>(13);
  int32_t v19 = get_arg_val<uint32_t>(14);
  int32_t v20 = get_arg_val<uint32_t>(15);
  int32_t v21 = get_arg_val<uint32_t>(16);
  int32_t v22 = get_arg_val<uint32_t>(17);
  int32_t v23 = get_arg_val<uint32_t>(18);
  int32_t v24 = get_arg_val<uint32_t>(19);
  int32_t v25 = get_arg_val<uint32_t>(20);
  int32_t v26 = get_arg_val<uint32_t>(21);
  int32_t v27 = get_arg_val<uint32_t>(22);
  int32_t v28 = get_arg_val<uint32_t>(23);
  int32_t v29 = get_arg_val<uint32_t>(24);
  int32_t v30 = get_arg_val<uint32_t>(25);
  int32_t v31 = get_arg_val<uint32_t>(26);
  int32_t v32 = get_arg_val<uint32_t>(27);
  int32_t v33 = get_arg_val<uint32_t>(28);
  int32_t v34 = get_arg_val<uint32_t>(29);
  int32_t v35 = get_arg_val<uint32_t>(30);
  int32_t v36 = get_arg_val<uint32_t>(31);
  int32_t v37 = get_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v3);
  int32_t v38 = get_arg_val<uint32_t>(34);
  int32_t v39 = get_arg_val<uint32_t>(33);
  for (int32_t i40 = v39; i40 < v38; i40 += v4) {
    tile_regs_acquire();
    for (int32_t j41 = v3; j41 < ((int32_t) ((uint32_t) v37 + (uint32_t) 31) / 32); j41 += v4) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v4);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v4);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1, v1, v2);
      cb_pop_front(get_compile_time_arg_val(0), v4);
      cb_pop_front(get_compile_time_arg_val(1), v4);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v4);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v2, get_compile_time_arg_val(2), v1);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v4);
  }
  return;
}
void MAIN { kernel_main(); }
}
