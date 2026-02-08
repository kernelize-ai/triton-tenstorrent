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
  size_t v3 = 2;
  size_t v4 = 3;
  size_t v5 = 4;
  size_t v6 = 5;
  size_t v7 = 6;
  size_t v8 = 7;
  int32_t v9 = 0;
  int32_t v10 = 1;
  int32_t v11 = 32;
  int32_t v12 = 64;
  int32_t v13 = 16;
  int32_t v14 = 48;
  int32_t v15 = 17;
  int32_t v16 = 33;
  int32_t v17 = 49;
  int32_t v18 = 2;
  int32_t v19 = 18;
  int32_t v20 = 34;
  int32_t v21 = 50;
  int32_t v22 = 3;
  int32_t v23 = 19;
  int32_t v24 = 35;
  int32_t v25 = 51;
  int32_t v26 = 4;
  int32_t v27 = 20;
  int32_t v28 = 36;
  int32_t v29 = 52;
  int32_t v30 = 5;
  int32_t v31 = 21;
  int32_t v32 = 37;
  int32_t v33 = 53;
  int32_t v34 = 6;
  int32_t v35 = 22;
  int32_t v36 = 38;
  int32_t v37 = 54;
  int32_t v38 = 7;
  int32_t v39 = 23;
  int32_t v40 = 39;
  int32_t v41 = 55;
  int32_t v42 = 8;
  int32_t v43 = 24;
  int32_t v44 = 40;
  int32_t v45 = 56;
  int32_t v46 = 9;
  int32_t v47 = 25;
  int32_t v48 = 41;
  int32_t v49 = 57;
  int32_t v50 = 10;
  int32_t v51 = 26;
  int32_t v52 = 42;
  int32_t v53 = 58;
  int32_t v54 = 11;
  int32_t v55 = 27;
  int32_t v56 = 43;
  int32_t v57 = 59;
  int32_t v58 = 12;
  int32_t v59 = 28;
  int32_t v60 = 44;
  int32_t v61 = 60;
  int32_t v62 = 13;
  int32_t v63 = 29;
  int32_t v64 = 45;
  int32_t v65 = 61;
  int32_t v66 = 14;
  int32_t v67 = 30;
  int32_t v68 = 46;
  int32_t v69 = 62;
  int32_t v70 = 15;
  int32_t v71 = 31;
  int32_t v72 = 47;
  int32_t v73 = 63;
  int32_t v74 = get_common_arg_val<uint32_t>(32);
  mm_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2), v9);
  int32_t v75 = get_arg_val<uint32_t>(v2);
  int32_t v76 = get_arg_val<uint32_t>(v1);
  for (int32_t i77 = v76; i77 < v75; i77 += v10) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9);
    for (int32_t j78 = v9; j78 < ((int32_t) ((uint32_t) v74 + (uint32_t) 511) / 512); j78 += v10) {
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(0), v11);
      }
      {
      DeviceZoneScopedN("cb_wait_front");
      cb_wait_front(get_compile_time_arg_val(1), v12);
      }
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v9, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v13, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v11, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v14, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v9, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v13, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v11, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v14, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v10, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v15, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v16, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v17, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v10, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v15, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v16, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v17, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v18, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v19, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v20, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v21, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v18, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v19, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v20, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v21, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v22, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v23, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v24, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v25, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v22, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v23, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v24, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v25, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v26, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v27, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v28, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v29, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v26, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v27, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v28, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v29, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v30, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v31, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v32, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v33, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v30, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v31, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v32, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v33, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v34, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v35, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v36, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v37, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v34, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v35, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v36, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v37, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v38, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v39, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v40, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v41, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v38, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v39, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v40, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v41, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v42, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v43, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v44, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v45, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v43, v42, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v43, v43, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v43, v44, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v43, v45, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v46, v46, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v46, v47, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v46, v48, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v46, v49, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v47, v46, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v47, v47, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v47, v48, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v47, v49, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v50, v50, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v50, v51, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v50, v52, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v50, v53, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v51, v50, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v51, v51, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v51, v52, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v51, v53, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v54, v54, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v54, v55, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v54, v56, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v54, v57, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v55, v54, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v55, v55, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v55, v56, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v55, v57, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v58, v58, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v58, v59, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v58, v60, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v58, v61, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v59, v58, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v59, v59, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v59, v60, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v59, v61, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v62, v62, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v62, v63, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v62, v64, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v62, v65, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v63, v62, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v63, v63, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v63, v64, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v63, v65, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v66, v66, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v66, v67, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v66, v68, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v66, v69, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v67, v66, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v67, v67, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v67, v68, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v67, v69, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v70, v70, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v70, v71, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v70, v72, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v70, v73, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v71, v70, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v71, v71, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v71, v72, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v71, v73, v8);
      cb_pop_front(get_compile_time_arg_val(0), v11);
      cb_pop_front(get_compile_time_arg_val(1), v12);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v42);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v9, get_compile_time_arg_val(2), v9);
    pack_tile<true>(v10, get_compile_time_arg_val(2), v10);
    pack_tile<true>(v18, get_compile_time_arg_val(2), v18);
    pack_tile<true>(v22, get_compile_time_arg_val(2), v22);
    pack_tile<true>(v26, get_compile_time_arg_val(2), v26);
    pack_tile<true>(v30, get_compile_time_arg_val(2), v30);
    pack_tile<true>(v34, get_compile_time_arg_val(2), v34);
    pack_tile<true>(v38, get_compile_time_arg_val(2), v38);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v42);
  }
  return;
}
void MAIN { kernel_main(); }
}
