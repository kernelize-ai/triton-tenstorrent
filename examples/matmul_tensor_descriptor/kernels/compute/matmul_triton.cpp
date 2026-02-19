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
  size_t v5 = 4;
  size_t v6 = 5;
  size_t v7 = 6;
  size_t v8 = 7;
  int32_t v9 = 0;
  int32_t v10 = 1;
  int32_t v11 = 32;
  int32_t v12 = 64;
  int32_t v13 = 2;
  int32_t v14 = 3;
  int32_t v15 = 16;
  int32_t v16 = 4;
  int32_t v17 = 5;
  int32_t v18 = 6;
  int32_t v19 = 7;
  int32_t v20 = 17;
  int32_t v21 = 8;
  int32_t v22 = 9;
  int32_t v23 = 10;
  int32_t v24 = 11;
  int32_t v25 = 18;
  int32_t v26 = 12;
  int32_t v27 = 13;
  int32_t v28 = 14;
  int32_t v29 = 15;
  int32_t v30 = 19;
  int32_t v31 = 20;
  int32_t v32 = 21;
  int32_t v33 = 22;
  int32_t v34 = 23;
  int32_t v35 = 24;
  int32_t v36 = 25;
  int32_t v37 = 26;
  int32_t v38 = 27;
  int32_t v39 = 28;
  int32_t v40 = 29;
  int32_t v41 = 30;
  int32_t v42 = 31;
  int32_t v43 = 33;
  int32_t v44 = 34;
  int32_t v45 = 35;
  int32_t v46 = 36;
  int32_t v47 = 37;
  int32_t v48 = 38;
  int32_t v49 = 39;
  int32_t v50 = 40;
  int32_t v51 = 41;
  int32_t v52 = 42;
  int32_t v53 = 43;
  int32_t v54 = 44;
  int32_t v55 = 45;
  int32_t v56 = 46;
  int32_t v57 = 47;
  int32_t v58 = 48;
  int32_t v59 = 49;
  int32_t v60 = 50;
  int32_t v61 = 51;
  int32_t v62 = 52;
  int32_t v63 = 53;
  int32_t v64 = 54;
  int32_t v65 = 55;
  int32_t v66 = 56;
  int32_t v67 = 57;
  int32_t v68 = 58;
  int32_t v69 = 59;
  int32_t v70 = 60;
  int32_t v71 = 61;
  int32_t v72 = 62;
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
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v10, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v13, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v14, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v9, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v10, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v13, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v15, v14, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v16, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v17, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v18, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v19, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v16, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v17, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v18, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v20, v19, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v21, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v22, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v23, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v24, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v21, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v22, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v23, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v25, v24, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v26, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v27, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v28, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v29, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v26, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v27, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v28, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v30, v29, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v15, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v20, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v25, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v30, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v15, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v20, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v25, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v31, v30, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v31, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v32, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v33, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v34, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v32, v31, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v32, v32, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v32, v33, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v32, v34, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v35, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v36, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v37, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v18, v38, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v33, v35, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v33, v36, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v33, v37, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v33, v38, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v39, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v40, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v41, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v19, v42, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v39, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v40, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v41, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v34, v42, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v11, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v43, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v44, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v21, v45, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v11, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v43, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v44, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v35, v45, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v46, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v47, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v48, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v22, v49, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v36, v46, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v36, v47, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v36, v48, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v36, v49, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v50, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v51, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v52, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v23, v53, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v37, v50, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v37, v51, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v37, v52, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v37, v53, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v54, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v55, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v56, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v24, v57, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v54, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v55, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v56, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v38, v57, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v58, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v59, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v60, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v26, v61, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v58, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v59, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v60, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v39, v61, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v62, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v63, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v64, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v27, v65, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v40, v62, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v40, v63, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v40, v64, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v40, v65, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v28, v66, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v28, v67, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v28, v68, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v28, v69, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v41, v66, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v41, v67, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v41, v68, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v41, v69, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v29, v70, v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v29, v71, v2);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v29, v72, v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v29, v73, v4);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v70, v5);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v71, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v72, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v42, v73, v8);
      cb_pop_front(get_compile_time_arg_val(0), v11);
      cb_pop_front(get_compile_time_arg_val(1), v12);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v21);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v9, get_compile_time_arg_val(2), v9);
    pack_tile<true>(v10, get_compile_time_arg_val(2), v10);
    pack_tile<true>(v13, get_compile_time_arg_val(2), v13);
    pack_tile<true>(v14, get_compile_time_arg_val(2), v14);
    pack_tile<true>(v16, get_compile_time_arg_val(2), v16);
    pack_tile<true>(v17, get_compile_time_arg_val(2), v17);
    pack_tile<true>(v18, get_compile_time_arg_val(2), v18);
    pack_tile<true>(v19, get_compile_time_arg_val(2), v19);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v21);
  }
  return;
}
