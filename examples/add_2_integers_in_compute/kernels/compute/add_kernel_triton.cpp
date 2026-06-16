// add_kernel__compute
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
void kernel_main() {
  size_t v1 = 0;
  size_t v2 = 1;
  int32_t v3 = 1;
  int32_t v4 = 0;
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  int32_t v5 = get_common_arg_val<uint32_t>(4);
  int32_t v6 = get_common_arg_val<uint32_t>(6);
  int32_t v7 = get_common_arg_val<uint32_t>(7);
  int32_t v8 = get_common_arg_val<uint32_t>(5);
  int32_t v9 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_x())) * (uint32_t) v5)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_y())) * (uint32_t) v8)));
  for (int32_t i10 = v9; i10 < ((int32_t) ((uint32_t) v9 + (uint32_t) v5) < (int32_t) ((uint32_t) v6 * (uint32_t) v7) ? (int32_t) ((uint32_t) v9 + (uint32_t) v5) : (int32_t) ((uint32_t) v6 * (uint32_t) v7)); i10 += v3) {
    tile_regs_acquire();
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_ctarg_0.wait_front(v3);
    }
    copy_tile_init(get_compile_time_arg_val(0));
    copy_tile(get_compile_time_arg_val(0), v1, v1);
    cb_ctarg_0.pop_front(v3);
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_ctarg_1.wait_front(v3);
    }
    copy_tile_init(get_compile_time_arg_val(1));
    copy_tile(get_compile_time_arg_val(1), v1, v2);
    cb_ctarg_1.pop_front(v3);
    add_binary_tile_init();
    add_binary_tile(v1, v2, v1);
    cb_ctarg_2.reserve_back(v3);
    tile_regs_commit();
    {
    DeviceZoneScopedN("tile_regs_wait");
    tile_regs_wait();
    }
    pack_tile<true>(v4, get_compile_time_arg_val(2), v4);
    tile_regs_release();
    cb_ctarg_2.push_back(v3);
  }
  return;
}
