// matmul_kernel__compute
#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
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
