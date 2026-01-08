// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = 2;
  int32_t v5 = get_arg_val<uint32_t>(20);
  int32_t v6 = get_arg_val<uint32_t>(22);
  int32_t v7 = get_arg_val<uint32_t>(30);
  int32_t v8 = get_arg_val<uint32_t>(31);
  DataFormat v9 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v10 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v11;
  v11.bank_base_address = v5;
  v11.page_size = v10;
  v11.data_format = v9;
  InterleavedAddrGenFast<true> v12 = v11;
  int32_t v13 = get_arg_val<uint32_t>(34);
  int32_t v14 = get_arg_val<uint32_t>(33);
  for (int32_t i15 = v14; i15 < v13; i15 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v4);
    }
    int32_t v16 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v17;
    int32_t v18;
    v17 = v16;
    v18 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) + (uint32_t) (i15 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)))) * (uint32_t) 32) / 32) * (uint32_t) (v6 != (int32_t) ((uint32_t) (v6 / 64) * (uint32_t) 64) & v6 < v3 == false ? (int32_t) ((uint32_t) (v6 / 64) + (uint32_t) v1) : v6 / 64))) + (uint32_t) ((int32_t) ((uint32_t) ((i15 % ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)) * (uint32_t) 64) / 64))) * (uint32_t) 2048)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) + (uint32_t) (i15 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)))) * (uint32_t) 32)) % (uint32_t) 32)) * (uint32_t) 64)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i15 % ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)) * (uint32_t) 64)) % (uint32_t) 64)))))) * (uint32_t) v4)) / (uint32_t) v10);
    for (int32_t j19 = v3; j19 < v4; j19 += v1) {
      int32_t v20 = v17;
      int32_t v21 = v18;
      uint64_t temp_192 = v12.get_noc_addr(v21, v3);
      noc_async_write(v20, temp_192, v10);
      v17 = (int32_t) ((uint32_t) v20 + (uint32_t) v10);
      v18 = (int32_t) ((uint32_t) v21 + (uint32_t) v1);
    }
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v4);
  }
  return;
}
