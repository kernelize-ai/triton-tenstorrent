// matmul_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  size_t v1 = 0;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = 32;
  int32_t v5 = 1;
  int32_t v6 = 31;
  int32_t v7 = 1024;
  int32_t v8 = 2;
  int32_t v9 = get_common_arg_val<uint32_t>(v1);
  int32_t v10 = get_common_arg_val<uint32_t>(2);
  int32_t v11 = get_common_arg_val<uint32_t>(3);
  int32_t v12 = get_common_arg_val<uint32_t>(4);
  int32_t v13 = get_common_arg_val<uint32_t>(5);
  DataFormat v14 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v15 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v16;
  v16.bank_base_address = v10;
  v16.page_size = v15;
  v16.data_format = v14;
  InterleavedAddrGenFast<true> v17 = v16;
  DataFormat v18 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v19 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v20;
  v20.bank_base_address = v9;
  v20.page_size = v19;
  v20.data_format = v18;
  InterleavedAddrGenFast<true> v21 = v20;
  int32_t v22 = get_arg_val<uint32_t>(1);
  int32_t v23 = get_arg_val<uint32_t>(v1);
  int32_t v24 = (int32_t) ((uint32_t) v12 + (uint32_t) v6) / v4;
  int32_t v25 = (int32_t) ((uint32_t) v13 + (uint32_t) v6) / v4;
  for (int32_t i26 = v23; i26 < v22; i26 += v5) {
    int32_t v27 = i26 / v24;
    int32_t v28 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v6) / v4) - (uint32_t) v27) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v6) / v4) - (uint32_t) v27) : v5;
    int32_t v29 = i26 % v24;
    int32_t v30 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) (v29 % v28))) * (uint32_t) v4) % v11;
    int32_t v31 = v30 / v4;
    int32_t v32 = (int32_t) ((uint32_t) (v30 % v4) * (uint32_t) v4);
    for (int32_t j33 = v3; j33 < v25; j33 += v5) {
      cb_reserve_back(get_compile_time_arg_val(0), v5);
      int32_t v34 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_187 = v21.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v31 * (uint32_t) v25)) + (uint32_t) ((int32_t) ((uint32_t) j33 * (uint32_t) v4) / v4))) * (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) v32 + (uint32_t) ((int32_t) ((uint32_t) j33 * (uint32_t) v4) % v4))))) * (uint32_t) v8)) / (uint32_t) v19), v3);
      noc_async_read(temp_187, v34, v19);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v5);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v5);
    }
    int32_t v35 = get_read_ptr(get_compile_time_arg_val(2));
    uint64_t temp_176 = v17.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v31 * (uint32_t) v24)) + (uint32_t) (((int32_t) ((uint32_t) (v29 / v28) * (uint32_t) v4) % v12) / v4))) * (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) v32 + (uint32_t) (((int32_t) ((uint32_t) (v29 / v28) * (uint32_t) v4) % v12) % v4))))) * (uint32_t) v8)) / (uint32_t) v15), v3);
    noc_async_write(v35, temp_176, v15);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v5);
  }
  return;
}
