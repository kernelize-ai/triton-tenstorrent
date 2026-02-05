// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 2;
  int32_t v2 = 1;
  bool v3 = true;
  int32_t v4 = 4;
  int32_t v5 = 0;
  int32_t v6 = get_common_arg_val<uint32_t>(20);
  int32_t v7 = get_common_arg_val<uint32_t>(22);
  int32_t v8 = get_common_arg_val<uint32_t>(30);
  int32_t v9 = get_common_arg_val<uint32_t>(31);
  DataFormat v10 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v11 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v12;
  v12.bank_base_address = v6;
  v12.page_size = v11;
  v12.data_format = v10;
  InterleavedAddrGenFast<true> v13 = v12;
  int32_t v14 = get_arg_val<uint32_t>(1);
  int32_t v15 = get_arg_val<uint32_t>(0);
  for (int32_t i16 = v15; i16 < v14; i16 += v2) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v4);
    }
    int32_t v17 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v18 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i16 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) * (uint32_t) v1)) + (uint32_t) (i16 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i16 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) * (uint32_t) v1))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i16 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) * (uint32_t) v1))) : v1)))) * (uint32_t) 32) / 32) * (uint32_t) (v7 != (int32_t) ((uint32_t) (v7 / 32) * (uint32_t) 32) & v7 < v5 == false ? (int32_t) ((uint32_t) (v7 / 32) + (uint32_t) v2) : v7 / 32))) + (uint32_t) ((int32_t) ((uint32_t) ((i16 % (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i16 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) * (uint32_t) v1))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i16 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128) * (uint32_t) v1)) * (uint32_t) v1))) : v1)) * (uint32_t) 128) / 32));
    uint64_t temp_163 = v13.get_noc_addr(v18, v5);
    noc_async_write(v17, temp_163, v11);
    int32_t v19 = (int32_t) ((uint32_t) v17 + (uint32_t) 2048);
    uint64_t temp_175 = v13.get_noc_addr((int32_t) ((uint32_t) v18 + (uint32_t) v2), v5);
    noc_async_write(v19, temp_175, v11);
    int32_t v20 = (int32_t) ((uint32_t) v17 + (uint32_t) 4096);
    uint64_t temp_187 = v13.get_noc_addr((int32_t) ((uint32_t) v18 + (uint32_t) v1), v5);
    noc_async_write(v20, temp_187, v11);
    int32_t v21 = (int32_t) ((uint32_t) v17 + (uint32_t) 6144);
    uint64_t temp_199 = v13.get_noc_addr((int32_t) ((uint32_t) v18 + (uint32_t) 3), v5);
    noc_async_write(v21, temp_199, v11);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v4);
  }
  return;
}
