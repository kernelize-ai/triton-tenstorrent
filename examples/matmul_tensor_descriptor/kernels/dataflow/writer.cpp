// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 8;
  int32_t v4 = 0;
  int32_t v5 = 32;
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
  int32_t v16 = v7 != (int32_t) ((uint32_t) (v7 / v5) * (uint32_t) v5) & v7 < v4 == false ? (int32_t) ((uint32_t) (v7 / v5) + (uint32_t) v1) : v7 / v5;
  for (int32_t i17 = v15; i17 < v14; i17 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v3);
    }
    int32_t v18 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v19 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128)) + (uint32_t) (i17 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128))) : v1)))) * (uint32_t) 64) / v5) * (uint32_t) v16)) + (uint32_t) ((int32_t) ((uint32_t) ((i17 % ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) 127) / 128))) : v1)) * (uint32_t) 128) / v5));
    uint64_t temp_163 = v13.get_noc_addr(v19, v4);
    noc_async_write(v18, temp_163, v11);
    int32_t v20 = (int32_t) ((uint32_t) v18 + (uint32_t) 2048);
    int32_t v21 = (int32_t) ((uint32_t) v19 + (uint32_t) v1);
    uint64_t temp_175 = v13.get_noc_addr(v21, v4);
    noc_async_write(v20, temp_175, v11);
    int32_t v22 = (int32_t) ((uint32_t) v18 + (uint32_t) 4096);
    int32_t v23 = (int32_t) ((uint32_t) v19 + (uint32_t) 2);
    uint64_t temp_187 = v13.get_noc_addr(v23, v4);
    noc_async_write(v22, temp_187, v11);
    int32_t v24 = (int32_t) ((uint32_t) v18 + (uint32_t) 6144);
    int32_t v25 = (int32_t) ((uint32_t) v19 + (uint32_t) 3);
    uint64_t temp_199 = v13.get_noc_addr(v25, v4);
    noc_async_write(v24, temp_199, v11);
    int32_t v26 = (int32_t) ((uint32_t) v18 + (uint32_t) 8192);
    uint64_t temp_211 = v13.get_noc_addr((int32_t) ((uint32_t) v19 + (uint32_t) v16), v4);
    noc_async_write(v26, temp_211, v11);
    int32_t v27 = (int32_t) ((uint32_t) v18 + (uint32_t) 10240);
    uint64_t temp_223 = v13.get_noc_addr((int32_t) ((uint32_t) v21 + (uint32_t) v16), v4);
    noc_async_write(v27, temp_223, v11);
    int32_t v28 = (int32_t) ((uint32_t) v18 + (uint32_t) 12288);
    uint64_t temp_235 = v13.get_noc_addr((int32_t) ((uint32_t) v23 + (uint32_t) v16), v4);
    noc_async_write(v28, temp_235, v11);
    int32_t v29 = (int32_t) ((uint32_t) v18 + (uint32_t) 14336);
    uint64_t temp_247 = v13.get_noc_addr((int32_t) ((uint32_t) v25 + (uint32_t) v16), v4);
    noc_async_write(v29, temp_247, v11);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v3);
  }
  return;
}

