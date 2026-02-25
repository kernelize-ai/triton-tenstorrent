// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 63;
  int32_t v2 = 0;
  int32_t v3 = 64;
  int32_t v4 = 1;
  int32_t v5 = 4;
  bool v6 = true;
  int32_t v7 = 32;
  int32_t v8 = get_common_arg_val<uint32_t>(10);
  int32_t v9 = get_common_arg_val<uint32_t>(12);
  int32_t v10 = get_common_arg_val<uint32_t>(30);
  int32_t v11 = get_common_arg_val<uint32_t>(31);
  int32_t v12 = get_common_arg_val<uint32_t>(32);
  DataFormat v13 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v14 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v15 = get_arg_val<uint32_t>(1);
  int32_t v16 = get_arg_val<uint32_t>(0);
  int32_t v17 = v9 != (int32_t) ((uint32_t) (v9 / v7) * (uint32_t) v7) & v9 < v2 == false ? (int32_t) ((uint32_t) (v9 / v7) + (uint32_t) v4) : v9 / v7;
  for (int32_t i18 = v16; i18 < v15; i18 += v4) {
    for (int32_t j19 = v2; j19 < ((int32_t) ((uint32_t) v12 + (uint32_t) v1) / v3); j19 += v4) {
      cb_reserve_back(get_compile_time_arg_val(1), v5);
      InterleavedAddrGenFast<true> v20;
      v20.bank_base_address = v8;
      v20.page_size = v14;
      v20.data_format = v13;
      InterleavedAddrGenFast<true> v21 = v20;
      int32_t v22 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v23 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j19 * (uint32_t) v3) / v7) * (uint32_t) v17)) + (uint32_t) ((int32_t) ((uint32_t) ((i18 % ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v3)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 + (uint32_t) v1) / v3) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v3))) < v4 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 + (uint32_t) v1) / v3) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v3))) : v4)) * (uint32_t) v3) / v7));
      uint64_t temp_319 = v21.get_noc_addr(v23, v2);
      noc_async_read(temp_319, v22, v14);
      int32_t v24 = (int32_t) ((uint32_t) v22 + (uint32_t) 4096);
      int32_t v25 = (int32_t) ((uint32_t) v23 + (uint32_t) v4);
      uint64_t temp_331 = v21.get_noc_addr(v25, v2);
      noc_async_read(temp_331, v24, v14);
      int32_t v26 = (int32_t) ((uint32_t) v22 + (uint32_t) 2048);
      uint64_t temp_343 = v21.get_noc_addr((int32_t) ((uint32_t) v23 + (uint32_t) v17), v2);
      noc_async_read(temp_343, v26, v14);
      int32_t v27 = (int32_t) ((uint32_t) v22 + (uint32_t) 6144);
      uint64_t temp_355 = v21.get_noc_addr((int32_t) ((uint32_t) v25 + (uint32_t) v17), v2);
      noc_async_read(temp_355, v27, v14);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v5);
    }
  }
  return;
}
