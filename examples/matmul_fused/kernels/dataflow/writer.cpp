// matmul_kernel_fused__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 64;
  int32_t v2 = 32;
  int32_t v3 = 1;
  bool v4 = true;
  int32_t v5 = 2;
  int32_t v6 = 0;
  int32_t v7 = get_common_arg_val<uint32_t>(20);
  int32_t v8 = get_common_arg_val<uint32_t>(22);
  int32_t v9 = get_common_arg_val<uint32_t>(40);
  int32_t v10 = get_common_arg_val<uint32_t>(41);
  DataFormat v11 = get_dataformat(get_compile_time_arg_val(3));
  int32_t v12 = get_tile_size(get_compile_time_arg_val(3));
  InterleavedAddrGenFast<true> v13;
  v13.bank_base_address = v7;
  v13.page_size = v12;
  v13.data_format = v11;
  InterleavedAddrGenFast<true> v14 = v13;
  int32_t v15 = get_arg_val<uint32_t>(1);
  int32_t v16 = get_arg_val<uint32_t>(0);
  int32_t v17 = (int32_t) ((uint32_t) v10 + (uint32_t) 63) / v1;
  for (int32_t i18 = v16; i18 < v15; i18 += v3) {
    int32_t v19 = i18 / v17;
    int32_t v20 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 31) / v2) - (uint32_t) v19) < v3 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 31) / v2) - (uint32_t) v19) : v3;
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(3), v5);
    }
    int32_t v21 = (int32_t) ((uint32_t) ((i18 % v17) / v20) * (uint32_t) v1) / v2;
    int32_t v22 = get_read_ptr(get_compile_time_arg_val(3));
    int32_t v23 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v19 + (uint32_t) (i18 % v20))) * (uint32_t) v2) / v2) * (uint32_t) (v8 != (int32_t) ((uint32_t) (v8 / v2) * (uint32_t) v2) & v8 < v6 == false ? (int32_t) ((uint32_t) (v8 / v2) + (uint32_t) v3) : v8 / v2));
    uint64_t temp_143 = v14.get_noc_addr((int32_t) ((uint32_t) v23 + (uint32_t) v21), v6);
    noc_async_write(v22, temp_143, v12);
    int32_t v24 = (int32_t) ((uint32_t) v22 + (uint32_t) v12);
    uint64_t temp_160 = v14.get_noc_addr((int32_t) ((uint32_t) v23 + (uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) v3))), v6);
    noc_async_write(v24, temp_160, v12);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(3), v5);
  }
  return;
}
