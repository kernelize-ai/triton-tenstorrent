// matmul_kernel_fused__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 2;
  int32_t v4 = 0;
  int32_t v5 = get_common_arg_val<uint32_t>(20);
  int32_t v6 = get_common_arg_val<uint32_t>(22);
  int32_t v7 = get_common_arg_val<uint32_t>(40);
  int32_t v8 = get_common_arg_val<uint32_t>(41);
  DataFormat v9 = get_dataformat(get_compile_time_arg_val(3));
  int32_t v10 = get_tile_size(get_compile_time_arg_val(3));
  InterleavedAddrGenFast<true> v11;
  v11.bank_base_address = v5;
  v11.page_size = v10;
  v11.data_format = v9;
  InterleavedAddrGenFast<true> v12 = v11;
  int32_t v13 = get_arg_val<uint32_t>(1);
  int32_t v14 = get_arg_val<uint32_t>(0);
  for (int32_t i15 = v14; i15 < v13; i15 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(3), v3);
    }
    int32_t v16 = get_read_ptr(get_compile_time_arg_val(3));
    int32_t v17 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) + (uint32_t) (i15 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)))) * (uint32_t) 32) / 32) * (uint32_t) (v6 != (int32_t) ((uint32_t) (v6 / 32) * (uint32_t) 32) & v6 < v4 == false ? (int32_t) ((uint32_t) (v6 / 32) + (uint32_t) v1) : v6 / 32))) + (uint32_t) ((int32_t) ((uint32_t) ((i15 % ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v8 + (uint32_t) 63) / 64))) : v1)) * (uint32_t) 64) / 32));
    uint64_t temp_145 = v12.get_noc_addr(v17, v4);
    noc_async_write(v16, temp_145, v10);
    int32_t v18 = (int32_t) ((uint32_t) v16 + (uint32_t) 2048);
    uint64_t temp_157 = v12.get_noc_addr((int32_t) ((uint32_t) v17 + (uint32_t) v1), v4);
    noc_async_write(v18, temp_157, v10);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(3), v3);
  }
  return;
}
