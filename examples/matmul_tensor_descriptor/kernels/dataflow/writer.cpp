// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = get_common_arg_val<uint32_t>(20);
  int32_t v5 = get_common_arg_val<uint32_t>(22);
  int32_t v6 = get_common_arg_val<uint32_t>(30);
  int32_t v7 = get_common_arg_val<uint32_t>(31);
  DataFormat v8 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v9 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v10;
  v10.bank_base_address = v4;
  v10.page_size = v9;
  v10.data_format = v8;
  InterleavedAddrGenFast<true> v11 = v10;
  int32_t v12 = get_arg_val<uint32_t>(1);
  int32_t v13 = get_arg_val<uint32_t>(0);
  for (int32_t i14 = v13; i14 < v12; i14 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v1);
    }
    int32_t v15 = get_read_ptr(get_compile_time_arg_val(2));
    uint64_t temp_137 = v11.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i14 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) + (uint32_t) (i14 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i14 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i14 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v1)))) * (uint32_t) 32) / 32) * (uint32_t) (v5 != (int32_t) ((uint32_t) (v5 / 32) * (uint32_t) 32) & v5 < v3 == false ? (int32_t) ((uint32_t) (v5 / 32) + (uint32_t) v1) : v5 / 32))) + (uint32_t) ((int32_t) ((uint32_t) ((i14 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i14 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i14 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v1)) * (uint32_t) 32) / 32)), v3);
    noc_async_write(v15, temp_137, v9);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v1);
  }
  return;
}
