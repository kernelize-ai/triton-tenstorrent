// matmul_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  bool v1 = true;
  int32_t v2 = 1;
  int32_t v3 = get_arg_val<uint32_t>(0);
  int32_t v4 = get_arg_val<uint32_t>(1);
  int32_t v5 = get_arg_val<uint32_t>(2);
  int32_t v6 = get_arg_val<uint32_t>(3);
  int32_t v7 = get_arg_val<uint32_t>(4);
  int32_t v8 = get_arg_val<uint32_t>(5);
  int32_t v9 = get_arg_val<uint32_t>(6);
  int32_t v10 = get_arg_val<uint32_t>(7);
  int32_t v11 = get_arg_val<uint32_t>(8);
  DataFormat v12 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v13 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v14;
  v14.bank_base_address = v5;
  v14.page_size = v13;
  v14.data_format = v12;
  InterleavedAddrGenFast<true> v15 = v14;
  int32_t v16 = get_arg_val<uint32_t>(10);
  int32_t v17 = get_arg_val<uint32_t>(9);
  for (int32_t i18 = v17; i18 < v16; i18 += v2) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v2);
    }
    uint64_t temp_168 = v15.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) + (uint32_t) ((i18 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v2)))) * (uint32_t) 32) % v6) / 32) * (uint32_t) ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) + (uint32_t) (((int32_t) ((uint32_t) ((i18 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v2)) * (uint32_t) 32) % v7) / 32))) * (uint32_t) 1024)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) + (uint32_t) ((i18 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v2)))) * (uint32_t) 32) % v6) % 32) * (uint32_t) 32)) + (uint32_t) (((int32_t) ((uint32_t) ((i18 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v2)) * (uint32_t) 32) % v7) % 32))))) * (uint32_t) 2)) / (uint32_t) v13), 0);
    int32_t v19 = get_read_ptr(get_compile_time_arg_val(2));
    noc_async_write(v19, temp_168, v13);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v2);
  }
  return;
}
