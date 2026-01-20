// matmul_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = true;
  int32_t v2 = 1;
  int32_t v3 = get_common_arg_val<uint32_t>(2);
  int32_t v4 = get_common_arg_val<uint32_t>(3);
  int32_t v5 = get_common_arg_val<uint32_t>(4);
  DataFormat v6 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v7 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v8;
  v8.bank_base_address = v3;
  v8.page_size = v7;
  v8.data_format = v6;
  InterleavedAddrGenFast<true> v9 = v8;
  int32_t v10 = get_arg_val<uint32_t>(1);
  int32_t v11 = get_arg_val<uint32_t>(0);
  for (int32_t i12 = v11; i12 < v10; i12 += v2) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v2);
    }
    int32_t v13 = get_read_ptr(get_compile_time_arg_val(2));
    uint64_t temp_146 = v9.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) + (uint32_t) ((i12 % ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) : v2)))) * (uint32_t) 32) % v4) / 32) * (uint32_t) ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) + (uint32_t) (((int32_t) ((uint32_t) ((i12 % ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) : v2)) * (uint32_t) 32) % v5) / 32))) * (uint32_t) 1024)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) + (uint32_t) ((i12 % ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) : v2)))) * (uint32_t) 32) % v4) % 32) * (uint32_t) 32)) + (uint32_t) (((int32_t) ((uint32_t) ((i12 % ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32) - (uint32_t) (i12 / ((int32_t) ((uint32_t) v5 + (uint32_t) 31) / 32))) : v2)) * (uint32_t) 32) % v5) % 32))))) * (uint32_t) 2)) / (uint32_t) v7), 0);
    noc_async_write(v13, temp_146, v7);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v2);
  }
  return;
}
