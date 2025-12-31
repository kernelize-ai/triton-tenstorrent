// add_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = get_arg_val<uint32_t>(0);
  int32_t v4 = get_arg_val<uint32_t>(1);
  int32_t v5 = get_arg_val<uint32_t>(2);
  int32_t v6 = get_arg_val<uint32_t>(3);
  DataFormat v7 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v8 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v9;
  v9.bank_base_address = v5;
  v9.page_size = v8;
  v9.data_format = v7;
  InterleavedAddrGenFast<true> v10 = v9;
  int32_t v11 = get_arg_val<uint32_t>(5);
  int32_t v12 = get_arg_val<uint32_t>(4);
  for (int32_t i13 = v12; i13 < v11; i13 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v1);
    }
    uint64_t temp_54 = v10.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) i13 * (uint32_t) 2048)) / (uint32_t) v8), 0);
    int32_t v14 = get_read_ptr(get_compile_time_arg_val(2));
    noc_async_write(v14, temp_54, v8);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v1);
  }
  return;
}
