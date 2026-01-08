// add_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = get_arg_val<uint32_t>(2);
  DataFormat v4 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v5 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v6;
  v6.bank_base_address = v3;
  v6.page_size = v5;
  v6.data_format = v4;
  InterleavedAddrGenFast<true> v7 = v6;
  int32_t v8 = get_arg_val<uint32_t>(5);
  int32_t v9 = get_arg_val<uint32_t>(4);
  for (int32_t i10 = v9; i10 < v8; i10 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v1);
    }
    int32_t v11 = get_read_ptr(get_compile_time_arg_val(2));
    uint64_t temp_44 = v7.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) i10 * (uint32_t) 2048)) / (uint32_t) v5), 0);
    noc_async_write(v11, temp_44, v5);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v1);
  }
  return;
}

