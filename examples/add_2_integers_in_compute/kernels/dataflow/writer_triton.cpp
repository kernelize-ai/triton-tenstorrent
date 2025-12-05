// add_kernel__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = get_arg_val<uint32_t>(0);
  int32_t v4 = get_arg_val<uint32_t>(1);
  int32_t v5 = get_arg_val<uint32_t>(2);
  int32_t v6 = get_arg_val<uint32_t>(3);
  int32_t v7 = get_arg_val<uint32_t>(4);
  {
  DeviceZoneScopedN("cb_wait_front");
  cb_wait_front(get_compile_time_arg_val(2), v1);
  }
  int32_t v8 = get_tile_size(get_compile_time_arg_val(2));
  DataFormat v9 = get_dataformat(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v10;
  v10.bank_base_address = v5;
  v10.page_size = v8;
  v10.data_format = v9;
  InterleavedAddrGenFast<true> v11 = v10;
  uint64_t temp_41 = v11.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 * (uint32_t) 2048)) / (uint32_t) v8), 0);
  int32_t v12 = get_read_ptr(get_compile_time_arg_val(2));
  noc_async_write(v12, temp_41, v8);
  {
  DeviceZoneScopedN("noc_async_write_barrier");
  noc_async_write_barrier();
  }
  cb_pop_front(get_compile_time_arg_val(2), v1);
  return;
}
