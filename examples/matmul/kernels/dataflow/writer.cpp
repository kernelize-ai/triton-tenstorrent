// matmul_kernel__writer
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
  int32_t v8 = get_arg_val<uint32_t>(5);
  int32_t v9 = get_arg_val<uint32_t>(6);
  int32_t v10 = get_arg_val<uint32_t>(7);
  int32_t v11 = get_arg_val<uint32_t>(8);
  int32_t v12 = get_arg_val<uint32_t>(9);
  {
  DeviceZoneScopedN("cb_wait_front");
  cb_wait_front(get_compile_time_arg_val(2), v1);
  }
  int32_t v13 = get_tile_size(get_compile_time_arg_val(2));
  DataFormat v14 = get_dataformat(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v15;
  v15.bank_base_address = v5;
  v15.page_size = v13;
  v15.data_format = v14;
  InterleavedAddrGenFast<true> v16 = v15;
  DPRINT << "Write index " << (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 * (uint32_t) 2)) / (uint32_t) v13) << "\n";
  uint64_t temp_56 = v16.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 * (uint32_t) 2)) / (uint32_t) v13), 0);
  int32_t v17 = get_read_ptr(get_compile_time_arg_val(2));
  noc_async_write(v17, temp_56, v13);
  {
  DeviceZoneScopedN("noc_async_write_barrier");
  noc_async_write_barrier();
  }
  cb_pop_front(get_compile_time_arg_val(2), v1);
  return;
}
