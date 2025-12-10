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
  int32_t v13 = get_arg_val<uint32_t>(10);
  int32_t v14 = get_arg_val<uint32_t>(11);
  DataFormat v15 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v16 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v17;
  v17.bank_base_address = v5;
  v17.page_size = v16;
  v17.data_format = v15;
  InterleavedAddrGenFast<true> v18 = v17;
  int32_t v19 = get_arg_val<uint32_t>(12);
  {
  DeviceZoneScopedN("cb_wait_front");
  cb_wait_front(get_compile_time_arg_val(2), v1);
  }
  int32_t output_offset = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 * (uint32_t) ((int32_t) ((uint32_t) ((v19 % ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (v19 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v6 + (uint32_t) 31) / 32) - (uint32_t) (v19 / ((int32_t) ((uint32_t) v7 + (uint32_t) 31) / 32))) : v1)) * (uint32_t) 32)))) * (uint32_t) 2)) / (uint32_t) v16);
  DPRINT << "Output offset = " << output_offset << "\n";
  uint64_t temp_106 = v18.get_noc_addr(0, 0);
  int32_t v20 = get_read_ptr(get_compile_time_arg_val(2));
  noc_async_write(v20, temp_106, v16);
  {
  DeviceZoneScopedN("noc_async_write_barrier");
  noc_async_write_barrier();
  }
  cb_pop_front(get_compile_time_arg_val(2), v1);
  return;
}
