// add_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = get_arg_val<uint32_t>(0);
  int32_t v5 = get_arg_val<uint32_t>(1);
  DataFormat v6 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v7 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v8;
  v8.bank_base_address = v5;
  v8.page_size = v7;
  v8.data_format = v6;
  InterleavedAddrGenFast<true> v9 = v8;
  DataFormat v10 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v11 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v12;
  v12.bank_base_address = v4;
  v12.page_size = v11;
  v12.data_format = v10;
  InterleavedAddrGenFast<true> v13 = v12;
  int32_t v14 = get_arg_val<uint32_t>(5);
  int32_t v15 = get_arg_val<uint32_t>(4);
  for (int32_t i16 = v15; i16 < v14; i16 += v1) {
    int32_t v17 = (int32_t) ((uint32_t) i16 * (uint32_t) 2048);
    cb_reserve_back(get_compile_time_arg_val(0), v1);
    int32_t v18 = get_write_ptr(get_compile_time_arg_val(0));
    uint64_t temp_91 = v13.get_noc_addr((int32_t) ((uint32_t) v17 / (uint32_t) v11), v3);
    noc_async_read(temp_91, v18, v11);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(0), v1);
    cb_reserve_back(get_compile_time_arg_val(1), v1);
    int32_t v19 = get_write_ptr(get_compile_time_arg_val(1));
    uint64_t temp_100 = v9.get_noc_addr((int32_t) ((uint32_t) v17 / (uint32_t) v7), v3);
    noc_async_read(temp_100, v19, v7);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(1), v1);
  }
  return;
}
