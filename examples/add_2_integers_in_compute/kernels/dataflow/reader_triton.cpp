// add_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  size_t v1 = 0;
  size_t v2 = 1;
  int32_t v3 = 1;
  bool v4 = true;
  int32_t v5 = 0;
  int32_t v6 = get_common_arg_val<uint32_t>(v1);
  int32_t v7 = get_common_arg_val<uint32_t>(v2);
  DataFormat v8 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v9 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v10;
  v10.bank_base_address = v7;
  v10.page_size = v9;
  v10.data_format = v8;
  InterleavedAddrGenFast<true> v11 = v10;
  DataFormat v12 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v13 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v14;
  v14.bank_base_address = v6;
  v14.page_size = v13;
  v14.data_format = v12;
  InterleavedAddrGenFast<true> v15 = v14;
  int32_t v16 = get_arg_val<uint32_t>(v2);
  int32_t v17 = get_arg_val<uint32_t>(v1);
  for (int32_t i18 = v17; i18 < v16; i18 += v3) {
    int32_t v19 = (int32_t) ((uint32_t) i18 * (uint32_t) 2048);
    cb_reserve_back(get_compile_time_arg_val(0), v3);
    int32_t v20 = get_write_ptr(get_compile_time_arg_val(0));
    uint64_t temp_85 = v15.get_noc_addr((int32_t) ((uint32_t) v19 / (uint32_t) v13), v5);
    noc_async_read(temp_85, v20, v13);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(0), v3);
    cb_reserve_back(get_compile_time_arg_val(1), v3);
    int32_t v21 = get_write_ptr(get_compile_time_arg_val(1));
    uint64_t temp_94 = v11.get_noc_addr((int32_t) ((uint32_t) v19 / (uint32_t) v9), v5);
    noc_async_read(temp_94, v21, v9);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(1), v3);
  }
  return;
}
