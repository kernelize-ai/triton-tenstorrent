// add_kernel__reader
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = get_common_arg_val<uint32_t>(0);
  int32_t v5 = get_common_arg_val<uint32_t>(1);
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  DataFormat v6 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v7 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v8;
  v8.bank_base_address = v5;
  v8.page_size = v7;
  v8.data_format = v6;
  InterleavedAddrGenFast<true> v9 = v8;
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  DataFormat v10 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v11 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v12;
  v12.bank_base_address = v4;
  v12.page_size = v11;
  v12.data_format = v10;
  InterleavedAddrGenFast<true> v13 = v12;
  int32_t v14 = get_common_arg_val<uint32_t>(4);
  int32_t v15 = get_common_arg_val<uint32_t>(6);
  int32_t v16 = get_common_arg_val<uint32_t>(7);
  int32_t v17 = get_common_arg_val<uint32_t>(5);
  int32_t v18 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_x())) * (uint32_t) v14)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_y())) * (uint32_t) v17)));
  for (int32_t i19 = v18; i19 < ((int32_t) ((uint32_t) v18 + (uint32_t) v14) < (int32_t) ((uint32_t) v15 * (uint32_t) v16) ? (int32_t) ((uint32_t) v18 + (uint32_t) v14) : (int32_t) ((uint32_t) v15 * (uint32_t) v16)); i19 += v1) {
    int32_t v20 = (int32_t) ((uint32_t) i19 * (uint32_t) 2048);
    cb_ctarg_0.reserve_back(v1);
    uint64_t temp_166 = v13.get_noc_addr((int32_t) ((uint32_t) v20 / (uint32_t) v11), v3);
    noc_async_read(temp_166, cb_ctarg_0.get_write_ptr(), v11);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_ctarg_0.push_back(v1);
    cb_ctarg_1.reserve_back(v1);
    uint64_t temp_175 = v9.get_noc_addr((int32_t) ((uint32_t) v20 / (uint32_t) v7), v3);
    noc_async_read(temp_175, cb_ctarg_1.get_write_ptr(), v7);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_ctarg_1.push_back(v1);
  }
  return;
}
