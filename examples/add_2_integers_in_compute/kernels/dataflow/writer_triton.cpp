// add_kernel__writer
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = get_common_arg_val<uint32_t>(2);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  DataFormat v4 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v5 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v6;
  v6.bank_base_address = v3;
  v6.page_size = v5;
  v6.data_format = v4;
  InterleavedAddrGenFast<true> v7 = v6;
  int32_t v8 = get_common_arg_val<uint32_t>(4);
  int32_t v9 = get_common_arg_val<uint32_t>(6);
  int32_t v10 = get_common_arg_val<uint32_t>(7);
  int32_t v11 = get_common_arg_val<uint32_t>(5);
  int32_t v12 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_x())) * (uint32_t) v8)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) get_absolute_logical_y())) * (uint32_t) v11)));
  for (int32_t i13 = v12; i13 < ((int32_t) ((uint32_t) v12 + (uint32_t) v8) < (int32_t) ((uint32_t) v9 * (uint32_t) v10) ? (int32_t) ((uint32_t) v12 + (uint32_t) v8) : (int32_t) ((uint32_t) v9 * (uint32_t) v10)); i13 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_ctarg_2.wait_front(v1);
    }
    uint64_t temp_91 = v7.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) i13 * (uint32_t) 2048)) / (uint32_t) v5), 0);
    noc_async_write(cb_ctarg_2.get_read_ptr(), temp_91, v5);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_ctarg_2.pop_front(v1);
  }
  return;
}
