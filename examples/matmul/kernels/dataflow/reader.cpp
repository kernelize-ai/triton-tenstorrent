// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  size_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = 32;
  int32_t v5 = 1;
  int32_t v6 = 31;
  int32_t v7 = get_common_arg_val<uint32_t>(v1);
  int32_t v8 = get_common_arg_val<uint32_t>(3);
  int32_t v9 = get_common_arg_val<uint32_t>(4);
  int32_t v10 = get_common_arg_val<uint32_t>(5);
  DataFormat v11 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v12 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v13;
  v13.bank_base_address = v7;
  v13.page_size = v12;
  v13.data_format = v11;
  InterleavedAddrGenFast<true> v14 = v13;
  int32_t v15 = get_arg_val<uint32_t>(v1);
  int32_t v16 = get_arg_val<uint32_t>(0);
  for (int32_t i17 = v16; i17 < v15; i17 += v5) {
    for (int32_t j18 = v3; j18 < ((int32_t) ((uint32_t) v10 + (uint32_t) v6) / v4); j18 += v5) {
      cb_reserve_back(get_compile_time_arg_val(1), v5);
      int32_t v19 = get_write_ptr(get_compile_time_arg_val(1));
      uint64_t temp_233 = v14.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j18 * (uint32_t) v4) / v4) * (uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4))) + (uint32_t) (((int32_t) ((uint32_t) ((i17 % ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v6) / v4) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4))) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v6) / v4) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4))) : v5)) * (uint32_t) v4) % v9) / v4))) * (uint32_t) 1024)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j18 * (uint32_t) v4) % v4) * (uint32_t) v4)) + (uint32_t) (((int32_t) ((uint32_t) ((i17 % ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v6) / v4) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4))) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v6) / v4) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v9 + (uint32_t) v6) / v4))) : v5)) * (uint32_t) v4) % v9) % v4))))) * (uint32_t) 2)) / (uint32_t) v12), v3);
      noc_async_read(temp_233, v19, v12);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v5);
    }
  }
  return;
}
