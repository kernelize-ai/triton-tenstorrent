// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  size_t v1 = 0;
  size_t v2 = 1;
  bool v3 = true;
  int32_t v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int32_t v7 = 31;
  int32_t v8 = 1024;
  int32_t v9 = 2;
  int32_t v10 = get_common_arg_val<uint32_t>(v1);
  int32_t v11 = get_common_arg_val<uint32_t>(v2);
  int32_t v12 = get_common_arg_val<uint32_t>(3);
  int32_t v13 = get_common_arg_val<uint32_t>(4);
  int32_t v14 = get_common_arg_val<uint32_t>(5);
  DataFormat v15 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v16 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v17;
  v17.bank_base_address = v11;
  v17.page_size = v16;
  v17.data_format = v15;
  InterleavedAddrGenFast<true> v18 = v17;
  DataFormat v19 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v20 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v21;
  v21.bank_base_address = v10;
  v21.page_size = v20;
  v21.data_format = v19;
  InterleavedAddrGenFast<true> v22 = v21;
  int32_t v23 = get_arg_val<uint32_t>(v2);
  int32_t v24 = get_arg_val<uint32_t>(v1);
  int32_t v25 = (int32_t) ((uint32_t) v13 + (uint32_t) v7) / v5;
  int32_t v26 = (int32_t) ((uint32_t) v14 + (uint32_t) v7) / v5;
  for (int32_t i27 = v24; i27 < v23; i27 += v6) {
    int32_t v28 = i27 / v25;
    int32_t v29 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) v7) / v5) - (uint32_t) v28) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) v7) / v5) - (uint32_t) v28) : v6;
    int32_t v30 = i27 % v25;
    for (int32_t j31 = v4; j31 < v26; j31 += v6) {
      int32_t v32 = (int32_t) ((uint32_t) j31 * (uint32_t) v5);
      int32_t v33 = v32 / v5;
      int32_t v34 = v32 % v5;
      cb_reserve_back(get_compile_time_arg_val(0), v6);
      int32_t v35 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_279 = v22.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) (v30 % v29))) * (uint32_t) v5) % v12) / v5) * (uint32_t) v26)) + (uint32_t) v33)) * (uint32_t) v8)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) (v30 % v29))) * (uint32_t) v5) % v12) % v5) * (uint32_t) v5)) + (uint32_t) v34)))) * (uint32_t) v9)) / (uint32_t) v20), v4);
      noc_async_read(temp_279, v35, v20);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v6);
      cb_reserve_back(get_compile_time_arg_val(1), v6);
      int32_t v36 = get_write_ptr(get_compile_time_arg_val(1));
      uint64_t temp_288 = v18.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v33 * (uint32_t) v25)) + (uint32_t) (((int32_t) ((uint32_t) (v30 / v29) * (uint32_t) v5) % v13) / v5))) * (uint32_t) v8)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 * (uint32_t) v5)) + (uint32_t) (((int32_t) ((uint32_t) (v30 / v29) * (uint32_t) v5) % v13) % v5))))) * (uint32_t) v9)) / (uint32_t) v16), v4);
      noc_async_read(temp_288, v36, v16);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v6);
    }
  }
  return;
}
