// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
void kernel_main() {
  bool v1 = true;
  int32_t v2 = 0;
  int32_t v3 = 1;
  int32_t v4 = 32;
  int32_t v5 = 31;
  int32_t v6 = 1024;
  int32_t v7 = 2;
  int32_t v8 = get_arg_val<uint32_t>(0);
  int32_t v9 = get_arg_val<uint32_t>(1);
  int32_t v10 = get_arg_val<uint32_t>(2);
  int32_t v11 = get_arg_val<uint32_t>(3);
  int32_t v12 = get_arg_val<uint32_t>(4);
  int32_t v13 = get_arg_val<uint32_t>(5);
  int32_t v14 = get_arg_val<uint32_t>(6);
  int32_t v15 = get_arg_val<uint32_t>(7);
  int32_t v16 = get_arg_val<uint32_t>(8);
  DataFormat v17 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v18 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v19;
  v19.bank_base_address = v9;
  v19.page_size = v18;
  v19.data_format = v17;
  InterleavedAddrGenFast<true> v20 = v19;
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v8;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  int32_t v25 = (int32_t) ((uint32_t) v12 + (uint32_t) v5) / v4;
  int32_t v26 = (int32_t) ((uint32_t) v13 + (uint32_t) v5) / v4;
  int32_t v27 = get_arg_val<uint32_t>(9);
  int32_t v28 = v27 / v25;
  int32_t v29 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v4) - (uint32_t) v28) < v3 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v4) - (uint32_t) v28) : v3;
  int32_t v30 = v27 % v25;
  for (int32_t i31 = v2; i31 < v26; i31 += v3) {
    int32_t v32 = (int32_t) ((uint32_t) i31 * (uint32_t) v4);
    int32_t v33 = v32 / v4;
    int32_t v34 = v32 % v4;
    uint64_t temp_282 = v24.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) (v30 % v29))) * (uint32_t) v4) % v11) / v4) * (uint32_t) v26)) + (uint32_t) v33)) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) (v30 % v29))) * (uint32_t) v4) % v11) % v4) * (uint32_t) v4)) + (uint32_t) v34)))) * (uint32_t) v7)) / (uint32_t) v22), v2);
    cb_reserve_back(get_compile_time_arg_val(0), v3);
    int32_t v35 = get_write_ptr(get_compile_time_arg_val(0));
    noc_async_read(temp_282, v35, v22);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(0), v3);
    uint64_t temp_326 = v20.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v33 * (uint32_t) v25)) + (uint32_t) (((int32_t) ((uint32_t) (v30 / v29) * (uint32_t) v4) % v12) / v4))) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 * (uint32_t) v4)) + (uint32_t) (((int32_t) ((uint32_t) (v30 / v29) * (uint32_t) v4) % v12) % v4))))) * (uint32_t) v7)) / (uint32_t) v18), v2);
    cb_reserve_back(get_compile_time_arg_val(1), v3);
    int32_t v36 = get_write_ptr(get_compile_time_arg_val(1));
    noc_async_read(temp_326, v36, v18);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(1), v3);
  }
  return;
}
