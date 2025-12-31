// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = true;
  int32_t v2 = 0;
  int32_t v3 = 32;
  int32_t v4 = 1;
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
  int32_t v25 = get_arg_val<uint32_t>(10);
  int32_t v26 = get_arg_val<uint32_t>(9);
  int32_t v27 = (int32_t) ((uint32_t) v12 + (uint32_t) v5) / v3;
  int32_t v28 = (int32_t) ((uint32_t) v13 + (uint32_t) v5) / v3;
  for (int32_t i29 = v26; i29 < v25; i29 += v4) {
    int32_t v30 = i29 / v27;
    int32_t v31 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v3) - (uint32_t) v30) < v4 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v3) - (uint32_t) v30) : v4;
    int32_t v32 = i29 % v27;
    for (int32_t j33 = v2; j33 < v28; j33 += v4) {
      int32_t v34 = (int32_t) ((uint32_t) j33 * (uint32_t) v3);
      int32_t v35 = v34 / v3;
      int32_t v36 = v34 % v3;
      uint64_t temp_290 = v24.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v30 + (uint32_t) (v32 % v31))) * (uint32_t) v3) % v11) / v3) * (uint32_t) v28)) + (uint32_t) v35)) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v30 + (uint32_t) (v32 % v31))) * (uint32_t) v3) % v11) % v3) * (uint32_t) v3)) + (uint32_t) v36)))) * (uint32_t) v7)) / (uint32_t) v22), v2);
      cb_reserve_back(get_compile_time_arg_val(0), v4);
      int32_t v37 = get_write_ptr(get_compile_time_arg_val(0));
      noc_async_read(temp_290, v37, v22);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v4);
      uint64_t temp_334 = v20.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 * (uint32_t) v27)) + (uint32_t) (((int32_t) ((uint32_t) (v32 / v31) * (uint32_t) v3) % v12) / v3))) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v36 * (uint32_t) v3)) + (uint32_t) (((int32_t) ((uint32_t) (v32 / v31) * (uint32_t) v3) % v12) % v3))))) * (uint32_t) v7)) / (uint32_t) v18), v2);
      cb_reserve_back(get_compile_time_arg_val(1), v4);
      int32_t v38 = get_write_ptr(get_compile_time_arg_val(1));
      noc_async_read(temp_334, v38, v18);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v4);
    }
  }
  return;
}
