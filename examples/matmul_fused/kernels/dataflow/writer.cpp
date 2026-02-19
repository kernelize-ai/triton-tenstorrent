// matmul_kernel_fused__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  size_t v2 = 0;
  int32_t v3 = 63;
  int32_t v4 = 0;
  int32_t v5 = 64;
  int32_t v6 = 32;
  int32_t v7 = 1;
  bool v8 = true;
  int32_t v9 = 2;
  int32_t v10 = 2048;
  int32_t v11 = get_common_arg_val<uint32_t>(v2);
  int32_t v12 = get_common_arg_val<uint32_t>(2);
  int32_t v13 = get_common_arg_val<uint32_t>(20);
  int32_t v14 = get_common_arg_val<uint32_t>(22);
  int32_t v15 = get_common_arg_val<uint32_t>(40);
  int32_t v16 = get_common_arg_val<uint32_t>(41);
  int32_t v17 = get_common_arg_val<uint32_t>(42);
  DataFormat v18 = get_dataformat(get_compile_time_arg_val(3));
  int32_t v19 = get_tile_size(get_compile_time_arg_val(3));
  InterleavedAddrGenFast<true> v20;
  v20.bank_base_address = v13;
  v20.page_size = v19;
  v20.data_format = v18;
  InterleavedAddrGenFast<true> v21 = v20;
  DataFormat v22 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v23 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v24;
  v24.bank_base_address = v11;
  v24.page_size = v23;
  v24.data_format = v22;
  InterleavedAddrGenFast<true> v25 = v24;
  int32_t v26 = get_arg_val<uint32_t>(1);
  int32_t v27 = get_arg_val<uint32_t>(v2);
  int32_t v28 = (int32_t) ((uint32_t) v16 + (uint32_t) v3) / v5;
  for (int32_t i29 = v27; i29 < v26; i29 += v7) {
    int32_t v30 = i29 / v28;
    int32_t v31 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v6) - (uint32_t) v30) < v7 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v6) - (uint32_t) v30) : v7;
    int32_t v32 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v30 + (uint32_t) (i29 % v31))) * (uint32_t) v6) / v6;
    for (int32_t j33 = v4; j33 < ((int32_t) ((uint32_t) v17 + (uint32_t) v3) / v5); j33 += v7) {
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v34 = get_write_ptr(get_compile_time_arg_val(0));
      int32_t v35 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v32 * (uint32_t) (v12 != (int32_t) ((uint32_t) (v12 / v6) * (uint32_t) v6) & v12 < v4 == v1 ? (int32_t) ((uint32_t) (v12 / v6) + (uint32_t) v7) : v12 / v6))) + (uint32_t) ((int32_t) ((uint32_t) j33 * (uint32_t) v5) / v6));
      uint64_t temp_216 = v25.get_noc_addr(v35, v4);
      noc_async_read(temp_216, v34, v23);
      int32_t v36 = (int32_t) ((uint32_t) v34 + (uint32_t) v10);
      uint64_t temp_228 = v25.get_noc_addr((int32_t) ((uint32_t) v35 + (uint32_t) v7), v4);
      noc_async_read(temp_228, v36, v23);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(3), v9);
    }
    int32_t v37 = get_read_ptr(get_compile_time_arg_val(3));
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v32 * (uint32_t) (v14 != (int32_t) ((uint32_t) (v14 / v6) * (uint32_t) v6) & v14 < v4 == v1 ? (int32_t) ((uint32_t) (v14 / v6) + (uint32_t) v7) : v14 / v6))) + (uint32_t) ((int32_t) ((uint32_t) ((i29 % v28) / v31) * (uint32_t) v5) / v6));
    uint64_t temp_208 = v21.get_noc_addr(v38, v4);
    noc_async_write(v37, temp_208, v19);
    int32_t v39 = (int32_t) ((uint32_t) v37 + (uint32_t) v10);
    uint64_t temp_220 = v21.get_noc_addr((int32_t) ((uint32_t) v38 + (uint32_t) v7), v4);
    noc_async_write(v39, temp_220, v19);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(3), v9);
  }
  return;
}
