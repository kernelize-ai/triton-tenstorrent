// matmul_kernel_fused__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  int32_t v2 = 63;
  int32_t v3 = 0;
  int32_t v4 = 64;
  int32_t v5 = 32;
  int32_t v6 = 1;
  bool v7 = true;
  int32_t v8 = 4;
  int32_t v9 = 2048;
  int32_t v10 = 2;
  int32_t v11 = get_common_arg_val<uint32_t>(10);
  int32_t v12 = get_common_arg_val<uint32_t>(12);
  int32_t v13 = get_common_arg_val<uint32_t>(30);
  int32_t v14 = get_common_arg_val<uint32_t>(32);
  int32_t v15 = get_common_arg_val<uint32_t>(40);
  int32_t v16 = get_common_arg_val<uint32_t>(41);
  int32_t v17 = get_common_arg_val<uint32_t>(42);
  DataFormat v18 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v19 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v20;
  v20.bank_base_address = v13;
  v20.page_size = v19;
  v20.data_format = v18;
  InterleavedAddrGenFast<true> v21 = v20;
  DataFormat v22 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v23 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v24;
  v24.bank_base_address = v11;
  v24.page_size = v23;
  v24.data_format = v22;
  InterleavedAddrGenFast<true> v25 = v24;
  int32_t v26 = get_arg_val<uint32_t>(1);
  int32_t v27 = get_arg_val<uint32_t>(0);
  int32_t v28 = (int32_t) ((uint32_t) v16 + (uint32_t) v2) / v4;
  int32_t v29 = v12 != (int32_t) ((uint32_t) (v12 / v5) * (uint32_t) v5) & v12 < v3 == v1 ? (int32_t) ((uint32_t) (v12 / v5) + (uint32_t) v6) : v12 / v5;
  for (int32_t i30 = v27; i30 < v26; i30 += v6) {
    int32_t v31 = i30 / v28;
    int32_t v32 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v5) - (uint32_t) v31) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v5) - (uint32_t) v31) : v6;
    int32_t v33 = (int32_t) ((uint32_t) ((i30 % v28) / v32) * (uint32_t) v4) / v5;
    for (int32_t j34 = v3; j34 < ((int32_t) ((uint32_t) v17 + (uint32_t) v2) / v4); j34 += v6) {
      cb_reserve_back(get_compile_time_arg_val(1), v8);
      int32_t v35 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j34 * (uint32_t) v4) / v5) * (uint32_t) v29)) + (uint32_t) v33);
      uint64_t temp_335 = v25.get_noc_addr(v36, v3);
      noc_async_read(temp_335, v35, v23);
      int32_t v37 = (int32_t) ((uint32_t) v35 + (uint32_t) 4096);
      int32_t v38 = (int32_t) ((uint32_t) v36 + (uint32_t) v6);
      uint64_t temp_347 = v25.get_noc_addr(v38, v3);
      noc_async_read(temp_347, v37, v23);
      int32_t v39 = (int32_t) ((uint32_t) v35 + (uint32_t) v9);
      uint64_t temp_359 = v25.get_noc_addr((int32_t) ((uint32_t) v36 + (uint32_t) v29), v3);
      noc_async_read(temp_359, v39, v23);
      int32_t v40 = (int32_t) ((uint32_t) v35 + (uint32_t) 6144);
      uint64_t temp_371 = v25.get_noc_addr((int32_t) ((uint32_t) v38 + (uint32_t) v29), v3);
      noc_async_read(temp_371, v40, v23);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v8);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v10);
    int32_t v41 = get_write_ptr(get_compile_time_arg_val(2));
    int32_t v42 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v31 + (uint32_t) (i30 % v32))) * (uint32_t) v5) / v5) * (uint32_t) (v14 != (int32_t) ((uint32_t) (v14 / v5) * (uint32_t) v5) & v14 < v3 == v1 ? (int32_t) ((uint32_t) (v14 / v5) + (uint32_t) v6) : v14 / v5))) + (uint32_t) v33);
    uint64_t temp_322 = v21.get_noc_addr(v42, v3);
    noc_async_read(temp_322, v41, v19);
    int32_t v43 = (int32_t) ((uint32_t) v41 + (uint32_t) v9);
    uint64_t temp_334 = v21.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v6), v3);
    noc_async_read(temp_334, v43, v19);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v10);
  }
  return;
}
