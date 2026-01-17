// matmul_kernel_tma__reader
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
  int32_t v8 = 2;
  int32_t v9 = 4;
  int32_t v10 = get_arg_val<uint32_t>(0);
  int32_t v11 = get_arg_val<uint32_t>(2);
  int32_t v12 = get_arg_val<uint32_t>(10);
  int32_t v13 = get_arg_val<uint32_t>(12);
  int32_t v14 = get_arg_val<uint32_t>(30);
  int32_t v15 = get_arg_val<uint32_t>(31);
  int32_t v16 = get_arg_val<uint32_t>(32);
  DataFormat v17 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v18 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v19;
  v19.bank_base_address = v12;
  v19.page_size = v18;
  v19.data_format = v17;
  InterleavedAddrGenFast<true> v20 = v19;
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v10;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  int32_t v25 = get_arg_val<uint32_t>(34);
  int32_t v26 = get_arg_val<uint32_t>(33);
  int32_t v27 = (int32_t) ((uint32_t) v15 + (uint32_t) v2) / v4;
  int32_t v28 = v13 != (int32_t) ((uint32_t) (v13 / v5) * (uint32_t) v5) & v13 < v3 == v1 ? (int32_t) ((uint32_t) (v13 / v5) + (uint32_t) v6) : v13 / v5;
  for (int32_t i29 = v26; i29 < v25; i29 += v6) {
    int32_t v30 = i29 / v27;
    int32_t v31 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) 31) / v5) - (uint32_t) v30) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) 31) / v5) - (uint32_t) v30) : v6;
    int32_t v32 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v30 + (uint32_t) (i29 % v31))) * (uint32_t) v5) / v5) * (uint32_t) (v11 != (int32_t) ((uint32_t) (v11 / v5) * (uint32_t) v5) & v11 < v3 == v1 ? (int32_t) ((uint32_t) (v11 / v5) + (uint32_t) v6) : v11 / v5));
    int32_t v33 = (int32_t) ((uint32_t) ((i29 % v27) / v31) * (uint32_t) v4) / v5;
    int32_t v34 = (int32_t) ((uint32_t) v33 + (uint32_t) v6);
    for (int32_t j35 = v3; j35 < ((int32_t) ((uint32_t) v16 + (uint32_t) v2) / v4); j35 += v6) {
      int32_t v36 = (int32_t) ((uint32_t) j35 * (uint32_t) v4) / v5;
      cb_reserve_back(get_compile_time_arg_val(0), v8);
      int32_t v37 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_311 = v24.get_noc_addr((int32_t) ((uint32_t) v32 + (uint32_t) v36), v3);
      noc_async_read(temp_311, v37, v22);
      int32_t v38 = (int32_t) ((uint32_t) v36 + (uint32_t) v6);
      int32_t v39 = (int32_t) ((uint32_t) v37 + (uint32_t) v22);
      uint64_t temp_328 = v24.get_noc_addr((int32_t) ((uint32_t) v32 + (uint32_t) v38), v3);
      noc_async_read(temp_328, v39, v22);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v8);
      cb_reserve_back(get_compile_time_arg_val(1), v9);
      int32_t v40 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v41 = (int32_t) ((uint32_t) v36 * (uint32_t) v28);
      uint64_t temp_342 = v20.get_noc_addr((int32_t) ((uint32_t) v41 + (uint32_t) v33), v3);
      noc_async_read(temp_342, v40, v18);
      int32_t v42 = (int32_t) ((uint32_t) v38 * (uint32_t) v28);
      int32_t v43 = (int32_t) ((uint32_t) v40 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v8)));
      uint64_t temp_359 = v20.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v33), v3);
      noc_async_read(temp_359, v43, v18);
      int32_t v44 = (int32_t) ((uint32_t) v40 + (uint32_t) v18);
      uint64_t temp_371 = v20.get_noc_addr((int32_t) ((uint32_t) v41 + (uint32_t) v34), v3);
      noc_async_read(temp_371, v44, v18);
      int32_t v45 = (int32_t) ((uint32_t) v40 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) 3)));
      uint64_t temp_383 = v20.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v34), v3);
      noc_async_read(temp_383, v45, v18);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v9);
    }
  }
  return;
}
