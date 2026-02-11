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
  int32_t v5 = 1;
  int32_t v6 = 4;
  bool v7 = true;
  int32_t v8 = 32;
  int32_t v9 = 2048;
  int32_t v10 = 4096;
  int32_t v11 = 6144;
  int32_t v12 = get_common_arg_val<uint32_t>(10);
  int32_t v13 = get_common_arg_val<uint32_t>(12);
  int32_t v14 = get_common_arg_val<uint32_t>(30);
  int32_t v15 = get_common_arg_val<uint32_t>(32);
  int32_t v16 = get_common_arg_val<uint32_t>(40);
  int32_t v17 = get_common_arg_val<uint32_t>(41);
  int32_t v18 = get_common_arg_val<uint32_t>(42);
  DataFormat v19 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v20 = get_tile_size(get_compile_time_arg_val(2));
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v23 = get_arg_val<uint32_t>(1);
  int32_t v24 = get_arg_val<uint32_t>(0);
  int32_t v25 = (int32_t) ((uint32_t) v17 + (uint32_t) v2) / v4;
  int32_t v26 = v13 != (int32_t) ((uint32_t) (v13 / v8) * (uint32_t) v8) & v13 < v3 == v1 ? (int32_t) ((uint32_t) (v13 / v8) + (uint32_t) v5) : v13 / v8;
  int32_t v27 = v15 != (int32_t) ((uint32_t) (v15 / v8) * (uint32_t) v8) & v15 < v3 == v1 ? (int32_t) ((uint32_t) (v15 / v8) + (uint32_t) v5) : v15 / v8;
  for (int32_t i28 = v24; i28 < v23; i28 += v5) {
    int32_t v29 = i28 / v25;
    int32_t v30 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v16 + (uint32_t) v2) / v4) - (uint32_t) v29) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v16 + (uint32_t) v2) / v4) - (uint32_t) v29) : v5;
    int32_t v31 = (int32_t) ((uint32_t) ((i28 % v25) / v30) * (uint32_t) v4) / v8;
    for (int32_t j32 = v3; j32 < ((int32_t) ((uint32_t) v18 + (uint32_t) v2) / v4); j32 += v5) {
      cb_reserve_back(get_compile_time_arg_val(1), v6);
      InterleavedAddrGenFast<true> v33;
      v33.bank_base_address = v12;
      v33.page_size = v22;
      v33.data_format = v21;
      InterleavedAddrGenFast<true> v34 = v33;
      int32_t v35 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j32 * (uint32_t) v4) / v8) * (uint32_t) v26)) + (uint32_t) v31);
      uint64_t temp_380 = v34.get_noc_addr(v36, v3);
      noc_async_read(temp_380, v35, v22);
      int32_t v37 = (int32_t) ((uint32_t) v35 + (uint32_t) v9);
      int32_t v38 = (int32_t) ((uint32_t) v36 + (uint32_t) v5);
      uint64_t temp_392 = v34.get_noc_addr(v38, v3);
      noc_async_read(temp_392, v37, v22);
      int32_t v39 = (int32_t) ((uint32_t) v35 + (uint32_t) v10);
      uint64_t temp_404 = v34.get_noc_addr((int32_t) ((uint32_t) v36 + (uint32_t) v26), v3);
      noc_async_read(temp_404, v39, v22);
      int32_t v40 = (int32_t) ((uint32_t) v35 + (uint32_t) v11);
      uint64_t temp_416 = v34.get_noc_addr((int32_t) ((uint32_t) v38 + (uint32_t) v26), v3);
      noc_async_read(temp_416, v40, v22);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v6);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v6);
    InterleavedAddrGenFast<true> v41;
    v41.bank_base_address = v14;
    v41.page_size = v20;
    v41.data_format = v19;
    InterleavedAddrGenFast<true> v42 = v41;
    int32_t v43 = get_write_ptr(get_compile_time_arg_val(2));
    int32_t v44 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v29 + (uint32_t) (i28 % v30))) * (uint32_t) v4) / v8) * (uint32_t) v27)) + (uint32_t) v31);
    uint64_t temp_360 = v42.get_noc_addr(v44, v3);
    noc_async_read(temp_360, v43, v20);
    int32_t v45 = (int32_t) ((uint32_t) v43 + (uint32_t) v9);
    int32_t v46 = (int32_t) ((uint32_t) v44 + (uint32_t) v5);
    uint64_t temp_372 = v42.get_noc_addr(v46, v3);
    noc_async_read(temp_372, v45, v20);
    int32_t v47 = (int32_t) ((uint32_t) v43 + (uint32_t) v10);
    uint64_t temp_384 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v27), v3);
    noc_async_read(temp_384, v47, v20);
    int32_t v48 = (int32_t) ((uint32_t) v43 + (uint32_t) v11);
    uint64_t temp_396 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v27), v3);
    noc_async_read(temp_396, v48, v20);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v6);
  }
  return;
}

