// matmul_kernel_fused__reader
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
  int32_t v11 = 4;
  int32_t v12 = get_common_arg_val<uint32_t>(v2);
  int32_t v13 = get_common_arg_val<uint32_t>(2);
  int32_t v14 = get_common_arg_val<uint32_t>(10);
  int32_t v15 = get_common_arg_val<uint32_t>(12);
  int32_t v16 = get_common_arg_val<uint32_t>(30);
  int32_t v17 = get_common_arg_val<uint32_t>(32);
  int32_t v18 = get_common_arg_val<uint32_t>(40);
  int32_t v19 = get_common_arg_val<uint32_t>(41);
  int32_t v20 = get_common_arg_val<uint32_t>(42);
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v16;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  DataFormat v25 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v26 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v27;
  v27.bank_base_address = v14;
  v27.page_size = v26;
  v27.data_format = v25;
  InterleavedAddrGenFast<true> v28 = v27;
  DataFormat v29 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v30 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v31;
  v31.bank_base_address = v12;
  v31.page_size = v30;
  v31.data_format = v29;
  InterleavedAddrGenFast<true> v32 = v31;
  int32_t v33 = get_arg_val<uint32_t>(1);
  int32_t v34 = get_arg_val<uint32_t>(v2);
  int32_t v35 = (int32_t) ((uint32_t) v19 + (uint32_t) v3) / v5;
  int32_t v36 = v15 != (int32_t) ((uint32_t) (v15 / v6) * (uint32_t) v6) & v15 < v4 == v1 ? (int32_t) ((uint32_t) (v15 / v6) + (uint32_t) v7) : v15 / v6;
  for (int32_t i37 = v34; i37 < v33; i37 += v7) {
    int32_t v38 = i37 / v35;
    int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) 31) / v6) - (uint32_t) v38) < v7 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) 31) / v6) - (uint32_t) v38) : v7;
    int32_t v40 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v38 + (uint32_t) (i37 % v39))) * (uint32_t) v6) / v6;
    int32_t v41 = (int32_t) ((uint32_t) ((i37 % v35) / v39) * (uint32_t) v5) / v6;
    for (int32_t j42 = v4; j42 < ((int32_t) ((uint32_t) v20 + (uint32_t) v3) / v5); j42 += v7) {
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v43 = get_write_ptr(get_compile_time_arg_val(0));
      int32_t v44 = (int32_t) ((uint32_t) j42 * (uint32_t) v5) / v6;
      int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v40 * (uint32_t) (v13 != (int32_t) ((uint32_t) (v13 / v6) * (uint32_t) v6) & v13 < v4 == v1 ? (int32_t) ((uint32_t) (v13 / v6) + (uint32_t) v7) : v13 / v6))) + (uint32_t) v44);
      uint64_t temp_352 = v32.get_noc_addr(v45, v4);
      noc_async_read(temp_352, v43, v30);
      int32_t v46 = (int32_t) ((uint32_t) v43 + (uint32_t) v10);
      uint64_t temp_364 = v32.get_noc_addr((int32_t) ((uint32_t) v45 + (uint32_t) v7), v4);
      noc_async_read(temp_364, v46, v30);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
      cb_reserve_back(get_compile_time_arg_val(1), v11);
      int32_t v47 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v48 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v44 * (uint32_t) v36)) + (uint32_t) v41);
      uint64_t temp_378 = v28.get_noc_addr(v48, v4);
      noc_async_read(temp_378, v47, v26);
      int32_t v49 = (int32_t) ((uint32_t) v47 + (uint32_t) 4096);
      int32_t v50 = (int32_t) ((uint32_t) v48 + (uint32_t) v7);
      uint64_t temp_390 = v28.get_noc_addr(v50, v4);
      noc_async_read(temp_390, v49, v26);
      int32_t v51 = (int32_t) ((uint32_t) v47 + (uint32_t) v10);
      uint64_t temp_402 = v28.get_noc_addr((int32_t) ((uint32_t) v48 + (uint32_t) v36), v4);
      noc_async_read(temp_402, v51, v26);
      int32_t v52 = (int32_t) ((uint32_t) v47 + (uint32_t) 6144);
      uint64_t temp_414 = v28.get_noc_addr((int32_t) ((uint32_t) v50 + (uint32_t) v36), v4);
      noc_async_read(temp_414, v52, v26);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v11);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v9);
    int32_t v53 = get_write_ptr(get_compile_time_arg_val(2));
    int32_t v54 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v40 * (uint32_t) (v17 != (int32_t) ((uint32_t) (v17 / v6) * (uint32_t) v6) & v17 < v4 == v1 ? (int32_t) ((uint32_t) (v17 / v6) + (uint32_t) v7) : v17 / v6))) + (uint32_t) v41);
    uint64_t temp_343 = v24.get_noc_addr(v54, v4);
    noc_async_read(temp_343, v53, v22);
    int32_t v55 = (int32_t) ((uint32_t) v53 + (uint32_t) v10);
    uint64_t temp_355 = v24.get_noc_addr((int32_t) ((uint32_t) v54 + (uint32_t) v7), v4);
    noc_async_read(temp_355, v55, v22);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v9);
  }
  return;
}

