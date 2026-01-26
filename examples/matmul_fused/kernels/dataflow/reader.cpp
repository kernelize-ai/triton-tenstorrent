// matmul_kernel_tma__reader
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
  int32_t v10 = 4;
  int32_t v11 = get_common_arg_val<uint32_t>(v2);
  int32_t v12 = get_common_arg_val<uint32_t>(2);
  int32_t v13 = get_common_arg_val<uint32_t>(10);
  int32_t v14 = get_common_arg_val<uint32_t>(12);
  int32_t v15 = get_common_arg_val<uint32_t>(30);
  int32_t v16 = get_common_arg_val<uint32_t>(32);
  int32_t v17 = get_common_arg_val<uint32_t>(40);
  int32_t v18 = get_common_arg_val<uint32_t>(41);
  int32_t v19 = get_common_arg_val<uint32_t>(42);
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v22;
  v22.bank_base_address = v15;
  v22.page_size = v21;
  v22.data_format = v20;
  InterleavedAddrGenFast<true> v23 = v22;
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v26;
  v26.bank_base_address = v13;
  v26.page_size = v25;
  v26.data_format = v24;
  InterleavedAddrGenFast<true> v27 = v26;
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v30;
  v30.bank_base_address = v11;
  v30.page_size = v29;
  v30.data_format = v28;
  InterleavedAddrGenFast<true> v31 = v30;
  int32_t v32 = get_arg_val<uint32_t>(1);
  int32_t v33 = get_arg_val<uint32_t>(v2);
  int32_t v34 = (int32_t) ((uint32_t) v18 + (uint32_t) v3) / v5;
  int32_t v35 = v14 != (int32_t) ((uint32_t) (v14 / v6) * (uint32_t) v6) & v14 < v4 == v1 ? (int32_t) ((uint32_t) (v14 / v6) + (uint32_t) v7) : v14 / v6;
  for (int32_t i36 = v33; i36 < v32; i36 += v7) {
    int32_t v37 = i36 / v34;
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) 31) / v6) - (uint32_t) v37) < v7 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) 31) / v6) - (uint32_t) v37) : v7;
    int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v37 + (uint32_t) (i36 % v38))) * (uint32_t) v6) / v6;
    int32_t v40 = (int32_t) ((uint32_t) v39 * (uint32_t) (v12 != (int32_t) ((uint32_t) (v12 / v6) * (uint32_t) v6) & v12 < v4 == v1 ? (int32_t) ((uint32_t) (v12 / v6) + (uint32_t) v7) : v12 / v6));
    int32_t v41 = (int32_t) ((uint32_t) ((i36 % v34) / v38) * (uint32_t) v5) / v6;
    int32_t v42 = (int32_t) ((uint32_t) v41 + (uint32_t) v7);
    for (int32_t j43 = v4; j43 < ((int32_t) ((uint32_t) v19 + (uint32_t) v3) / v5); j43 += v7) {
      int32_t v44 = (int32_t) ((uint32_t) j43 * (uint32_t) v5) / v6;
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v45 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_365 = v31.get_noc_addr((int32_t) ((uint32_t) v40 + (uint32_t) v44), v4);
      noc_async_read(temp_365, v45, v29);
      int32_t v46 = (int32_t) ((uint32_t) v44 + (uint32_t) v7);
      int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v29);
      uint64_t temp_382 = v31.get_noc_addr((int32_t) ((uint32_t) v40 + (uint32_t) v46), v4);
      noc_async_read(temp_382, v47, v29);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
      cb_reserve_back(get_compile_time_arg_val(1), v10);
      int32_t v48 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v49 = (int32_t) ((uint32_t) v44 * (uint32_t) v35);
      uint64_t temp_396 = v27.get_noc_addr((int32_t) ((uint32_t) v49 + (uint32_t) v41), v4);
      noc_async_read(temp_396, v48, v25);
      int32_t v50 = (int32_t) ((uint32_t) v46 * (uint32_t) v35);
      int32_t v51 = (int32_t) ((uint32_t) v48 + (uint32_t) ((int32_t) ((uint32_t) v25 * (uint32_t) v9)));
      uint64_t temp_413 = v27.get_noc_addr((int32_t) ((uint32_t) v50 + (uint32_t) v41), v4);
      noc_async_read(temp_413, v51, v25);
      int32_t v52 = (int32_t) ((uint32_t) v48 + (uint32_t) v25);
      uint64_t temp_425 = v27.get_noc_addr((int32_t) ((uint32_t) v49 + (uint32_t) v42), v4);
      noc_async_read(temp_425, v52, v25);
      int32_t v53 = (int32_t) ((uint32_t) v48 + (uint32_t) ((int32_t) ((uint32_t) v25 * (uint32_t) 3)));
      uint64_t temp_437 = v27.get_noc_addr((int32_t) ((uint32_t) v50 + (uint32_t) v42), v4);
      noc_async_read(temp_437, v53, v25);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v10);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v9);
    int32_t v54 = get_write_ptr(get_compile_time_arg_val(2));
    int32_t v55 = (int32_t) ((uint32_t) v39 * (uint32_t) (v16 != (int32_t) ((uint32_t) (v16 / v6) * (uint32_t) v6) & v16 < v4 == v1 ? (int32_t) ((uint32_t) (v16 / v6) + (uint32_t) v7) : v16 / v6));
    uint64_t temp_356 = v23.get_noc_addr((int32_t) ((uint32_t) v55 + (uint32_t) v41), v4);
    noc_async_read(temp_356, v54, v21);
    int32_t v56 = (int32_t) ((uint32_t) v54 + (uint32_t) v21);
    uint64_t temp_368 = v23.get_noc_addr((int32_t) ((uint32_t) v55 + (uint32_t) v42), v4);
    noc_async_read(temp_368, v56, v21);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v9);
  }
  return;
}
