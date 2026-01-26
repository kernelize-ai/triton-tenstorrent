// matmul_kernel_fused__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  size_t v2 = 0;
  int32_t v3 = 31;
  int32_t v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  bool v7 = true;
  int32_t v8 = get_common_arg_val<uint32_t>(v2);
  int32_t v9 = get_common_arg_val<uint32_t>(2);
  int32_t v10 = get_common_arg_val<uint32_t>(10);
  int32_t v11 = get_common_arg_val<uint32_t>(12);
  int32_t v12 = get_common_arg_val<uint32_t>(30);
  int32_t v13 = get_common_arg_val<uint32_t>(32);
  int32_t v14 = get_common_arg_val<uint32_t>(40);
  int32_t v15 = get_common_arg_val<uint32_t>(41);
  int32_t v16 = get_common_arg_val<uint32_t>(42);
  DataFormat v17 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v18 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v19;
  v19.bank_base_address = v12;
  v19.page_size = v18;
  v19.data_format = v17;
  InterleavedAddrGenFast<true> v20 = v19;
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v10;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  DataFormat v25 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v26 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v27;
  v27.bank_base_address = v8;
  v27.page_size = v26;
  v27.data_format = v25;
  InterleavedAddrGenFast<true> v28 = v27;
  int32_t v29 = get_arg_val<uint32_t>(1);
  int32_t v30 = get_arg_val<uint32_t>(v2);
  int32_t v31 = (int32_t) ((uint32_t) v15 + (uint32_t) v3) / v5;
  for (int32_t i32 = v30; i32 < v29; i32 += v6) {
    int32_t v33 = i32 / v31;
    int32_t v34 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) v3) / v5) - (uint32_t) v33) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) v3) / v5) - (uint32_t) v33) : v6;
    int32_t v35 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v33 + (uint32_t) (i32 % v34))) * (uint32_t) v5) / v5;
    int32_t v36 = (int32_t) ((uint32_t) ((i32 % v31) / v34) * (uint32_t) v5) / v5;
    for (int32_t j37 = v4; j37 < ((int32_t) ((uint32_t) v16 + (uint32_t) v3) / v5); j37 += v6) {
      int32_t v38 = (int32_t) ((uint32_t) j37 * (uint32_t) v5) / v5;
      cb_reserve_back(get_compile_time_arg_val(0), v6);
      int32_t v39 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_323 = v28.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 * (uint32_t) (v9 != (int32_t) ((uint32_t) (v9 / v5) * (uint32_t) v5) & v9 < v4 == v1 ? (int32_t) ((uint32_t) (v9 / v5) + (uint32_t) v6) : v9 / v5))) + (uint32_t) v38), v4);
      noc_async_read(temp_323, v39, v26);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v6);
      cb_reserve_back(get_compile_time_arg_val(1), v6);
      int32_t v40 = get_write_ptr(get_compile_time_arg_val(1));
      uint64_t temp_337 = v24.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v38 * (uint32_t) (v11 != (int32_t) ((uint32_t) (v11 / v5) * (uint32_t) v5) & v11 < v4 == v1 ? (int32_t) ((uint32_t) (v11 / v5) + (uint32_t) v6) : v11 / v5))) + (uint32_t) v36), v4);
      noc_async_read(temp_337, v40, v22);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v6);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v6);
    int32_t v41 = get_write_ptr(get_compile_time_arg_val(2));
    uint64_t temp_317 = v20.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 * (uint32_t) (v13 != (int32_t) ((uint32_t) (v13 / v5) * (uint32_t) v5) & v13 < v4 == v1 ? (int32_t) ((uint32_t) (v13 / v5) + (uint32_t) v6) : v13 / v5))) + (uint32_t) v36), v4);
    noc_async_read(temp_317, v41, v18);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v6);
  }
  return;
}
