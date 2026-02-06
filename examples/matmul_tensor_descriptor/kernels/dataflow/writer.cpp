// matmul_kernel_tma__writer
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
  bool v6 = true;
  int32_t v7 = 4;
  int32_t v8 = 32;
  int8_t v9 = 0;
  int32_t v10 = 2048;
  int32_t v11 = 4096;
  int32_t v12 = 6144;
  int32_t v13 = get_common_arg_val<uint32_t>(10);
  int32_t v14 = get_common_arg_val<uint32_t>(12);
  int32_t v15 = get_common_arg_val<uint32_t>(20);
  int32_t v16 = get_common_arg_val<uint32_t>(22);
  int32_t v17 = get_common_arg_val<uint32_t>(30);
  int32_t v18 = get_common_arg_val<uint32_t>(31);
  int32_t v19 = get_common_arg_val<uint32_t>(32);
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v22;
  v22.bank_base_address = v15;
  v22.page_size = v21;
  v22.data_format = v20;
  InterleavedAddrGenFast<true> v23 = v22;
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v26 = get_arg_val<uint32_t>(1);
  int32_t v27 = get_arg_val<uint32_t>(0);
  int32_t v28 = (int32_t) ((uint32_t) v18 + (uint32_t) v2) / v4;
  int32_t v29 = v14 != (int32_t) ((uint32_t) (v14 / v8) * (uint32_t) v8) & v14 < v3 == v1 ? (int32_t) ((uint32_t) (v14 / v8) + (uint32_t) v5) : v14 / v8;
  int32_t v30 = v16 != (int32_t) ((uint32_t) (v16 / v8) * (uint32_t) v8) & v16 < v3 == v1 ? (int32_t) ((uint32_t) (v16 / v8) + (uint32_t) v5) : v16 / v8;
  for (int32_t i31 = v27; i31 < v26; i31 += v5) {
    int32_t v32 = i31 / v28;
    int32_t v33 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v2) / v4) - (uint32_t) v32) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v2) / v4) - (uint32_t) v32) : v5;
    int32_t v34 = (int32_t) ((uint32_t) ((i31 % v28) / v33) * (uint32_t) v4) / v8;
    for (int32_t j35 = v3; j35 < ((int32_t) ((uint32_t) v19 + (uint32_t) v2) / v4); j35 += v5) {
      cb_reserve_back(get_compile_time_arg_val(1), v7);
      InterleavedAddrGenFast<true> v36;
      v36.bank_base_address = v13;
      v36.page_size = v25;
      v36.data_format = v24;
      InterleavedAddrGenFast<true> v37 = v36;
      int32_t v38 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j35 * (uint32_t) v4) / v8) * (uint32_t) v29)) + (uint32_t) v34);
      uint64_t temp_228 = v37.get_noc_addr(v39, v3, v9);
      noc_async_read(temp_228, v38, v25);
      int32_t v40 = (int32_t) ((uint32_t) v38 + (uint32_t) v10);
      int32_t v41 = (int32_t) ((uint32_t) v39 + (uint32_t) v5);
      uint64_t temp_240 = v37.get_noc_addr(v41, v3, v9);
      noc_async_read(temp_240, v40, v25);
      int32_t v42 = (int32_t) ((uint32_t) v38 + (uint32_t) v11);
      uint64_t temp_252 = v37.get_noc_addr((int32_t) ((uint32_t) v39 + (uint32_t) v29), v3, v9);
      noc_async_read(temp_252, v42, v25);
      int32_t v43 = (int32_t) ((uint32_t) v38 + (uint32_t) v12);
      uint64_t temp_264 = v37.get_noc_addr((int32_t) ((uint32_t) v41 + (uint32_t) v29), v3, v9);
      noc_async_read(temp_264, v43, v25);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v7);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v7);
    }
    int32_t v44 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v32 + (uint32_t) (i31 % v33))) * (uint32_t) v4) / v8) * (uint32_t) v30)) + (uint32_t) v34);
    uint64_t temp_203 = v23.get_noc_addr(v45, v3);
    noc_async_write(v44, temp_203, v21);
    int32_t v46 = (int32_t) ((uint32_t) v44 + (uint32_t) v10);
    int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v5);
    uint64_t temp_215 = v23.get_noc_addr(v47, v3);
    noc_async_write(v46, temp_215, v21);
    int32_t v48 = (int32_t) ((uint32_t) v44 + (uint32_t) v11);
    uint64_t temp_227 = v23.get_noc_addr((int32_t) ((uint32_t) v45 + (uint32_t) v30), v3);
    noc_async_write(v48, temp_227, v21);
    int32_t v49 = (int32_t) ((uint32_t) v44 + (uint32_t) v12);
    uint64_t temp_239 = v23.get_noc_addr((int32_t) ((uint32_t) v47 + (uint32_t) v30), v3);
    noc_async_write(v49, temp_239, v21);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v7);
  }
  return;
}
