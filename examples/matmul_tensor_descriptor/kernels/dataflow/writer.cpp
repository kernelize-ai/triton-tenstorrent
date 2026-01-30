// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 128;
  int32_t v2 = 64;
  int32_t v3 = 1;
  bool v4 = true;
  int32_t v5 = 8;
  int32_t v6 = 32;
  int32_t v7 = 0;
  int32_t v8 = 2;
  int32_t v9 = 3;
  int32_t v10 = get_common_arg_val<uint32_t>(20);
  int32_t v11 = get_common_arg_val<uint32_t>(22);
  int32_t v12 = get_common_arg_val<uint32_t>(30);
  int32_t v13 = get_common_arg_val<uint32_t>(31);
  DataFormat v14 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v15 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v16;
  v16.bank_base_address = v10;
  v16.page_size = v15;
  v16.data_format = v14;
  InterleavedAddrGenFast<true> v17 = v16;
  int32_t v18 = get_arg_val<uint32_t>(1);
  int32_t v19 = get_arg_val<uint32_t>(0);
  int32_t v20 = (int32_t) ((uint32_t) v13 + (uint32_t) 127) / v1;
  int32_t v21 = v11 != (int32_t) ((uint32_t) (v11 / v6) * (uint32_t) v6) & v11 < v7 == false ? (int32_t) ((uint32_t) (v11 / v6) + (uint32_t) v3) : v11 / v6;
  for (int32_t i22 = v19; i22 < v18; i22 += v3) {
    int32_t v23 = i22 / v20;
    int32_t v24 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) 63) / v2) - (uint32_t) v23) < v3 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) 63) / v2) - (uint32_t) v23) : v3;
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v5);
    }
    int32_t v25 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) (i22 % v24))) * (uint32_t) v2) / v6;
    int32_t v26 = (int32_t) ((uint32_t) ((i22 % v20) / v24) * (uint32_t) v1) / v6;
    int32_t v27 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v28 = (int32_t) ((uint32_t) v25 * (uint32_t) v21);
    uint64_t temp_187 = v17.get_noc_addr((int32_t) ((uint32_t) v28 + (uint32_t) v26), v7);
    noc_async_write(v27, temp_187, v15);
    int32_t v29 = (int32_t) ((uint32_t) v26 + (uint32_t) v3);
    int32_t v30 = (int32_t) ((uint32_t) v27 + (uint32_t) v15);
    uint64_t temp_204 = v17.get_noc_addr((int32_t) ((uint32_t) v28 + (uint32_t) v29), v7);
    noc_async_write(v30, temp_204, v15);
    int32_t v31 = (int32_t) ((uint32_t) v26 + (uint32_t) v8);
    int32_t v32 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) v8)));
    uint64_t temp_221 = v17.get_noc_addr((int32_t) ((uint32_t) v28 + (uint32_t) v31), v7);
    noc_async_write(v32, temp_221, v15);
    int32_t v33 = (int32_t) ((uint32_t) v26 + (uint32_t) v9);
    int32_t v34 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) v9)));
    uint64_t temp_238 = v17.get_noc_addr((int32_t) ((uint32_t) v28 + (uint32_t) v33), v7);
    noc_async_write(v34, temp_238, v15);
    int32_t v35 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v25 + (uint32_t) v3)) * (uint32_t) v21);
    int32_t v36 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) 4)));
    uint64_t temp_260 = v17.get_noc_addr((int32_t) ((uint32_t) v35 + (uint32_t) v26), v7);
    noc_async_write(v36, temp_260, v15);
    int32_t v37 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) 5)));
    uint64_t temp_272 = v17.get_noc_addr((int32_t) ((uint32_t) v35 + (uint32_t) v29), v7);
    noc_async_write(v37, temp_272, v15);
    int32_t v38 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) 6)));
    uint64_t temp_284 = v17.get_noc_addr((int32_t) ((uint32_t) v35 + (uint32_t) v31), v7);
    noc_async_write(v38, temp_284, v15);
    int32_t v39 = (int32_t) ((uint32_t) v27 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) 7)));
    uint64_t temp_296 = v17.get_noc_addr((int32_t) ((uint32_t) v35 + (uint32_t) v33), v7);
    noc_async_write(v39, temp_296, v15);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v5);
  }
  return;
}
