// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 256;
  int32_t v2 = 32;
  int32_t v3 = 1;
  bool v4 = true;
  int32_t v5 = 8;
  int32_t v6 = 0;
  int32_t v7 = 2;
  int32_t v8 = 3;
  int32_t v9 = 4;
  int32_t v10 = 5;
  int32_t v11 = 6;
  int32_t v12 = 7;
  int32_t v13 = get_common_arg_val<uint32_t>(20);
  int32_t v14 = get_common_arg_val<uint32_t>(22);
  int32_t v15 = get_common_arg_val<uint32_t>(30);
  int32_t v16 = get_common_arg_val<uint32_t>(31);
  DataFormat v17 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v18 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v19;
  v19.bank_base_address = v13;
  v19.page_size = v18;
  v19.data_format = v17;
  InterleavedAddrGenFast<true> v20 = v19;
  int32_t v21 = get_arg_val<uint32_t>(1);
  int32_t v22 = get_arg_val<uint32_t>(0);
  int32_t v23 = (int32_t) ((uint32_t) v16 + (uint32_t) 255) / v1;
  for (int32_t i24 = v22; i24 < v21; i24 += v3) {
    int32_t v25 = i24 / v23;
    int32_t v26 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v2) - (uint32_t) v25) < v3 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v2) - (uint32_t) v25) : v3;
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v5);
    }
    int32_t v27 = (int32_t) ((uint32_t) ((i24 % v23) / v26) * (uint32_t) v1) / v2;
    int32_t v28 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v29 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v25 + (uint32_t) (i24 % v26))) * (uint32_t) v2) / v2) * (uint32_t) (v14 != (int32_t) ((uint32_t) (v14 / v2) * (uint32_t) v2) & v14 < v6 == false ? (int32_t) ((uint32_t) (v14 / v2) + (uint32_t) v3) : v14 / v2));
    uint64_t temp_185 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) v27), v6);
    noc_async_write(v28, temp_185, v18);
    int32_t v30 = (int32_t) ((uint32_t) v28 + (uint32_t) v18);
    uint64_t temp_202 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v3))), v6);
    noc_async_write(v30, temp_202, v18);
    int32_t v31 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v7)));
    uint64_t temp_219 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v7))), v6);
    noc_async_write(v31, temp_219, v18);
    int32_t v32 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v8)));
    uint64_t temp_236 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v8))), v6);
    noc_async_write(v32, temp_236, v18);
    int32_t v33 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v9)));
    uint64_t temp_253 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v9))), v6);
    noc_async_write(v33, temp_253, v18);
    int32_t v34 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v10)));
    uint64_t temp_270 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v10))), v6);
    noc_async_write(v34, temp_270, v18);
    int32_t v35 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v11)));
    uint64_t temp_287 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v11))), v6);
    noc_async_write(v35, temp_287, v18);
    int32_t v36 = (int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) v18 * (uint32_t) v12)));
    uint64_t temp_304 = v20.get_noc_addr((int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v12))), v6);
    noc_async_write(v36, temp_304, v18);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v5);
  }
  return;
}

