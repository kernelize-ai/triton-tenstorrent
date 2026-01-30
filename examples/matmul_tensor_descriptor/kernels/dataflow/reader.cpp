// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  uint32_t v1 = 0;
  uint32_t v2 = 32;
  bool v3 = false;
  size_t v4 = 0;
  size_t v5 = 1;
  int32_t v6 = 0;
  int32_t v7 = 512;
  int32_t v8 = 256;
  int32_t v9 = 32;
  int32_t v10 = 1;
  bool v11 = true;
  int32_t v12 = 16;
  int32_t v13 = 64;
  int32_t v14 = 2;
  int32_t v15 = 128;
  int32_t v16 = 3;
  int32_t v17 = get_common_arg_val<uint32_t>(v4);
  int32_t v18 = get_common_arg_val<uint32_t>(2);
  int32_t v19 = get_common_arg_val<uint32_t>(10);
  int32_t v20 = get_common_arg_val<uint32_t>(12);
  int32_t v21 = get_common_arg_val<uint32_t>(30);
  int32_t v22 = get_common_arg_val<uint32_t>(31);
  int32_t v23 = get_common_arg_val<uint32_t>(32);
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v26;
  v26.bank_base_address = v19;
  v26.page_size = v25;
  v26.data_format = v24;
  InterleavedAddrGenFast<true> v27 = v26;
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v30;
  v30.bank_base_address = v17;
  v30.page_size = v29;
  v30.data_format = v28;
  InterleavedAddrGenFast<true> v31 = v30;
  int32_t v32 = get_arg_val<uint32_t>(v5);
  int32_t v33 = get_arg_val<uint32_t>(v4);
  int32_t v34 = (int32_t) ((uint32_t) v22 + (uint32_t) 255) / v8;
  for (int32_t i35 = v33; i35 < v32; i35 += v10) {
    int32_t v36 = i35 / v34;
    int32_t v37 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) 31) / v9) - (uint32_t) v36) < v10 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) 31) / v9) - (uint32_t) v36) : v10;
    for (int32_t j38 = v6; j38 < ((int32_t) ((uint32_t) v23 + (uint32_t) 511) / v7); j38 += v10) {
      int32_t v39 = (int32_t) ((uint32_t) j38 * (uint32_t) v7) / v9;
      cb_reserve_back(get_compile_time_arg_val(0), v12);
      int32_t v40 = get_write_ptr(get_compile_time_arg_val(0));
      for (size_t k41 = v4; k41 < (16); k41 += v5) {
        int32_t v42 = (int32_t) ((ptrdiff_t) k41);
        int32_t v43 = (int32_t) ((uint32_t) v40 + (uint32_t) ((int32_t) ((uint32_t) v42 * (uint32_t) v29)));
        uint64_t temp_488 = v31.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v36 + (uint32_t) (i35 % v37))) * (uint32_t) v9) / v9) * (uint32_t) (v18 != (int32_t) ((uint32_t) (v18 / v9) * (uint32_t) v9) & v18 < v6 == v3 ? (int32_t) ((uint32_t) (v18 / v9) + (uint32_t) v10) : v18 / v9))) + (uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v42 & (uint32_t) v10)) * (uint32_t) v9)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 < v2 ? (uint32_t) v42 >> (uint32_t) v10 : v1)) & (uint32_t) v10)) * (uint32_t) v13)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 < v2 ? (uint32_t) v42 >> (uint32_t) v14 : v1)) & (uint32_t) v10)) * (uint32_t) v15)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v16 < v2 ? (uint32_t) v42 >> (uint32_t) v16 : v1)) & (uint32_t) v10)) * (uint32_t) v8))) / v9)))), v6);
        noc_async_read(temp_488, v43, v29);
      }
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v12);
      cb_reserve_back(get_compile_time_arg_val(1), v15);
      int32_t v44 = get_write_ptr(get_compile_time_arg_val(1));
      for (size_t k45 = v4; k45 < (128); k45 += v5) {
        int32_t v46 = (int32_t) ((ptrdiff_t) k45);
        int32_t v47 = (int32_t) ((uint32_t) v44 + (uint32_t) ((int32_t) ((uint32_t) v46 * (uint32_t) v25)));
        uint64_t temp_530 = v27.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v46 & (uint32_t) v10)) * (uint32_t) v9)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 < v2 ? (uint32_t) v46 >> (uint32_t) v10 : v1)) & (uint32_t) v10)) * (uint32_t) v13)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 < v2 ? (uint32_t) v46 >> (uint32_t) v14 : v1)) & (uint32_t) v10)) * (uint32_t) v15)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v16 < v2 ? (uint32_t) v46 >> (uint32_t) v16 : v1)) & (uint32_t) v10)) * (uint32_t) v8))) / v9))) * (uint32_t) (v20 != (int32_t) ((uint32_t) (v20 / v9) * (uint32_t) v9) & v20 < v6 == v3 ? (int32_t) ((uint32_t) (v20 / v9) + (uint32_t) v10) : v20 / v9))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i35 % v34) / v37) * (uint32_t) v8) / v9) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) 4 < v2 ? (uint32_t) v46 >> (uint32_t) 4 : v1)) & (uint32_t) v10)) * (uint32_t) v9)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) 5 < v2 ? (uint32_t) v46 >> (uint32_t) 5 : v1)) & (uint32_t) v10)) * (uint32_t) v13)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) 6 < v2 ? (uint32_t) v46 >> (uint32_t) 6 : v1)) & (uint32_t) v10)) * (uint32_t) v15))) / v9)))), v6);
        noc_async_read(temp_530, v47, v25);
      }
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v15);
    }
  }
  return;
}
