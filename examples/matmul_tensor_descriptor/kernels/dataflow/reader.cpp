// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  size_t v2 = 0;
  int32_t v3 = 0;
  int32_t v4 = 512;
  int32_t v5 = 128;
  int32_t v6 = 64;
  int32_t v7 = 1;
  bool v8 = true;
  int32_t v9 = 32;
  int32_t v10 = 2;
  int32_t v11 = 16;
  int32_t v12 = 4;
  int32_t v13 = 8;
  int32_t v14 = get_common_arg_val<uint32_t>(v2);
  int32_t v15 = get_common_arg_val<uint32_t>(2);
  int32_t v16 = get_common_arg_val<uint32_t>(10);
  int32_t v17 = get_common_arg_val<uint32_t>(12);
  int32_t v18 = get_common_arg_val<uint32_t>(30);
  int32_t v19 = get_common_arg_val<uint32_t>(31);
  int32_t v20 = get_common_arg_val<uint32_t>(32);
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v16;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  DataFormat v25 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v26 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v27;
  v27.bank_base_address = v14;
  v27.page_size = v26;
  v27.data_format = v25;
  InterleavedAddrGenFast<true> v28 = v27;
  int32_t v29 = get_arg_val<uint32_t>(1);
  int32_t v30 = get_arg_val<uint32_t>(v2);
  int32_t v31 = (int32_t) ((uint32_t) v19 + (uint32_t) 127) / v5;
  for (int32_t i32 = v30; i32 < v29; i32 += v7) {
    int32_t v33 = i32 / v31;
    int32_t v34 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) 63) / v6) - (uint32_t) v33) < v7 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) 63) / v6) - (uint32_t) v33) : v7;
    for (int32_t j35 = v3; j35 < ((int32_t) ((uint32_t) v20 + (uint32_t) 511) / v4); j35 += v7) {
      int32_t v36 = (int32_t) ((uint32_t) j35 * (uint32_t) v4) / v9;
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v37 = get_write_ptr(get_compile_time_arg_val(0));
      for (int32_t k38 = v3; k38 < v10; k38 += v7) {
        for (int32_t l39 = v3; l39 < v11; l39 += v7) {
          int32_t v40 = (int32_t) ((uint32_t) v37 + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) k38 & (uint32_t) v7)) * (uint32_t) v11)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) l39 & (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) l39 & (uint32_t) v10)))) + (uint32_t) ((int32_t) ((uint32_t) l39 & (uint32_t) v12)))) + (uint32_t) ((int32_t) ((uint32_t) l39 & (uint32_t) v13)))))) * (uint32_t) v26)));
          uint64_t temp_412 = v28.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v33 + (uint32_t) (i32 % v34))) * (uint32_t) v6) / v9) + (uint32_t) k38)) * (uint32_t) (v15 != (int32_t) ((uint32_t) (v15 / v9) * (uint32_t) v9) & v15 < v3 == v1 ? (int32_t) ((uint32_t) (v15 / v9) + (uint32_t) v7) : v15 / v9))) + (uint32_t) ((int32_t) ((uint32_t) v36 + (uint32_t) l39))), v3);
          noc_async_read(temp_412, v40, v26);
        }
      }
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
      cb_reserve_back(get_compile_time_arg_val(1), v6);
      int32_t v41 = get_write_ptr(get_compile_time_arg_val(1));
      for (int32_t k42 = v3; k42 < v11; k42 += v7) {
        for (int32_t l43 = v3; l43 < v12; l43 += v7) {
          int32_t v44 = (int32_t) ((uint32_t) v41 + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) k42 & (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) k42 & (uint32_t) v10)))) + (uint32_t) ((int32_t) ((uint32_t) k42 & (uint32_t) v12)))) + (uint32_t) ((int32_t) ((uint32_t) k42 & (uint32_t) v13)))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) l43 & (uint32_t) v7)) * (uint32_t) v11)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v7 < 32 ? (uint32_t) l43 >> (uint32_t) v7 : 0)) & (uint32_t) v7)) * (uint32_t) v9)))))) * (uint32_t) v22)));
          uint64_t temp_415 = v24.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v36 + (uint32_t) k42)) * (uint32_t) (v17 != (int32_t) ((uint32_t) (v17 / v9) * (uint32_t) v9) & v17 < v3 == v1 ? (int32_t) ((uint32_t) (v17 / v9) + (uint32_t) v7) : v17 / v9))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i32 % v31) / v34) * (uint32_t) v5) / v9) + (uint32_t) l43))), v3);
          noc_async_read(temp_415, v44, v22);
        }
      }
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v6);
    }
  }
  return;
}
