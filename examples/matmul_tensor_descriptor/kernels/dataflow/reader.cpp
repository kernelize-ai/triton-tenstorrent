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
  int32_t v16 = get_common_arg_val<uint32_t>(31);
  int32_t v17 = get_common_arg_val<uint32_t>(32);
  DataFormat v18 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v19 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v20;
  v20.bank_base_address = v13;
  v20.page_size = v19;
  v20.data_format = v18;
  InterleavedAddrGenFast<true> v21 = v20;
  DataFormat v22 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v23 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v24;
  v24.bank_base_address = v11;
  v24.page_size = v23;
  v24.data_format = v22;
  InterleavedAddrGenFast<true> v25 = v24;
  int32_t v26 = get_arg_val<uint32_t>(1);
  int32_t v27 = get_arg_val<uint32_t>(v2);
  int32_t v28 = (int32_t) ((uint32_t) v16 + (uint32_t) v3) / v5;
  int32_t v29 = v14 != (int32_t) ((uint32_t) (v14 / v6) * (uint32_t) v6) & v14 < v4 == v1 ? (int32_t) ((uint32_t) (v14 / v6) + (uint32_t) v7) : v14 / v6;
  for (int32_t i30 = v27; i30 < v26; i30 += v7) {
    int32_t v31 = i30 / v28;
    int32_t v32 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v6) - (uint32_t) v31) < v7 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v15 + (uint32_t) 31) / v6) - (uint32_t) v31) : v7;
    int32_t v33 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v31 + (uint32_t) (i30 % v32))) * (uint32_t) v6) / v6) * (uint32_t) (v12 != (int32_t) ((uint32_t) (v12 / v6) * (uint32_t) v6) & v12 < v4 == v1 ? (int32_t) ((uint32_t) (v12 / v6) + (uint32_t) v7) : v12 / v6));
    int32_t v34 = (int32_t) ((uint32_t) ((i30 % v28) / v32) * (uint32_t) v5) / v6;
    int32_t v35 = (int32_t) ((uint32_t) v34 + (uint32_t) v7);
    for (int32_t j36 = v4; j36 < ((int32_t) ((uint32_t) v17 + (uint32_t) v3) / v5); j36 += v7) {
      int32_t v37 = (int32_t) ((uint32_t) j36 * (uint32_t) v5) / v6;
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v38 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_307 = v25.get_noc_addr((int32_t) ((uint32_t) v33 + (uint32_t) v37), v4);
      noc_async_read(temp_307, v38, v23);
      int32_t v39 = (int32_t) ((uint32_t) v37 + (uint32_t) v7);
      int32_t v40 = (int32_t) ((uint32_t) v38 + (uint32_t) v23);
      uint64_t temp_324 = v25.get_noc_addr((int32_t) ((uint32_t) v33 + (uint32_t) v39), v4);
      noc_async_read(temp_324, v40, v23);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
      cb_reserve_back(get_compile_time_arg_val(1), v10);
      int32_t v41 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v42 = (int32_t) ((uint32_t) v37 * (uint32_t) v29);
      uint64_t temp_338 = v21.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v34), v4);
      noc_async_read(temp_338, v41, v19);
      int32_t v43 = (int32_t) ((uint32_t) v39 * (uint32_t) v29);
      int32_t v44 = (int32_t) ((uint32_t) v41 + (uint32_t) ((int32_t) ((uint32_t) v19 * (uint32_t) v9)));
      uint64_t temp_355 = v21.get_noc_addr((int32_t) ((uint32_t) v43 + (uint32_t) v34), v4);
      noc_async_read(temp_355, v44, v19);
      int32_t v45 = (int32_t) ((uint32_t) v41 + (uint32_t) v19);
      uint64_t temp_367 = v21.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v35), v4);
      noc_async_read(temp_367, v45, v19);
      int32_t v46 = (int32_t) ((uint32_t) v41 + (uint32_t) ((int32_t) ((uint32_t) v19 * (uint32_t) 3)));
      uint64_t temp_379 = v21.get_noc_addr((int32_t) ((uint32_t) v43 + (uint32_t) v35), v4);
      noc_async_read(temp_379, v46, v19);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v10);
    }
  }
  return;
}
