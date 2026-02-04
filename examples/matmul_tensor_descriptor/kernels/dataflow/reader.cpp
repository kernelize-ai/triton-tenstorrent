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
  int32_t v6 = 1;
  bool v7 = true;
  int32_t v8 = 4;
  int32_t v9 = 32;
  int32_t v10 = 2048;
  int32_t v11 = 4096;
  int32_t v12 = 6144;
  int32_t v13 = get_common_arg_val<uint32_t>(v2);
  int32_t v14 = get_common_arg_val<uint32_t>(2);
  int32_t v15 = get_common_arg_val<uint32_t>(10);
  int32_t v16 = get_common_arg_val<uint32_t>(12);
  int32_t v17 = get_common_arg_val<uint32_t>(30);
  int32_t v18 = get_common_arg_val<uint32_t>(31);
  int32_t v19 = get_common_arg_val<uint32_t>(32);
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v22;
  v22.bank_base_address = v15;
  v22.page_size = v21;
  v22.data_format = v20;
  InterleavedAddrGenFast<true> v23 = v22;
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v26;
  v26.bank_base_address = v13;
  v26.page_size = v25;
  v26.data_format = v24;
  InterleavedAddrGenFast<true> v27 = v26;
  int32_t v28 = get_arg_val<uint32_t>(1);
  int32_t v29 = get_arg_val<uint32_t>(v2);
  int32_t v30 = (int32_t) ((uint32_t) v18 + (uint32_t) v3) / v5;
  int32_t v31 = v14 != (int32_t) ((uint32_t) (v14 / v9) * (uint32_t) v9) & v14 < v4 == v1 ? (int32_t) ((uint32_t) (v14 / v9) + (uint32_t) v6) : v14 / v9;
  int32_t v32 = v16 != (int32_t) ((uint32_t) (v16 / v9) * (uint32_t) v9) & v16 < v4 == v1 ? (int32_t) ((uint32_t) (v16 / v9) + (uint32_t) v6) : v16 / v9;
  for (int32_t i33 = v29; i33 < v28; i33 += v6) {
    int32_t v34 = i33 / v30;
    int32_t v35 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v3) / v5) - (uint32_t) v34) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v3) / v5) - (uint32_t) v34) : v6;
    for (int32_t j36 = v4; j36 < ((int32_t) ((uint32_t) v19 + (uint32_t) v3) / v5); j36 += v6) {
      cb_reserve_back(get_compile_time_arg_val(0), v8);
      int32_t v37 = get_write_ptr(get_compile_time_arg_val(0));
      int32_t v38 = (int32_t) ((uint32_t) j36 * (uint32_t) v5) / v9;
      int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) (i33 % v35))) * (uint32_t) v5) / v9) * (uint32_t) v31)) + (uint32_t) v38);
      uint64_t temp_298 = v27.get_noc_addr(v39, v4);
      noc_async_read(temp_298, v37, v25);
      int32_t v40 = (int32_t) ((uint32_t) v37 + (uint32_t) v10);
      int32_t v41 = (int32_t) ((uint32_t) v39 + (uint32_t) v6);
      uint64_t temp_310 = v27.get_noc_addr(v41, v4);
      noc_async_read(temp_310, v40, v25);
      int32_t v42 = (int32_t) ((uint32_t) v37 + (uint32_t) v11);
      uint64_t temp_322 = v27.get_noc_addr((int32_t) ((uint32_t) v39 + (uint32_t) v31), v4);
      noc_async_read(temp_322, v42, v25);
      int32_t v43 = (int32_t) ((uint32_t) v37 + (uint32_t) v12);
      uint64_t temp_334 = v27.get_noc_addr((int32_t) ((uint32_t) v41 + (uint32_t) v31), v4);
      noc_async_read(temp_334, v43, v25);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v8);
      cb_reserve_back(get_compile_time_arg_val(1), v8);
      int32_t v44 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v38 * (uint32_t) v32)) + (uint32_t) ((int32_t) ((uint32_t) ((i33 % v30) / v35) * (uint32_t) v5) / v9));
      uint64_t temp_348 = v23.get_noc_addr(v45, v4);
      noc_async_read(temp_348, v44, v21);
      int32_t v46 = (int32_t) ((uint32_t) v44 + (uint32_t) v11);
      int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v6);
      uint64_t temp_360 = v23.get_noc_addr(v47, v4);
      noc_async_read(temp_360, v46, v21);
      int32_t v48 = (int32_t) ((uint32_t) v44 + (uint32_t) v10);
      uint64_t temp_372 = v23.get_noc_addr((int32_t) ((uint32_t) v45 + (uint32_t) v32), v4);
      noc_async_read(temp_372, v48, v21);
      int32_t v49 = (int32_t) ((uint32_t) v44 + (uint32_t) v12);
      uint64_t temp_384 = v23.get_noc_addr((int32_t) ((uint32_t) v47 + (uint32_t) v32), v4);
      noc_async_read(temp_384, v49, v21);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v8);
    }
  }
  return;
}
