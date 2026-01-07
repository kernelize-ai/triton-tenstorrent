// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  int32_t v2 = 31;
  int32_t v3 = 0;
  int32_t v4 = 32;
  int32_t v5 = 1;
  int32_t v6 = 1024;
  bool v7 = true;
  int32_t v8 = 2;
  int32_t v9 = get_arg_val<uint32_t>(0);
  int32_t v10 = get_arg_val<uint32_t>(1);
  int32_t v11 = get_arg_val<uint32_t>(2);
  int32_t v12 = get_arg_val<uint32_t>(3);
  int32_t v13 = get_arg_val<uint32_t>(4);
  int32_t v14 = get_arg_val<uint32_t>(5);
  int32_t v15 = get_arg_val<uint32_t>(6);
  int32_t v16 = get_arg_val<uint32_t>(7);
  int32_t v17 = get_arg_val<uint32_t>(8);
  int32_t v18 = get_arg_val<uint32_t>(9);
  int32_t v19 = get_arg_val<uint32_t>(10);
  int32_t v20 = get_arg_val<uint32_t>(11);
  int32_t v21 = get_arg_val<uint32_t>(12);
  int32_t v22 = get_arg_val<uint32_t>(13);
  int32_t v23 = get_arg_val<uint32_t>(14);
  int32_t v24 = get_arg_val<uint32_t>(15);
  int32_t v25 = get_arg_val<uint32_t>(16);
  int32_t v26 = get_arg_val<uint32_t>(17);
  int32_t v27 = get_arg_val<uint32_t>(18);
  int32_t v28 = get_arg_val<uint32_t>(19);
  int32_t v29 = get_arg_val<uint32_t>(20);
  int32_t v30 = get_arg_val<uint32_t>(21);
  int32_t v31 = get_arg_val<uint32_t>(22);
  int32_t v32 = get_arg_val<uint32_t>(23);
  int32_t v33 = get_arg_val<uint32_t>(24);
  int32_t v34 = get_arg_val<uint32_t>(25);
  int32_t v35 = get_arg_val<uint32_t>(26);
  int32_t v36 = get_arg_val<uint32_t>(27);
  int32_t v37 = get_arg_val<uint32_t>(28);
  int32_t v38 = get_arg_val<uint32_t>(29);
  int32_t v39 = get_arg_val<uint32_t>(30);
  int32_t v40 = get_arg_val<uint32_t>(31);
  int32_t v41 = get_arg_val<uint32_t>(32);
  DataFormat v42 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v43 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v44;
  v44.bank_base_address = v19;
  v44.page_size = v43;
  v44.data_format = v42;
  InterleavedAddrGenFast<true> v45 = v44;
  DataFormat v46 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v47 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v48;
  v48.bank_base_address = v9;
  v48.page_size = v47;
  v48.data_format = v46;
  InterleavedAddrGenFast<true> v49 = v48;
  int32_t v50 = get_arg_val<uint32_t>(34);
  int32_t v51 = get_arg_val<uint32_t>(33);
  int32_t v52 = (int32_t) ((uint32_t) v40 + (uint32_t) v2) / v4;
  for (int32_t i53 = v51; i53 < v50; i53 += v5) {
    int32_t v54 = i53 / v52;
    int32_t v55 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) v2) / v4) - (uint32_t) v54) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) v2) / v4) - (uint32_t) v54) : v5;
    for (int32_t j56 = v3; j56 < ((int32_t) ((uint32_t) v41 + (uint32_t) v2) / v4); j56 += v5) {
      int32_t v57 = (int32_t) ((uint32_t) j56 * (uint32_t) v4);
      int32_t v58 = v57 / v4;
      int32_t v59 = (int32_t) ((uint32_t) v57 % (uint32_t) v4);
      cb_reserve_back(get_compile_time_arg_val(0), v5);
      int32_t v60 = get_write_ptr(get_compile_time_arg_val(0));
      uint64_t temp_556 = v49.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v54 + (uint32_t) (i53 % v55))) * (uint32_t) v4) / v4) * (uint32_t) (v11 != (int32_t) ((uint32_t) (v11 / v4) * (uint32_t) v4) & v11 < v3 == v1 ? (int32_t) ((uint32_t) (v11 / v4) + (uint32_t) v5) : v11 / v4))) + (uint32_t) v58)) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v54 + (uint32_t) (i53 % v55))) * (uint32_t) v4)) % (uint32_t) v4)) * (uint32_t) v4)) + (uint32_t) v59)))) * (uint32_t) v8)) / (uint32_t) v47), v3);
      noc_async_read(temp_556, v60, v47);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v5);
      cb_reserve_back(get_compile_time_arg_val(1), v5);
      int32_t v61 = get_write_ptr(get_compile_time_arg_val(1));
      uint64_t temp_600 = v45.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v58 * (uint32_t) (v21 != (int32_t) ((uint32_t) (v21 / v4) * (uint32_t) v4) & v21 < v3 == v1 ? (int32_t) ((uint32_t) (v21 / v4) + (uint32_t) v5) : v21 / v4))) + (uint32_t) ((int32_t) ((uint32_t) ((i53 % v52) / v55) * (uint32_t) v4) / v4))) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v59 * (uint32_t) v4)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i53 % v52) / v55) * (uint32_t) v4)) % (uint32_t) v4)))))) * (uint32_t) v8)) / (uint32_t) v43), v3);
      noc_async_read(temp_600, v61, v43);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v5);
    }
  }
  return;
}
