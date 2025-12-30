// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  int32_t v2 = 31;
  int32_t v3 = 0;
  int32_t v4 = 32;
  int32_t v5 = 8;
  int32_t v6 = 1;
  int32_t v7 = 1024;
  bool v8 = true;
  int32_t v9 = 2;
  int32_t v10 = get_arg_val<uint32_t>(0);
  int32_t v11 = get_arg_val<uint32_t>(1);
  int32_t v12 = get_arg_val<uint32_t>(2);
  int32_t v13 = get_arg_val<uint32_t>(3);
  int32_t v14 = get_arg_val<uint32_t>(4);
  int32_t v15 = get_arg_val<uint32_t>(5);
  int32_t v16 = get_arg_val<uint32_t>(6);
  int32_t v17 = get_arg_val<uint32_t>(7);
  int32_t v18 = get_arg_val<uint32_t>(8);
  int32_t v19 = get_arg_val<uint32_t>(9);
  int32_t v20 = get_arg_val<uint32_t>(10);
  int32_t v21 = get_arg_val<uint32_t>(11);
  int32_t v22 = get_arg_val<uint32_t>(12);
  int32_t v23 = get_arg_val<uint32_t>(13);
  int32_t v24 = get_arg_val<uint32_t>(14);
  int32_t v25 = get_arg_val<uint32_t>(15);
  int32_t v26 = get_arg_val<uint32_t>(16);
  int32_t v27 = get_arg_val<uint32_t>(17);
  int32_t v28 = get_arg_val<uint32_t>(18);
  int32_t v29 = get_arg_val<uint32_t>(19);
  int32_t v30 = get_arg_val<uint32_t>(20);
  int32_t v31 = get_arg_val<uint32_t>(21);
  int32_t v32 = get_arg_val<uint32_t>(22);
  int32_t v33 = get_arg_val<uint32_t>(23);
  int32_t v34 = get_arg_val<uint32_t>(24);
  int32_t v35 = get_arg_val<uint32_t>(25);
  int32_t v36 = get_arg_val<uint32_t>(26);
  int32_t v37 = get_arg_val<uint32_t>(27);
  int32_t v38 = get_arg_val<uint32_t>(28);
  int32_t v39 = get_arg_val<uint32_t>(29);
  int32_t v40 = get_arg_val<uint32_t>(30);
  int32_t v41 = get_arg_val<uint32_t>(31);
  int32_t v42 = get_arg_val<uint32_t>(32);
  DataFormat v43 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v44 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v45;
  v45.bank_base_address = v20;
  v45.page_size = v44;
  v45.data_format = v43;
  InterleavedAddrGenFast<true> v46 = v45;
  DataFormat v47 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v48 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v49;
  v49.bank_base_address = v10;
  v49.page_size = v48;
  v49.data_format = v47;
  InterleavedAddrGenFast<true> v50 = v49;
  int32_t v51 = get_arg_val<uint32_t>(34);
  int32_t v52 = get_arg_val<uint32_t>(33);
  int32_t v53 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v41 + (uint32_t) v2) / v4) * (uint32_t) v5);
  for (int32_t i54 = v52; i54 < v51; i54 += v6) {
    int32_t v55 = (int32_t) ((uint32_t) (i54 / v53) * (uint32_t) v5);
    int32_t v56 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v40 + (uint32_t) v2) / v4) - (uint32_t) v55) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v40 + (uint32_t) v2) / v4) - (uint32_t) v55) : v5;
    for (int32_t j57 = v3; j57 < ((int32_t) ((uint32_t) v42 + (uint32_t) v2) / v4); j57 += v6) {
      int32_t v58 = (int32_t) ((uint32_t) j57 * (uint32_t) v4);
      int32_t v59 = v58 / v4;
      int32_t v60 = (int32_t) ((uint32_t) v58 % (uint32_t) v4);
      uint64_t temp_569 = v50.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v55 + (uint32_t) (i54 % v56))) * (uint32_t) v4) / v4) * (uint32_t) (v12 != (int32_t) ((uint32_t) (v12 / v4) * (uint32_t) v4) & v12 < v3 == v1 ? (int32_t) ((uint32_t) (v12 / v4) + (uint32_t) v6) : v12 / v4))) + (uint32_t) v59)) * (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v55 + (uint32_t) (i54 % v56))) * (uint32_t) v4)) % (uint32_t) v4)) * (uint32_t) v4)) + (uint32_t) v60)))) * (uint32_t) v9)) / (uint32_t) v48), v3);
      cb_reserve_back(get_compile_time_arg_val(0), v6);
      int32_t v61 = get_write_ptr(get_compile_time_arg_val(0));
      noc_async_read(temp_569, v61, v48);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v6);
      uint64_t temp_613 = v46.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v59 * (uint32_t) (v22 != (int32_t) ((uint32_t) (v22 / v4) * (uint32_t) v4) & v22 < v3 == v1 ? (int32_t) ((uint32_t) (v22 / v4) + (uint32_t) v6) : v22 / v4))) + (uint32_t) ((int32_t) ((uint32_t) ((i54 % v53) / v56) * (uint32_t) v4) / v4))) * (uint32_t) v7)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v60 * (uint32_t) v4)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i54 % v53) / v56) * (uint32_t) v4)) % (uint32_t) v4)))))) * (uint32_t) v9)) / (uint32_t) v44), v3);
      cb_reserve_back(get_compile_time_arg_val(1), v6);
      int32_t v62 = get_write_ptr(get_compile_time_arg_val(1));
      noc_async_read(temp_613, v62, v44);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v6);
    }
  }
  return;
}
