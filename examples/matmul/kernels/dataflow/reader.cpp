// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
void kernel_main() {
  int32_t v1 = 31;
  int32_t v2 = 1;
  int32_t v3 = 0;
  bool v4 = true;
  int32_t v5 = 32;
  int32_t v6 = 64;
  int32_t v7 = get_arg_val<uint32_t>(0);
  int32_t v8 = get_arg_val<uint32_t>(1);
  int32_t v9 = get_arg_val<uint32_t>(2);
  int32_t v10 = get_arg_val<uint32_t>(3);
  int32_t v11 = get_arg_val<uint32_t>(4);
  int32_t v12 = get_arg_val<uint32_t>(5);
  int32_t v13 = get_arg_val<uint32_t>(6);
  int32_t v14 = get_arg_val<uint32_t>(7);
  int32_t v15 = get_arg_val<uint32_t>(8);
  DataFormat v16 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v17 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v18;
  v18.bank_base_address = v8;
  v18.page_size = v17;
  v18.data_format = v16;
  InterleavedAddrGenFast<true> v19 = v18;
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v22;
  v22.bank_base_address = v7;
  v22.page_size = v21;
  v22.data_format = v20;
  InterleavedAddrGenFast<true> v23 = v22;
  int32_t v24 = get_arg_val<uint32_t>(9);
  int32_t v25;
  int32_t v26;
  v25 = (int32_t) ((uint32_t) v13 * (uint32_t) 2);
  v26 = (int32_t) ((uint32_t) ((v24 % ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v5)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 + (uint32_t) v1) / v5) - (uint32_t) (v24 / ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v5))) < v2 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v10 + (uint32_t) v1) / v5) - (uint32_t) (v24 / ((int32_t) ((uint32_t) v11 + (uint32_t) v1) / v5))) : v2)) * (uint32_t) v6);
  for (int32_t i27 = v3; i27 < ((int32_t) ((uint32_t) v12 + (uint32_t) v1) / v5); i27 += v2) {
    int32_t v28 = v25;
    int32_t v29 = v26;
    DPRINT << "Get noc addr for a " << v28 << " and b " << v29 << "\n";
    DPRINT << "Tile index a " << (int32_t) ((uint32_t) v28 / (uint32_t) v21) << " tile index b "
           << (int32_t) ((uint32_t) v29 / (uint32_t) v17) << "\n";
    uint64_t temp_202 = v23.get_noc_addr((int32_t) ((uint32_t) v28 / (uint32_t) v21), v3);
    cb_reserve_back(get_compile_time_arg_val(0), v2);
    int32_t v30 = get_write_ptr(get_compile_time_arg_val(0));
    noc_async_read(temp_202, v30, v21);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(0), v2);
    uint64_t temp_211 = v19.get_noc_addr((int32_t) ((uint32_t) v29 / (uint32_t) v17), v3);
    cb_reserve_back(get_compile_time_arg_val(1), v2);
    int32_t v31 = get_write_ptr(get_compile_time_arg_val(1));
    noc_async_read(temp_211, v31, v17);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(1), v2);
    v25 = v6;
    v26 = (int32_t) ((uint32_t) v14 * (uint32_t) v6);
  }
  return;
}
