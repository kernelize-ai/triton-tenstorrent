// matmul_kernel__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  int32_t v2 = 0;
  int32_t v3 = 32;
  int32_t v4 = 31;
  bool v5 = true;
  int32_t v6 = 2;
  int32_t v7 = 64;
  int32_t v8 = get_arg_val<uint32_t>(0);
  int32_t v9 = get_arg_val<uint32_t>(1);
  int32_t v10 = get_arg_val<uint32_t>(2);
  int32_t v11 = get_arg_val<uint32_t>(3);
  int32_t v12 = get_arg_val<uint32_t>(4);
  int32_t v13 = get_arg_val<uint32_t>(5);
  int32_t v14 = get_arg_val<uint32_t>(6);
  int32_t v15 = get_arg_val<uint32_t>(7);
  int32_t v16 = get_arg_val<uint32_t>(8);
  int32_t v17 = get_arg_val<uint32_t>(9);
  int32_t v18 = get_arg_val<uint32_t>(10);
  int32_t v19 = get_arg_val<uint32_t>(11);
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v22;
  v22.bank_base_address = v9;
  v22.page_size = v21;
  v22.data_format = v20;
  InterleavedAddrGenFast<true> v23 = v22;
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v26;
  v26.bank_base_address = v8;
  v26.page_size = v25;
  v26.data_format = v24;
  InterleavedAddrGenFast<true> v27 = v26;
  int32_t v28 = get_arg_val<uint32_t>(12);
  int32_t v29 = (int32_t) ((uint32_t) v12 + (uint32_t) v4) / v3;
  int32_t v30 = v28 / v29;
  int32_t v31 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v4) / v3) - (uint32_t) v30) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v4) / v3) - (uint32_t) v30) : v1;
  int32_t v32 = v28 % v29;
  int32_t v33;
  int32_t v34;
  v33 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v30 + (uint32_t) (v32 % v31))) * (uint32_t) v3) % v11) * (uint32_t) v14)) * (uint32_t) v6);
  v34 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v32 / v31) * (uint32_t) v3) % v12) * (uint32_t) v17)) * (uint32_t) v6);
  for (int32_t i35 = v2; i35 < ((int32_t) ((uint32_t) v13 + (uint32_t) v4) / v3); i35 += v1) {
    int32_t v36 = v33;
    int32_t v37 = v34;
    DPRINT << "A matrix offset = " << v36 << " / " << v25 << " = " << (v36 / v25) << "\n";
    DPRINT << "B matrix offset = " << v37 << " / " << v21 << " = " << (v37 / v21) << "\n";
    uint64_t temp_276 = v27.get_noc_addr((int32_t) ((uint32_t) v36 / (uint32_t) v25), v2);
    cb_reserve_back(get_compile_time_arg_val(0), v1);
    int32_t v38 = get_write_ptr(get_compile_time_arg_val(0));
    noc_async_read(temp_276, v38, v25);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(0), v1);
    uint64_t temp_285 = v23.get_noc_addr((int32_t) ((uint32_t) v37 / (uint32_t) v21), v2);
    cb_reserve_back(get_compile_time_arg_val(1), v1);
    int32_t v39 = get_write_ptr(get_compile_time_arg_val(1));
    noc_async_read(temp_285, v39, v21);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(1), v1);
    v33 = (int32_t) ((uint32_t) v36 + (uint32_t) ((int32_t) ((uint32_t) v15 * (uint32_t) v7)));
    v34 = (int32_t) ((uint32_t) v37 + (uint32_t) ((int32_t) ((uint32_t) v16 * (uint32_t) v7)));
  }
  return;
}
