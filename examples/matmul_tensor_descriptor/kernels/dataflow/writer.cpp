// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 0;
  int32_t v4 = get_arg_val<uint32_t>(0);
  int32_t v5 = get_arg_val<uint32_t>(1);
  int32_t v6 = get_arg_val<uint32_t>(2);
  int32_t v7 = get_arg_val<uint32_t>(3);
  int32_t v8 = get_arg_val<uint32_t>(4);
  int32_t v9 = get_arg_val<uint32_t>(5);
  int32_t v10 = get_arg_val<uint32_t>(6);
  int32_t v11 = get_arg_val<uint32_t>(7);
  int32_t v12 = get_arg_val<uint32_t>(8);
  int32_t v13 = get_arg_val<uint32_t>(9);
  int32_t v14 = get_arg_val<uint32_t>(10);
  int32_t v15 = get_arg_val<uint32_t>(11);
  int32_t v16 = get_arg_val<uint32_t>(12);
  int32_t v17 = get_arg_val<uint32_t>(13);
  int32_t v18 = get_arg_val<uint32_t>(14);
  int32_t v19 = get_arg_val<uint32_t>(15);
  int32_t v20 = get_arg_val<uint32_t>(16);
  int32_t v21 = get_arg_val<uint32_t>(17);
  int32_t v22 = get_arg_val<uint32_t>(18);
  int32_t v23 = get_arg_val<uint32_t>(19);
  int32_t v24 = get_arg_val<uint32_t>(20);
  int32_t v25 = get_arg_val<uint32_t>(21);
  int32_t v26 = get_arg_val<uint32_t>(22);
  int32_t v27 = get_arg_val<uint32_t>(23);
  int32_t v28 = get_arg_val<uint32_t>(24);
  int32_t v29 = get_arg_val<uint32_t>(25);
  int32_t v30 = get_arg_val<uint32_t>(26);
  int32_t v31 = get_arg_val<uint32_t>(27);
  int32_t v32 = get_arg_val<uint32_t>(28);
  int32_t v33 = get_arg_val<uint32_t>(29);
  int32_t v34 = get_arg_val<uint32_t>(30);
  int32_t v35 = get_arg_val<uint32_t>(31);
  int32_t v36 = get_arg_val<uint32_t>(32);
  DataFormat v37 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v38 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v39;
  v39.bank_base_address = v24;
  v39.page_size = v38;
  v39.data_format = v37;
  InterleavedAddrGenFast<true> v40 = v39;
  int32_t v41 = get_arg_val<uint32_t>(34);
  int32_t v42 = get_arg_val<uint32_t>(33);
  for (int32_t i43 = v42; i43 < v41; i43 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v1);
    }
    uint64_t temp_307 = v40.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8)) + (uint32_t) (i43 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) < 8 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) : 8)))) * (uint32_t) 32) / 32) * (uint32_t) (v26 != (int32_t) ((uint32_t) (v26 / 32) * (uint32_t) 32) & v26 < v3 == false ? (int32_t) ((uint32_t) (v26 / 32) + (uint32_t) v1) : v26 / 32))) + (uint32_t) ((int32_t) ((uint32_t) ((i43 % (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) < 8 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) : 8)) * (uint32_t) 32) / 32))) * (uint32_t) 1024)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8)) + (uint32_t) (i43 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) < 8 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) : 8)))) * (uint32_t) 32)) % (uint32_t) 32)) * (uint32_t) 32)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i43 % (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) < 8 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 + (uint32_t) 31) / 32) - (uint32_t) ((int32_t) ((uint32_t) (i43 / (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) 31) / 32) * (uint32_t) 8)) * (uint32_t) 8))) : 8)) * (uint32_t) 32)) % (uint32_t) 32)))))) * (uint32_t) 2)) / (uint32_t) v38), v3);
    int32_t v44 = get_read_ptr(get_compile_time_arg_val(2));
    noc_async_write(v44, temp_307, v38);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v1);
  }
  return;
}
