// matmul_kernel_tma__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 1;
  bool v2 = true;
  int32_t v3 = 8;
  int32_t v4 = 0;
  int32_t v5 = 2;
  int32_t v6 = 4;
  int32_t v7 = get_common_arg_val<uint32_t>(20);
  int32_t v8 = get_common_arg_val<uint32_t>(22);
  int32_t v9 = get_common_arg_val<uint32_t>(30);
  int32_t v10 = get_common_arg_val<uint32_t>(31);
  DataFormat v11 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v12 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v13;
  v13.bank_base_address = v7;
  v13.page_size = v12;
  v13.data_format = v11;
  InterleavedAddrGenFast<true> v14 = v13;
  int32_t v15 = get_arg_val<uint32_t>(1);
  int32_t v16 = get_arg_val<uint32_t>(0);
  for (int32_t i17 = v16; i17 < v15; i17 += v1) {
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v3);
    }
    int32_t v18 = get_read_ptr(get_compile_time_arg_val(2));
    for (int32_t j19 = v4; j19 < v5; j19 += v1) {
      for (int32_t k20 = v4; k20 < v6; k20 += v1) {
        int32_t v21 = (int32_t) ((uint32_t) v18 + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j19 & (uint32_t) v1)) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) k20 & (uint32_t) v1)) + (uint32_t) ((int32_t) ((uint32_t) k20 & (uint32_t) v5)))))) * (uint32_t) v12)));
        uint64_t temp_199 = v14.get_noc_addr((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i17 / ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128)) + (uint32_t) (i17 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128))) : v1)))) * (uint32_t) 64) / 32) + (uint32_t) j19)) * (uint32_t) (v8 != (int32_t) ((uint32_t) (v8 / 32) * (uint32_t) 32) & v8 < v4 == false ? (int32_t) ((uint32_t) (v8 / 32) + (uint32_t) v1) : v8 / 32))) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i17 % ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128))) < v1 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) 63) / 64) - (uint32_t) (i17 / ((int32_t) ((uint32_t) v10 + (uint32_t) 127) / 128))) : v1)) * (uint32_t) 128) / 32) + (uint32_t) k20))), v4);
        noc_async_write(v21, temp_199, v12);
      }
    }
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v3);
  }
  return;
}
