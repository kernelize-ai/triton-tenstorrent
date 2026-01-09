// matmul_kernel_tma__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  int32_t v2 = 63;
  int32_t v3 = 0;
  int32_t v4 = 64;
  int32_t v5 = 32;
  int32_t v6 = 1;
  bool v7 = true;
  int32_t v8 = 2;
  int32_t v9 = 4;
  int32_t v10 = get_arg_val<uint32_t>(0);
  int32_t v11 = get_arg_val<uint32_t>(2);
  int32_t v12 = get_arg_val<uint32_t>(10);
  int32_t v13 = get_arg_val<uint32_t>(12);
  int32_t v14 = get_arg_val<uint32_t>(30);
  int32_t v15 = get_arg_val<uint32_t>(31);
  int32_t v16 = get_arg_val<uint32_t>(32);
  DataFormat v17 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v18 = get_tile_size(get_compile_time_arg_val(1));
  InterleavedAddrGenFast<true> v19;
  v19.bank_base_address = v12;
  v19.page_size = v18;
  v19.data_format = v17;
  InterleavedAddrGenFast<true> v20 = v19;
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(0));
  InterleavedAddrGenFast<true> v23;
  v23.bank_base_address = v10;
  v23.page_size = v22;
  v23.data_format = v21;
  InterleavedAddrGenFast<true> v24 = v23;
  int32_t v25 = get_arg_val<uint32_t>(34);
  int32_t v26 = get_arg_val<uint32_t>(33);
  int32_t v27 = (int32_t) ((uint32_t) v15 + (uint32_t) v2) / v4;
  for (int32_t i28 = v26; i28 < v25; i28 += v6) {
    int32_t v29 = i28 / v27;
    int32_t v30 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) 31) / v5) - (uint32_t) v29) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 + (uint32_t) 31) / v5) - (uint32_t) v29) : v6;
    for (int32_t j31 = v3; j31 < ((int32_t) ((uint32_t) v16 + (uint32_t) v2) / v4); j31 += v6) {
      int32_t v32 = (int32_t) ((uint32_t) j31 * (uint32_t) v4);
      int32_t v33 = v32 / v4;
      int32_t v34 = (int32_t) ((uint32_t) v32 % (uint32_t) v4);
      cb_reserve_back(get_compile_time_arg_val(0), v8);
      int32_t v35 = get_write_ptr(get_compile_time_arg_val(0));
      int32_t v36;
      int32_t v37;
      v36 = v35;
      v37 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v29 + (uint32_t) (i28 % v30))) * (uint32_t) v5) / v5) * (uint32_t) (v11 != (int32_t) ((uint32_t) (v11 / v4) * (uint32_t) v4) & v11 < v3 == v1 ? (int32_t) ((uint32_t) (v11 / v4) + (uint32_t) v6) : v11 / v4))) + (uint32_t) v33)) * (uint32_t) 2048)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v29 + (uint32_t) (i28 % v30))) * (uint32_t) v5)) % (uint32_t) v5)) * (uint32_t) v4)) + (uint32_t) v34)))) * (uint32_t) v8)) / (uint32_t) v22);
      for (int32_t k38 = v3; k38 < v8; k38 += v6) {
        int32_t v39 = v36;
        int32_t v40 = v37;
        uint64_t temp_371 = v24.get_noc_addr(v40, v3);
        noc_async_read(temp_371, v39, v22);
        v36 = (int32_t) ((uint32_t) v39 + (uint32_t) v22);
        v37 = (int32_t) ((uint32_t) v40 + (uint32_t) v6);
      }
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(0), v8);
      cb_reserve_back(get_compile_time_arg_val(1), v9);
      int32_t v41 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v42;
      int32_t v43;
      v42 = v41;
      v43 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v33 * (uint32_t) (v13 != (int32_t) ((uint32_t) (v13 / v4) * (uint32_t) v4) & v13 < v3 == v1 ? (int32_t) ((uint32_t) (v13 / v4) + (uint32_t) v6) : v13 / v4))) + (uint32_t) ((int32_t) ((uint32_t) ((i28 % v27) / v30) * (uint32_t) v4) / v4))) * (uint32_t) 4096)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v34 * (uint32_t) v4)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((i28 % v27) / v30) * (uint32_t) v4)) % (uint32_t) v4)))))) * (uint32_t) v8)) / (uint32_t) v18);
      int32_t tiles_per_row_B = (v13 + 31) / 32;   // N in elements, 32 elements per tile col
#if 1
      uint32_t pid_n = (i28 % v27) / v30;

      uint32_t N = v15;
      uint32_t Nt_tiles = N / 32;        // since you assert multiples of 32
      uint32_t BNt = 64 / 32;            // 2 (block N in tiles)
      uint32_t BKt = 64 / 32;            // 2 (block K in tiles)

      uint32_t n0 = pid_n * BNt;         // pid_n is your block column id
      uint32_t k_iter = j31;
      uint32_t k0 = k_iter * BKt;        // k_iter is your 64-wide K block id

      int32_t dst = v42;
      for (uint32_t r = 0; r < BKt; r++) {
        for (uint32_t c = 0; c < BNt; c++) {
        
        uint32_t tileIndex = (k0 + r) * Nt_tiles + (n0 + c);
        noc_async_read(v20.get_noc_addr(tileIndex, 0), dst, v18);
        dst += v18;
        }
      }
#else
      for (int32_t k44 = v3; k44 < v9; k44 += v6) {
        int32_t v45 = v42;
        int32_t v46 = v43;
        DPRINT << "Reading tile k44=" << k44 << " to L1 addr " << v45 << " from tile id " << v46 << "\n";
        uint64_t temp_405 = v20.get_noc_addr(v46, v3);
        noc_async_read(temp_405, v45, v18);
        v42 = (int32_t) ((uint32_t) v45 + (uint32_t) v18);
        v43 = (int32_t) ((uint32_t) v46 + (uint32_t) v6);
      }
#endif
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v9);
    }
  }
  return;
}
