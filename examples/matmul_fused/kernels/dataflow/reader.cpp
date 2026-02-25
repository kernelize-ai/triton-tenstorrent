// matmul_kernel_fused__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  bool v1 = false;
  int32_t v2 = 63;
  int32_t v3 = 0;
  int32_t v4 = 256;
  int32_t v5 = 64;
  int32_t v6 = 1;
  int32_t v7 = 16;
  bool v8 = true;
  int32_t v9 = 32;
  int32_t v10 = 2048;
  int32_t v11 = 4096;
  int32_t v12 = 6144;
  int32_t v13 = 4;
  int32_t v14 = get_common_arg_val<uint32_t>(10);
  int32_t v15 = get_common_arg_val<uint32_t>(12);
  int32_t v16 = get_common_arg_val<uint32_t>(30);
  int32_t v17 = get_common_arg_val<uint32_t>(32);
  int32_t v18 = get_common_arg_val<uint32_t>(40);
  int32_t v19 = get_common_arg_val<uint32_t>(41);
  int32_t v20 = get_common_arg_val<uint32_t>(42);
  DataFormat v21 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(2));
  DataFormat v23 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v24 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v25 = get_arg_val<uint32_t>(1);
  int32_t v26 = get_arg_val<uint32_t>(0);
  int32_t v27 = (int32_t) ((uint32_t) v19 + (uint32_t) v2) / v5;
  int32_t v28 = v15 != (int32_t) ((uint32_t) (v15 / v9) * (uint32_t) v9) & v15 < v3 == v1 ? (int32_t) ((uint32_t) (v15 / v9) + (uint32_t) v6) : v15 / v9;
  int32_t v29 = (int32_t) ((uint32_t) v28 * (uint32_t) 2);
  int32_t v30 = (int32_t) ((uint32_t) v28 * (uint32_t) 3);
  int32_t v31 = (int32_t) ((uint32_t) v28 * (uint32_t) v13);
  int32_t v32 = (int32_t) ((uint32_t) v28 * (uint32_t) 5);
  int32_t v33 = (int32_t) ((uint32_t) v28 * (uint32_t) 6);
  int32_t v34 = (int32_t) ((uint32_t) v28 * (uint32_t) 7);
  int32_t v35 = v17 != (int32_t) ((uint32_t) (v17 / v9) * (uint32_t) v9) & v17 < v3 == v1 ? (int32_t) ((uint32_t) (v17 / v9) + (uint32_t) v6) : v17 / v9;
  for (int32_t i36 = v26; i36 < v25; i36 += v6) {
    int32_t v37 = i36 / v27;
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) v2) / v5) - (uint32_t) v37) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) v2) / v5) - (uint32_t) v37) : v6;
    int32_t v39 = (int32_t) ((uint32_t) ((i36 % v27) / v38) * (uint32_t) v5) / v9;
    for (int32_t j40 = v3; j40 < ((int32_t) ((uint32_t) v20 + (uint32_t) 255) / v4); j40 += v6) {
      cb_reserve_back(get_compile_time_arg_val(1), v7);
      InterleavedAddrGenFast<true> v41;
      v41.bank_base_address = v14;
      v41.page_size = v24;
      v41.data_format = v23;
      InterleavedAddrGenFast<true> v42 = v41;
      int32_t v43 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v44 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j40 * (uint32_t) v4) / v9) * (uint32_t) v28)) + (uint32_t) v39);
      uint64_t temp_531 = v42.get_noc_addr(v44, v3);
      noc_async_read(temp_531, v43, v24);
      int32_t v45 = (int32_t) ((uint32_t) v43 + (uint32_t) 16384);
      int32_t v46 = (int32_t) ((uint32_t) v44 + (uint32_t) v6);
      uint64_t temp_543 = v42.get_noc_addr(v46, v3);
      noc_async_read(temp_543, v45, v24);
      int32_t v47 = (int32_t) ((uint32_t) v43 + (uint32_t) v10);
      uint64_t temp_555 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v28), v3);
      noc_async_read(temp_555, v47, v24);
      int32_t v48 = (int32_t) ((uint32_t) v43 + (uint32_t) 18432);
      uint64_t temp_567 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v28), v3);
      noc_async_read(temp_567, v48, v24);
      int32_t v49 = (int32_t) ((uint32_t) v43 + (uint32_t) v11);
      uint64_t temp_579 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v29), v3);
      noc_async_read(temp_579, v49, v24);
      int32_t v50 = (int32_t) ((uint32_t) v43 + (uint32_t) 20480);
      uint64_t temp_591 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v29), v3);
      noc_async_read(temp_591, v50, v24);
      int32_t v51 = (int32_t) ((uint32_t) v43 + (uint32_t) v12);
      uint64_t temp_603 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v30), v3);
      noc_async_read(temp_603, v51, v24);
      int32_t v52 = (int32_t) ((uint32_t) v43 + (uint32_t) 22528);
      uint64_t temp_615 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v30), v3);
      noc_async_read(temp_615, v52, v24);
      int32_t v53 = (int32_t) ((uint32_t) v43 + (uint32_t) 8192);
      uint64_t temp_627 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v31), v3);
      noc_async_read(temp_627, v53, v24);
      int32_t v54 = (int32_t) ((uint32_t) v43 + (uint32_t) 24576);
      uint64_t temp_639 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v31), v3);
      noc_async_read(temp_639, v54, v24);
      int32_t v55 = (int32_t) ((uint32_t) v43 + (uint32_t) 10240);
      uint64_t temp_651 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v32), v3);
      noc_async_read(temp_651, v55, v24);
      int32_t v56 = (int32_t) ((uint32_t) v43 + (uint32_t) 26624);
      uint64_t temp_663 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v32), v3);
      noc_async_read(temp_663, v56, v24);
      int32_t v57 = (int32_t) ((uint32_t) v43 + (uint32_t) 12288);
      uint64_t temp_675 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v33), v3);
      noc_async_read(temp_675, v57, v24);
      int32_t v58 = (int32_t) ((uint32_t) v43 + (uint32_t) 28672);
      uint64_t temp_687 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v33), v3);
      noc_async_read(temp_687, v58, v24);
      int32_t v59 = (int32_t) ((uint32_t) v43 + (uint32_t) 14336);
      uint64_t temp_699 = v42.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v34), v3);
      noc_async_read(temp_699, v59, v24);
      int32_t v60 = (int32_t) ((uint32_t) v43 + (uint32_t) 30720);
      uint64_t temp_711 = v42.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v34), v3);
      noc_async_read(temp_711, v60, v24);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v7);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v13);
    InterleavedAddrGenFast<true> v61;
    v61.bank_base_address = v16;
    v61.page_size = v22;
    v61.data_format = v21;
    InterleavedAddrGenFast<true> v62 = v61;
    int32_t v63 = get_write_ptr(get_compile_time_arg_val(2));
    int32_t v64 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v37 + (uint32_t) (i36 % v38))) * (uint32_t) v5) / v9) * (uint32_t) v35)) + (uint32_t) v39);
    uint64_t temp_511 = v62.get_noc_addr(v64, v3);
    noc_async_read(temp_511, v63, v22);
    int32_t v65 = (int32_t) ((uint32_t) v63 + (uint32_t) v10);
    int32_t v66 = (int32_t) ((uint32_t) v64 + (uint32_t) v6);
    uint64_t temp_523 = v62.get_noc_addr(v66, v3);
    noc_async_read(temp_523, v65, v22);
    int32_t v67 = (int32_t) ((uint32_t) v63 + (uint32_t) v11);
    uint64_t temp_535 = v62.get_noc_addr((int32_t) ((uint32_t) v64 + (uint32_t) v35), v3);
    noc_async_read(temp_535, v67, v22);
    int32_t v68 = (int32_t) ((uint32_t) v63 + (uint32_t) v12);
    uint64_t temp_547 = v62.get_noc_addr((int32_t) ((uint32_t) v66 + (uint32_t) v35), v3);
    noc_async_read(temp_547, v68, v22);
    {
    DeviceZoneScopedN("noc_async_read_barrier");
    noc_async_read_barrier();
    }
    cb_push_back(get_compile_time_arg_val(2), v13);
  }
  return;
}
