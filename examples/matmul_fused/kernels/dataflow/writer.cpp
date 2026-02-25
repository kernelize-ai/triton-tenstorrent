// matmul_kernel_fused__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DATAFLOW_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DATAFLOW_API_H

namespace experimental {

FORCE_INLINE
std::uint64_t
get_noc_multicast_addr(std::uint32_t noc_x_start, std::uint32_t noc_y_start,
                       std::uint32_t noc_x_end, std::uint32_t noc_y_end,
                       std::uint32_t addr, uint8_t noc = noc_index) {
  /*
      Get an encoding which contains tensix core and address you want to
      read from/write to via the noc
  */
  if (noc) {
    // noc 1
    return NOC_MULTICAST_ADDR(
        DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end),
        DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start), addr);
  } else {
    // noc 0
    return NOC_MULTICAST_ADDR(
        DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start),
        DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end), addr);
  }
}

} // namespace experimental

#endif


// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_COORD_TRANSLATION_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_COORD_TRANSLATION_H

namespace experimental {

FORCE_INLINE
std::uint32_t convert_logical_x_to_translated(std::uint32_t logical_x) {
  /*
      convert x coord from LOGICAL to TRANSLATED coordinate system
  */
  return worker_logical_col_to_virtual_col[logical_x];
}

FORCE_INLINE
std::uint32_t convert_logical_y_to_translated(std::uint32_t logical_y) {
  /*
      convert x coord from LOGICAL to TRANSLATED coordinate system
  */
  return worker_logical_row_to_virtual_row[logical_y];
}

} // namespace experimental

#endif

void kernel_main() {
  bool v1 = false;
  uint32_t v2 = 0;
  uint32_t v3 = 32;
  size_t v4 = 0;
  size_t v5 = 1;
  size_t v6 = 2;
  int32_t v7 = 63;
  int32_t v8 = 0;
  int32_t v9 = 256;
  int32_t v10 = 64;
  int32_t v11 = 1;
  bool v12 = true;
  int32_t v13 = 65535;
  int32_t v14 = 16;
  int32_t v15 = 32;
  int32_t v16 = 2048;
  int32_t v17 = 4096;
  int32_t v18 = 6144;
  int32_t v19 = 4;
  int32_t v20 = 32768;
  int32_t v21 = get_common_arg_val<uint32_t>(v4);
  int32_t v22 = get_common_arg_val<uint32_t>(v6);
  int32_t v23 = get_common_arg_val<uint32_t>(20);
  int32_t v24 = get_common_arg_val<uint32_t>(22);
  int32_t v25 = get_common_arg_val<uint32_t>(40);
  int32_t v26 = get_common_arg_val<uint32_t>(41);
  int32_t v27 = get_common_arg_val<uint32_t>(42);
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(3));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(3));
  InterleavedAddrGenFast<true> v30;
  v30.bank_base_address = v23;
  v30.page_size = v29;
  v30.data_format = v28;
  InterleavedAddrGenFast<true> v31 = v30;
  DataFormat v32 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v33 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v34 = get_arg_val<uint32_t>(v5);
  int32_t v35 = get_arg_val<uint32_t>(v4);
  int32_t v36 = (int32_t) ((uint32_t) v26 + (uint32_t) v7) / v10;
  int32_t v37 = get_arg_val<uint32_t>(v6);
  int32_t v38 = get_arg_val<uint32_t>(3);
  int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 < v3 ? (uint32_t) v37 >> (uint32_t) v14 : v2)) & (uint32_t) v13);
  int32_t v40 = (int32_t) ((uint32_t) v37 & (uint32_t) v13);
  int32_t v41 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v14 < v3 ? (uint32_t) v38 >> (uint32_t) v14 : v2)) & (uint32_t) v13);
  int32_t v42 = (int32_t) ((uint32_t) v38 & (uint32_t) v13);
  int32_t v43 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v41 + (uint32_t) v11)) * (uint32_t) ((int32_t) ((uint32_t) v42 + (uint32_t) v11)))) - (uint32_t) v11);
  int32_t v44 = v24 != (int32_t) ((uint32_t) (v24 / v15) * (uint32_t) v15) & v24 < v8 == v1 ? (int32_t) ((uint32_t) (v24 / v15) + (uint32_t) v11) : v24 / v15;
  for (int32_t i45 = v35; i45 < v34; i45 += v11) {
    int32_t v46 = i45 / v36;
    int32_t v47 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v25 + (uint32_t) v7) / v10) - (uint32_t) v46) < v11 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v25 + (uint32_t) v7) / v10) - (uint32_t) v46) : v11;
    int32_t v48 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v46 + (uint32_t) (i45 % v47))) * (uint32_t) v10);
    for (int32_t j49 = v8; j49 < ((int32_t) ((uint32_t) v27 + (uint32_t) 255) / v9); j49 += v11) {
      int32_t v50 = get_semaphore(get_compile_time_arg_val(4));
      int32_t v51 = get_semaphore(get_compile_time_arg_val(5));
      if (v39 == (int32_t) ((ptrdiff_t) get_absolute_logical_x()) & v40 == (int32_t) ((ptrdiff_t) get_absolute_logical_y())) {
        cb_reserve_back(get_compile_time_arg_val(0), v14);
        InterleavedAddrGenFast<true> v52;
        v52.bank_base_address = v21;
        v52.page_size = v33;
        v52.data_format = v32;
        InterleavedAddrGenFast<true> v53 = v52;
        int32_t v54 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v55 = v22 != (int32_t) ((uint32_t) (v22 / v15) * (uint32_t) v15) & v22 < v8 == v1 ? (int32_t) ((uint32_t) (v22 / v15) + (uint32_t) v11) : v22 / v15;
        int32_t v56 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v48 / v15) * (uint32_t) v55)) + (uint32_t) ((int32_t) ((uint32_t) j49 * (uint32_t) v9) / v15));
        uint64_t temp_359 = v53.get_noc_addr(v56, v8);
        noc_async_read(temp_359, v54, v33);
        int32_t v57 = (int32_t) ((uint32_t) v54 + (uint32_t) v16);
        int32_t v58 = (int32_t) ((uint32_t) v56 + (uint32_t) v11);
        uint64_t temp_371 = v53.get_noc_addr(v58, v8);
        noc_async_read(temp_371, v57, v33);
        int32_t v59 = (int32_t) ((uint32_t) v54 + (uint32_t) v17);
        int32_t v60 = (int32_t) ((uint32_t) v56 + (uint32_t) 2);
        uint64_t temp_383 = v53.get_noc_addr(v60, v8);
        noc_async_read(temp_383, v59, v33);
        int32_t v61 = (int32_t) ((uint32_t) v54 + (uint32_t) v18);
        int32_t v62 = (int32_t) ((uint32_t) v56 + (uint32_t) 3);
        uint64_t temp_395 = v53.get_noc_addr(v62, v8);
        noc_async_read(temp_395, v61, v33);
        int32_t v63 = (int32_t) ((uint32_t) v54 + (uint32_t) 8192);
        int32_t v64 = (int32_t) ((uint32_t) v56 + (uint32_t) v19);
        uint64_t temp_407 = v53.get_noc_addr(v64, v8);
        noc_async_read(temp_407, v63, v33);
        int32_t v65 = (int32_t) ((uint32_t) v54 + (uint32_t) 10240);
        int32_t v66 = (int32_t) ((uint32_t) v56 + (uint32_t) 5);
        uint64_t temp_419 = v53.get_noc_addr(v66, v8);
        noc_async_read(temp_419, v65, v33);
        int32_t v67 = (int32_t) ((uint32_t) v54 + (uint32_t) 12288);
        int32_t v68 = (int32_t) ((uint32_t) v56 + (uint32_t) 6);
        uint64_t temp_431 = v53.get_noc_addr(v68, v8);
        noc_async_read(temp_431, v67, v33);
        int32_t v69 = (int32_t) ((uint32_t) v54 + (uint32_t) 14336);
        int32_t v70 = (int32_t) ((uint32_t) v56 + (uint32_t) 7);
        uint64_t temp_443 = v53.get_noc_addr(v70, v8);
        noc_async_read(temp_443, v69, v33);
        int32_t v71 = (int32_t) ((uint32_t) v54 + (uint32_t) 16384);
        uint64_t temp_455 = v53.get_noc_addr((int32_t) ((uint32_t) v56 + (uint32_t) v55), v8);
        noc_async_read(temp_455, v71, v33);
        int32_t v72 = (int32_t) ((uint32_t) v54 + (uint32_t) 18432);
        uint64_t temp_467 = v53.get_noc_addr((int32_t) ((uint32_t) v58 + (uint32_t) v55), v8);
        noc_async_read(temp_467, v72, v33);
        int32_t v73 = (int32_t) ((uint32_t) v54 + (uint32_t) 20480);
        uint64_t temp_479 = v53.get_noc_addr((int32_t) ((uint32_t) v60 + (uint32_t) v55), v8);
        noc_async_read(temp_479, v73, v33);
        int32_t v74 = (int32_t) ((uint32_t) v54 + (uint32_t) 22528);
        uint64_t temp_491 = v53.get_noc_addr((int32_t) ((uint32_t) v62 + (uint32_t) v55), v8);
        noc_async_read(temp_491, v74, v33);
        int32_t v75 = (int32_t) ((uint32_t) v54 + (uint32_t) 24576);
        uint64_t temp_503 = v53.get_noc_addr((int32_t) ((uint32_t) v64 + (uint32_t) v55), v8);
        noc_async_read(temp_503, v75, v33);
        int32_t v76 = (int32_t) ((uint32_t) v54 + (uint32_t) 26624);
        uint64_t temp_515 = v53.get_noc_addr((int32_t) ((uint32_t) v66 + (uint32_t) v55), v8);
        noc_async_read(temp_515, v76, v33);
        int32_t v77 = (int32_t) ((uint32_t) v54 + (uint32_t) 28672);
        uint64_t temp_527 = v53.get_noc_addr((int32_t) ((uint32_t) v68 + (uint32_t) v55), v8);
        noc_async_read(temp_527, v77, v33);
        int32_t v78 = (int32_t) ((uint32_t) v54 + (uint32_t) 30720);
        uint64_t temp_539 = v53.get_noc_addr((int32_t) ((uint32_t) v70 + (uint32_t) v55), v8);
        noc_async_read(temp_539, v78, v33);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        int32_t v79 = get_write_ptr(get_compile_time_arg_val(0));
        volatile tt_l1_ptr uint32_t* v80 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v50);
        noc_semaphore_wait(v80, v43);
        noc_semaphore_set(v80, v4);
        size_t v81 = experimental::convert_logical_y_to_translated(v42);
        size_t v82 = experimental::convert_logical_x_to_translated(v41);
        size_t v83 = experimental::convert_logical_y_to_translated(v40);
        size_t v84 = experimental::convert_logical_x_to_translated(v39);
        int64_t v85 = experimental::get_noc_multicast_addr(v84, v83, v82, v81, v79);
        noc_async_write_multicast(v79, v85, v20, v43);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v86 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v51);
        noc_semaphore_set(v86, v5);
        size_t v87 = experimental::convert_logical_y_to_translated(v42);
        size_t v88 = experimental::convert_logical_x_to_translated(v41);
        size_t v89 = experimental::convert_logical_y_to_translated(v40);
        size_t v90 = experimental::convert_logical_x_to_translated(v39);
        int64_t v91 = experimental::get_noc_multicast_addr(v90, v89, v88, v87, v51);
        noc_semaphore_set_multicast(v51, v91, v43);
      } else {
        cb_reserve_back(get_compile_time_arg_val(0), v14);
        size_t v92 = experimental::convert_logical_y_to_translated(v40);
        size_t v93 = experimental::convert_logical_x_to_translated(v39);
        int64_t v94 = get_noc_addr(v93, v92, v50);
        noc_semaphore_inc(v94, v5);
        volatile tt_l1_ptr uint32_t* v95 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v51);
        noc_semaphore_wait(v95, v5);
        noc_semaphore_set(v95, v4);
      }
      cb_push_back(get_compile_time_arg_val(0), v14);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(3), v19);
    }
    int32_t v96 = get_read_ptr(get_compile_time_arg_val(3));
    int32_t v97 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v48 / v15) * (uint32_t) v44)) + (uint32_t) ((int32_t) ((uint32_t) ((i45 % v36) / v47) * (uint32_t) v10) / v15));
    uint64_t temp_296 = v31.get_noc_addr(v97, v8);
    noc_async_write(v96, temp_296, v29);
    int32_t v98 = (int32_t) ((uint32_t) v96 + (uint32_t) v16);
    int32_t v99 = (int32_t) ((uint32_t) v97 + (uint32_t) v11);
    uint64_t temp_308 = v31.get_noc_addr(v99, v8);
    noc_async_write(v98, temp_308, v29);
    int32_t v100 = (int32_t) ((uint32_t) v96 + (uint32_t) v17);
    uint64_t temp_320 = v31.get_noc_addr((int32_t) ((uint32_t) v97 + (uint32_t) v44), v8);
    noc_async_write(v100, temp_320, v29);
    int32_t v101 = (int32_t) ((uint32_t) v96 + (uint32_t) v18);
    uint64_t temp_332 = v31.get_noc_addr((int32_t) ((uint32_t) v99 + (uint32_t) v44), v8);
    noc_async_write(v101, temp_332, v29);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(3), v19);
  }
  return;
}
