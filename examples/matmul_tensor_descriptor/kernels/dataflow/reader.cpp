// matmul_kernel_tma__reader
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
  size_t v2 = get_absolute_logical_y();
  size_t v3 = get_absolute_logical_x();
  size_t v4 = 0;
  size_t v5 = 1;
  size_t v6 = 4;
  int32_t v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 128;
  int32_t v10 = 32;
  int32_t v11 = 2;
  int32_t v12 = 1;
  bool v13 = true;
  int32_t v14 = 2048;
  int32_t v15 = 4096;
  int32_t v16 = 8;
  int32_t v17 = 16384;
  size_t v18 = 19;
  int32_t v19 = 19;
  int32_t v20 = get_common_arg_val<uint32_t>(v4);
  int32_t v21 = get_common_arg_val<uint32_t>(2);
  int32_t v22 = get_common_arg_val<uint32_t>(10);
  int32_t v23 = get_common_arg_val<uint32_t>(12);
  int32_t v24 = get_common_arg_val<uint32_t>(30);
  int32_t v25 = get_common_arg_val<uint32_t>(31);
  int32_t v26 = get_common_arg_val<uint32_t>(32);
  DataFormat v27 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v28 = get_tile_size(get_compile_time_arg_val(1));
  DataFormat v29 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v30 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v31 = get_arg_val<uint32_t>(v5);
  int32_t v32 = get_arg_val<uint32_t>(v4);
  int32_t v33 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v25 + (uint32_t) 127) / v9) * (uint32_t) v11);
  bool v34 = v3 < v6;
  size_t v35 = v34 ? v4 : v6;
  size_t v36 = v34 ? 3 : 7;
  size_t v37 = v3 + v6;
  for (int32_t i38 = v32; i38 < v31; i38 += v12) {
    int32_t v39 = (int32_t) ((uint32_t) (i38 / v33) * (uint32_t) v11);
    int32_t v40 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v24 + (uint32_t) 31) / v10) - (uint32_t) v39) < v11 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v24 + (uint32_t) 31) / v10) - (uint32_t) v39) : v11;
    for (int32_t j41 = v7; j41 < ((int32_t) ((uint32_t) v26 + (uint32_t) 63) / v8); j41 += v12) {
      int32_t v42 = (int32_t) ((uint32_t) j41 * (uint32_t) v8);
      cb_reserve_back(get_compile_time_arg_val(0), v11);
      int32_t v43 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v44 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) v2 == (ptrdiff_t) v4 & (v34 & (ptrdiff_t) v3 == (ptrdiff_t) v4 | v3 >= v6 & (ptrdiff_t) v3 == (ptrdiff_t) v6)) {
        InterleavedAddrGenFast<true> v45;
        v45.bank_base_address = v20;
        v45.page_size = v30;
        v45.data_format = v29;
        InterleavedAddrGenFast<true> v46 = v45;
        int32_t v47 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v48 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) (i38 % v40))) * (uint32_t) v10) / v10) * (uint32_t) (v21 != (int32_t) ((uint32_t) (v21 / v10) * (uint32_t) v10) & v21 < v7 == v1 ? (int32_t) ((uint32_t) (v21 / v10) + (uint32_t) v12) : v21 / v10))) + (uint32_t) (v42 / v10));
        uint64_t temp_415 = v46.get_noc_addr(v48, v7);
        noc_async_read(temp_415, v47, v30);
        int32_t v49 = (int32_t) ((uint32_t) v47 + (uint32_t) v14);
        uint64_t temp_427 = v46.get_noc_addr((int32_t) ((uint32_t) v48 + (uint32_t) v12), v7);
        noc_async_read(temp_427, v49, v30);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v50 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v43);
        noc_semaphore_wait(v50, v18);
        noc_semaphore_set(v50, v4);
        size_t v51 = experimental::convert_logical_y_to_translated(v6);
        size_t v52 = experimental::convert_logical_x_to_translated(v36);
        size_t v53 = experimental::convert_logical_y_to_translated(v4);
        size_t v54 = experimental::convert_logical_x_to_translated(v35);
        int64_t v55 = experimental::get_noc_multicast_addr(v54, v53, v52, v51, v47);
        noc_async_write_multicast(v47, v55, v15, v19);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v56 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v44);
        noc_semaphore_set(v56, v5);
        size_t v57 = experimental::convert_logical_y_to_translated(v6);
        size_t v58 = experimental::convert_logical_x_to_translated(v36);
        size_t v59 = experimental::convert_logical_y_to_translated(v4);
        size_t v60 = experimental::convert_logical_x_to_translated(v35);
        int64_t v61 = experimental::get_noc_multicast_addr(v60, v59, v58, v57, v44);
        noc_semaphore_set_multicast(v44, v61, v19);
      } else {
        size_t v62 = experimental::convert_logical_y_to_translated(v4);
        size_t v63 = experimental::convert_logical_x_to_translated(v35);
        int64_t v64 = get_noc_addr(v63, v62, v43);
        noc_semaphore_inc(v64, v5);
        volatile tt_l1_ptr uint32_t* v65 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v44);
        noc_semaphore_wait(v65, v5);
        noc_semaphore_set(v65, v4);
      }
      cb_push_back(get_compile_time_arg_val(0), v11);
      cb_reserve_back(get_compile_time_arg_val(1), v16);
      int32_t v66 = get_semaphore(get_compile_time_arg_val(5));
      int32_t v67 = get_semaphore(get_compile_time_arg_val(6));
      if (v34) {
        InterleavedAddrGenFast<true> v68;
        v68.bank_base_address = v22;
        v68.page_size = v28;
        v68.data_format = v27;
        InterleavedAddrGenFast<true> v69 = v68;
        int32_t v70 = get_write_ptr(get_compile_time_arg_val(1));
        int32_t v71 = v23 != (int32_t) ((uint32_t) (v23 / v10) * (uint32_t) v10) & v23 < v7 == v1 ? (int32_t) ((uint32_t) (v23 / v10) + (uint32_t) v12) : v23 / v10;
        int32_t v72 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v42 / v10) * (uint32_t) v71)) + (uint32_t) ((int32_t) ((uint32_t) ((i38 % v33) / v40) * (uint32_t) v9) / v10));
        uint64_t temp_366 = v69.get_noc_addr(v72, v7);
        noc_async_read(temp_366, v70, v28);
        int32_t v73 = (int32_t) ((uint32_t) v70 + (uint32_t) v14);
        int32_t v74 = (int32_t) ((uint32_t) v72 + (uint32_t) v12);
        uint64_t temp_378 = v69.get_noc_addr(v74, v7);
        noc_async_read(temp_378, v73, v28);
        int32_t v75 = (int32_t) ((uint32_t) v70 + (uint32_t) v15);
        int32_t v76 = (int32_t) ((uint32_t) v72 + (uint32_t) v11);
        uint64_t temp_390 = v69.get_noc_addr(v76, v7);
        noc_async_read(temp_390, v75, v28);
        int32_t v77 = (int32_t) ((uint32_t) v70 + (uint32_t) 6144);
        int32_t v78 = (int32_t) ((uint32_t) v72 + (uint32_t) 3);
        uint64_t temp_402 = v69.get_noc_addr(v78, v7);
        noc_async_read(temp_402, v77, v28);
        int32_t v79 = (int32_t) ((uint32_t) v70 + (uint32_t) 8192);
        uint64_t temp_414 = v69.get_noc_addr((int32_t) ((uint32_t) v72 + (uint32_t) v71), v7);
        noc_async_read(temp_414, v79, v28);
        int32_t v80 = (int32_t) ((uint32_t) v70 + (uint32_t) 10240);
        uint64_t temp_426 = v69.get_noc_addr((int32_t) ((uint32_t) v74 + (uint32_t) v71), v7);
        noc_async_read(temp_426, v80, v28);
        int32_t v81 = (int32_t) ((uint32_t) v70 + (uint32_t) 12288);
        uint64_t temp_438 = v69.get_noc_addr((int32_t) ((uint32_t) v76 + (uint32_t) v71), v7);
        noc_async_read(temp_438, v81, v28);
        int32_t v82 = (int32_t) ((uint32_t) v70 + (uint32_t) 14336);
        uint64_t temp_450 = v69.get_noc_addr((int32_t) ((uint32_t) v78 + (uint32_t) v71), v7);
        noc_async_read(temp_450, v82, v28);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v83 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v66);
        noc_semaphore_wait(v83, v5);
        noc_semaphore_set(v83, v4);
        size_t v84 = experimental::convert_logical_y_to_translated(v2);
        size_t v85 = experimental::convert_logical_x_to_translated(v37);
        int64_t v86 = get_noc_addr(v85, v84, v70);
        noc_async_write(v70, v86, v17);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v87 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v67);
        noc_semaphore_set(v87, v5);
        size_t v88 = experimental::convert_logical_y_to_translated(v2);
        size_t v89 = experimental::convert_logical_x_to_translated(v37);
        int64_t v90 = get_noc_addr(v89, v88, v67);
        noc_semaphore_inc(v90, v5);
      } else {
        size_t v91 = experimental::convert_logical_y_to_translated(v2);
        size_t v92 = experimental::convert_logical_x_to_translated(v3 - v6);
        int64_t v93 = get_noc_addr(v92, v91, v66);
        noc_semaphore_inc(v93, v5);
        volatile tt_l1_ptr uint32_t* v94 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v67);
        noc_semaphore_wait(v94, v5);
        noc_semaphore_set(v94, v4);
      }
      cb_push_back(get_compile_time_arg_val(1), v16);
    }
  }
  return;
}

