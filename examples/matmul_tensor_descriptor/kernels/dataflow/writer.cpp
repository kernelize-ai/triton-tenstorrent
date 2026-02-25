// matmul_kernel_tma__writer
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
  int32_t v9 = 64;
  int32_t v10 = 1;
  bool v11 = true;
  int32_t v12 = 65535;
  int32_t v13 = 16;
  int32_t v14 = 4;
  int32_t v15 = 32;
  int32_t v16 = 2048;
  int32_t v17 = 4096;
  int32_t v18 = 6144;
  int32_t v19 = 8192;
  int32_t v20 = get_common_arg_val<uint32_t>(v4);
  int32_t v21 = get_common_arg_val<uint32_t>(v6);
  int32_t v22 = get_common_arg_val<uint32_t>(20);
  int32_t v23 = get_common_arg_val<uint32_t>(22);
  int32_t v24 = get_common_arg_val<uint32_t>(30);
  int32_t v25 = get_common_arg_val<uint32_t>(31);
  int32_t v26 = get_common_arg_val<uint32_t>(32);
  DataFormat v27 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v28 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v29;
  v29.bank_base_address = v22;
  v29.page_size = v28;
  v29.data_format = v27;
  InterleavedAddrGenFast<true> v30 = v29;
  DataFormat v31 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v32 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v33 = get_arg_val<uint32_t>(v5);
  int32_t v34 = get_arg_val<uint32_t>(v4);
  int32_t v35 = (int32_t) ((uint32_t) v25 + (uint32_t) v7) / v9;
  int32_t v36 = get_arg_val<uint32_t>(v6);
  int32_t v37 = get_arg_val<uint32_t>(3);
  int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v13 < v3 ? (uint32_t) v36 >> (uint32_t) v13 : v2)) & (uint32_t) v12);
  int32_t v39 = (int32_t) ((uint32_t) v36 & (uint32_t) v12);
  int32_t v40 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v13 < v3 ? (uint32_t) v37 >> (uint32_t) v13 : v2)) & (uint32_t) v12);
  int32_t v41 = (int32_t) ((uint32_t) v37 & (uint32_t) v12);
  int32_t v42 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v40 + (uint32_t) v10)) * (uint32_t) ((int32_t) ((uint32_t) v41 + (uint32_t) v10)))) - (uint32_t) v10);
  int32_t v43 = v23 != (int32_t) ((uint32_t) (v23 / v15) * (uint32_t) v15) & v23 < v8 == v1 ? (int32_t) ((uint32_t) (v23 / v15) + (uint32_t) v10) : v23 / v15;
  for (int32_t i44 = v34; i44 < v33; i44 += v10) {
    int32_t v45 = i44 / v35;
    int32_t v46 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v24 + (uint32_t) v7) / v9) - (uint32_t) v45) < v10 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v24 + (uint32_t) v7) / v9) - (uint32_t) v45) : v10;
    int32_t v47 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v45 + (uint32_t) (i44 % v46))) * (uint32_t) v9);
    for (int32_t j48 = v8; j48 < ((int32_t) ((uint32_t) v26 + (uint32_t) v7) / v9); j48 += v10) {
      int32_t v49 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v50 = get_semaphore(get_compile_time_arg_val(4));
      if (v38 == (int32_t) ((ptrdiff_t) get_absolute_logical_x()) & v39 == (int32_t) ((ptrdiff_t) get_absolute_logical_y())) {
        cb_reserve_back(get_compile_time_arg_val(0), v14);
        InterleavedAddrGenFast<true> v51;
        v51.bank_base_address = v20;
        v51.page_size = v32;
        v51.data_format = v31;
        InterleavedAddrGenFast<true> v52 = v51;
        int32_t v53 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v54 = v21 != (int32_t) ((uint32_t) (v21 / v15) * (uint32_t) v15) & v21 < v8 == v1 ? (int32_t) ((uint32_t) (v21 / v15) + (uint32_t) v10) : v21 / v15;
        int32_t v55 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v47 / v15) * (uint32_t) v54)) + (uint32_t) ((int32_t) ((uint32_t) j48 * (uint32_t) v9) / v15));
        uint64_t temp_321 = v52.get_noc_addr(v55, v8);
        noc_async_read(temp_321, v53, v32);
        int32_t v56 = (int32_t) ((uint32_t) v53 + (uint32_t) v16);
        int32_t v57 = (int32_t) ((uint32_t) v55 + (uint32_t) v10);
        uint64_t temp_333 = v52.get_noc_addr(v57, v8);
        noc_async_read(temp_333, v56, v32);
        int32_t v58 = (int32_t) ((uint32_t) v53 + (uint32_t) v17);
        uint64_t temp_345 = v52.get_noc_addr((int32_t) ((uint32_t) v55 + (uint32_t) v54), v8);
        noc_async_read(temp_345, v58, v32);
        int32_t v59 = (int32_t) ((uint32_t) v53 + (uint32_t) v18);
        uint64_t temp_357 = v52.get_noc_addr((int32_t) ((uint32_t) v57 + (uint32_t) v54), v8);
        noc_async_read(temp_357, v59, v32);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        int32_t v60 = get_write_ptr(get_compile_time_arg_val(0));
        volatile tt_l1_ptr uint32_t* v61 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v49);
        noc_semaphore_wait(v61, v42);
        noc_semaphore_set(v61, v4);
        size_t v62 = experimental::convert_logical_y_to_translated(v41);
        size_t v63 = experimental::convert_logical_x_to_translated(v40);
        size_t v64 = experimental::convert_logical_y_to_translated(v39);
        size_t v65 = experimental::convert_logical_x_to_translated(v38);
        int64_t v66 = experimental::get_noc_multicast_addr(v65, v64, v63, v62, v60);
        noc_async_write_multicast(v60, v66, v19, v42);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v67 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v50);
        noc_semaphore_set(v67, v5);
        size_t v68 = experimental::convert_logical_y_to_translated(v41);
        size_t v69 = experimental::convert_logical_x_to_translated(v40);
        size_t v70 = experimental::convert_logical_y_to_translated(v39);
        size_t v71 = experimental::convert_logical_x_to_translated(v38);
        int64_t v72 = experimental::get_noc_multicast_addr(v71, v70, v69, v68, v50);
        noc_semaphore_set_multicast(v50, v72, v42);
      } else {
        cb_reserve_back(get_compile_time_arg_val(0), v14);
        size_t v73 = experimental::convert_logical_y_to_translated(v39);
        size_t v74 = experimental::convert_logical_x_to_translated(v38);
        int64_t v75 = get_noc_addr(v74, v73, v49);
        noc_semaphore_inc(v75, v5);
        volatile tt_l1_ptr uint32_t* v76 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v50);
        noc_semaphore_wait(v76, v5);
        noc_semaphore_set(v76, v4);
      }
      cb_push_back(get_compile_time_arg_val(0), v14);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v14);
    }
    int32_t v77 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v78 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v47 / v15) * (uint32_t) v43)) + (uint32_t) ((int32_t) ((uint32_t) ((i44 % v35) / v46) * (uint32_t) v9) / v15));
    uint64_t temp_258 = v30.get_noc_addr(v78, v8);
    noc_async_write(v77, temp_258, v28);
    int32_t v79 = (int32_t) ((uint32_t) v77 + (uint32_t) v16);
    int32_t v80 = (int32_t) ((uint32_t) v78 + (uint32_t) v10);
    uint64_t temp_270 = v30.get_noc_addr(v80, v8);
    noc_async_write(v79, temp_270, v28);
    int32_t v81 = (int32_t) ((uint32_t) v77 + (uint32_t) v17);
    uint64_t temp_282 = v30.get_noc_addr((int32_t) ((uint32_t) v78 + (uint32_t) v43), v8);
    noc_async_write(v81, temp_282, v28);
    int32_t v82 = (int32_t) ((uint32_t) v77 + (uint32_t) v18);
    uint64_t temp_294 = v30.get_noc_addr((int32_t) ((uint32_t) v80 + (uint32_t) v43), v8);
    noc_async_write(v82, temp_294, v28);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v14);
  }
  return;
}
