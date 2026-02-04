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
  size_t v2 = get_absolute_logical_x();
  size_t v3 = 0;
  size_t v4 = 1;
  size_t v5 = 5;
  int32_t v6 = 63;
  int32_t v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 1;
  int32_t v10 = 4;
  bool v11 = true;
  int32_t v12 = 32;
  int32_t v13 = 2048;
  int32_t v14 = 4096;
  int32_t v15 = 6144;
  int32_t v16 = 8192;
  int32_t v17 = get_common_arg_val<uint32_t>(v3);
  int32_t v18 = get_common_arg_val<uint32_t>(2);
  int32_t v19 = get_common_arg_val<uint32_t>(10);
  int32_t v20 = get_common_arg_val<uint32_t>(12);
  int32_t v21 = get_common_arg_val<uint32_t>(30);
  int32_t v22 = get_common_arg_val<uint32_t>(31);
  int32_t v23 = get_common_arg_val<uint32_t>(32);
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(1));
  DataFormat v26 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v27 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v28 = get_arg_val<uint32_t>(v4);
  int32_t v29 = get_arg_val<uint32_t>(v3);
  int32_t v30 = (int32_t) ((uint32_t) v22 + (uint32_t) v6) / v8;
  bool v31 = (ptrdiff_t) v2 == (ptrdiff_t) v5;
  size_t v32 = v31 ? 4 : 6;
  size_t v33 = v31 ? v5 : 7;
  int32_t v34 = v20 != (int32_t) ((uint32_t) (v20 / v12) * (uint32_t) v12) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v12) + (uint32_t) v9) : v20 / v12;
  for (int32_t i35 = v29; i35 < v28; i35 += v9) {
    int32_t v36 = i35 / v30;
    int32_t v37 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) v6) / v8) - (uint32_t) v36) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) v6) / v8) - (uint32_t) v36) : v9;
    for (int32_t j38 = v7; j38 < ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8); j38 += v9) {
      int32_t v39 = (int32_t) ((uint32_t) j38 * (uint32_t) v8);
      cb_reserve_back(get_compile_time_arg_val(0), v10);
      int32_t v40 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v41 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v3) {
        InterleavedAddrGenFast<true> v42;
        v42.bank_base_address = v17;
        v42.page_size = v27;
        v42.data_format = v26;
        InterleavedAddrGenFast<true> v43 = v42;
        int32_t v44 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v45 = v18 != (int32_t) ((uint32_t) (v18 / v12) * (uint32_t) v12) & v18 < v7 == v1 ? (int32_t) ((uint32_t) (v18 / v12) + (uint32_t) v9) : v18 / v12;
        int32_t v46 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v36 + (uint32_t) (i35 % v37))) * (uint32_t) v8) / v12) * (uint32_t) v45)) + (uint32_t) (v39 / v12));
        uint64_t temp_345 = v43.get_noc_addr(v46, v7);
        noc_async_read(temp_345, v44, v27);
        int32_t v47 = (int32_t) ((uint32_t) v44 + (uint32_t) v13);
        int32_t v48 = (int32_t) ((uint32_t) v46 + (uint32_t) v9);
        uint64_t temp_357 = v43.get_noc_addr(v48, v7);
        noc_async_read(temp_357, v47, v27);
        int32_t v49 = (int32_t) ((uint32_t) v44 + (uint32_t) v14);
        uint64_t temp_369 = v43.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v45), v7);
        noc_async_read(temp_369, v49, v27);
        int32_t v50 = (int32_t) ((uint32_t) v44 + (uint32_t) v15);
        uint64_t temp_381 = v43.get_noc_addr((int32_t) ((uint32_t) v48 + (uint32_t) v45), v7);
        noc_async_read(temp_381, v50, v27);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v51 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v40);
        size_t v52 = v33 - v4;
        noc_semaphore_wait(v51, v52);
        noc_semaphore_set(v51, v3);
        size_t v53 = experimental::convert_logical_y_to_translated(v32);
        size_t v54 = experimental::convert_logical_x_to_translated(v2);
        size_t v55 = experimental::convert_logical_y_to_translated(v3);
        size_t v56 = experimental::convert_logical_x_to_translated(v2);
        int64_t v57 = experimental::get_noc_multicast_addr(v56, v55, v54, v53, v44);
        int32_t v58 = (int32_t) ((uint32_t) ((int32_t) ((ptrdiff_t) v33)) - (uint32_t) v9);
        noc_async_write_multicast(v44, v57, v16, v58);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v59 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v41);
        noc_semaphore_set(v59, v4);
        size_t v60 = experimental::convert_logical_y_to_translated(v32);
        size_t v61 = experimental::convert_logical_x_to_translated(v2);
        size_t v62 = experimental::convert_logical_y_to_translated(v3);
        size_t v63 = experimental::convert_logical_x_to_translated(v2);
        int64_t v64 = experimental::get_noc_multicast_addr(v63, v62, v61, v60, v41);
        noc_semaphore_set_multicast(v41, v64, v58);
      } else {
        size_t v65 = experimental::convert_logical_y_to_translated(v3);
        size_t v66 = experimental::convert_logical_x_to_translated(v2);
        int64_t v67 = get_noc_addr(v66, v65, v40);
        noc_semaphore_inc(v67, v4);
        volatile tt_l1_ptr uint32_t* v68 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v41);
        noc_semaphore_wait(v68, v4);
        noc_semaphore_set(v68, v3);
      }
      cb_push_back(get_compile_time_arg_val(0), v10);
      cb_reserve_back(get_compile_time_arg_val(1), v10);
      InterleavedAddrGenFast<true> v69;
      v69.bank_base_address = v19;
      v69.page_size = v25;
      v69.data_format = v24;
      InterleavedAddrGenFast<true> v70 = v69;
      int32_t v71 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v72 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v39 / v12) * (uint32_t) v34)) + (uint32_t) ((int32_t) ((uint32_t) ((i35 % v30) / v37) * (uint32_t) v8) / v12));
      uint64_t temp_295 = v70.get_noc_addr(v72, v7);
      noc_async_read(temp_295, v71, v25);
      int32_t v73 = (int32_t) ((uint32_t) v71 + (uint32_t) v14);
      int32_t v74 = (int32_t) ((uint32_t) v72 + (uint32_t) v9);
      uint64_t temp_307 = v70.get_noc_addr(v74, v7);
      noc_async_read(temp_307, v73, v25);
      int32_t v75 = (int32_t) ((uint32_t) v71 + (uint32_t) v13);
      uint64_t temp_319 = v70.get_noc_addr((int32_t) ((uint32_t) v72 + (uint32_t) v34), v7);
      noc_async_read(temp_319, v75, v25);
      int32_t v76 = (int32_t) ((uint32_t) v71 + (uint32_t) v15);
      uint64_t temp_331 = v70.get_noc_addr((int32_t) ((uint32_t) v74 + (uint32_t) v34), v7);
      noc_async_read(temp_331, v76, v25);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v10);
    }
  }
  return;
}
