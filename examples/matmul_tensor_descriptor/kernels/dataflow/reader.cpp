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
  size_t v2 = 0;
  size_t v3 = 1;
  size_t v4 = 4;
  size_t v5 = 7;
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
  size_t v17 = 39;
  int32_t v18 = 39;
  int32_t v19 = get_common_arg_val<uint32_t>(v2);
  int32_t v20 = get_common_arg_val<uint32_t>(2);
  int32_t v21 = get_common_arg_val<uint32_t>(10);
  int32_t v22 = get_common_arg_val<uint32_t>(12);
  int32_t v23 = get_common_arg_val<uint32_t>(30);
  int32_t v24 = get_common_arg_val<uint32_t>(31);
  int32_t v25 = get_common_arg_val<uint32_t>(32);
  DataFormat v26 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v27 = get_tile_size(get_compile_time_arg_val(1));
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v30 = get_arg_val<uint32_t>(v3);
  int32_t v31 = get_arg_val<uint32_t>(v2);
  int32_t v32 = (int32_t) ((uint32_t) v24 + (uint32_t) v6) / v8;
  int32_t v33 = v22 != (int32_t) ((uint32_t) (v22 / v12) * (uint32_t) v12) & v22 < v7 == v1 ? (int32_t) ((uint32_t) (v22 / v12) + (uint32_t) v9) : v22 / v12;
  for (int32_t i34 = v31; i34 < v30; i34 += v9) {
    int32_t v35 = i34 / v32;
    int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v35) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v35) : v9;
    for (int32_t j37 = v7; j37 < ((int32_t) ((uint32_t) v25 + (uint32_t) v6) / v8); j37 += v9) {
      int32_t v38 = (int32_t) ((uint32_t) j37 * (uint32_t) v8);
      cb_reserve_back(get_compile_time_arg_val(0), v10);
      int32_t v39 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v40 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v2 & (ptrdiff_t) get_absolute_logical_x() == (ptrdiff_t) v2) {
        InterleavedAddrGenFast<true> v41;
        v41.bank_base_address = v19;
        v41.page_size = v29;
        v41.data_format = v28;
        InterleavedAddrGenFast<true> v42 = v41;
        int32_t v43 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v44 = v20 != (int32_t) ((uint32_t) (v20 / v12) * (uint32_t) v12) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v12) + (uint32_t) v9) : v20 / v12;
        int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) (i34 % v36))) * (uint32_t) v8) / v12) * (uint32_t) v44)) + (uint32_t) (v38 / v12));
        uint64_t temp_343 = v42.get_noc_addr(v45, v7);
        noc_async_read(temp_343, v43, v29);
        int32_t v46 = (int32_t) ((uint32_t) v43 + (uint32_t) v13);
        int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v9);
        uint64_t temp_355 = v42.get_noc_addr(v47, v7);
        noc_async_read(temp_355, v46, v29);
        int32_t v48 = (int32_t) ((uint32_t) v43 + (uint32_t) v14);
        uint64_t temp_367 = v42.get_noc_addr((int32_t) ((uint32_t) v45 + (uint32_t) v44), v7);
        noc_async_read(temp_367, v48, v29);
        int32_t v49 = (int32_t) ((uint32_t) v43 + (uint32_t) v15);
        uint64_t temp_379 = v42.get_noc_addr((int32_t) ((uint32_t) v47 + (uint32_t) v44), v7);
        noc_async_read(temp_379, v49, v29);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v50 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v39);
        noc_semaphore_wait(v50, v17);
        noc_semaphore_set(v50, v2);
        size_t v51 = experimental::convert_logical_y_to_translated(v4);
        size_t v52 = experimental::convert_logical_x_to_translated(v5);
        size_t v53 = experimental::convert_logical_y_to_translated(v2);
        size_t v54 = experimental::convert_logical_x_to_translated(v2);
        int64_t v55 = experimental::get_noc_multicast_addr(v54, v53, v52, v51, v43);
        noc_async_write_multicast(v43, v55, v16, v18);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v56 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v40);
        noc_semaphore_set(v56, v3);
        size_t v57 = experimental::convert_logical_y_to_translated(v4);
        size_t v58 = experimental::convert_logical_x_to_translated(v5);
        size_t v59 = experimental::convert_logical_y_to_translated(v2);
        size_t v60 = experimental::convert_logical_x_to_translated(v2);
        int64_t v61 = experimental::get_noc_multicast_addr(v60, v59, v58, v57, v40);
        noc_semaphore_set_multicast(v40, v61, v18);
      } else {
        size_t v62 = experimental::convert_logical_y_to_translated(v2);
        size_t v63 = experimental::convert_logical_x_to_translated(v2);
        int64_t v64 = get_noc_addr(v63, v62, v39);
        noc_semaphore_inc(v64, v3);
        volatile tt_l1_ptr uint32_t* v65 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v40);
        noc_semaphore_wait(v65, v3);
        noc_semaphore_set(v65, v2);
      }
      cb_push_back(get_compile_time_arg_val(0), v10);
      cb_reserve_back(get_compile_time_arg_val(1), v10);
      InterleavedAddrGenFast<true> v66;
      v66.bank_base_address = v21;
      v66.page_size = v27;
      v66.data_format = v26;
      InterleavedAddrGenFast<true> v67 = v66;
      int32_t v68 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v69 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v38 / v12) * (uint32_t) v33)) + (uint32_t) ((int32_t) ((uint32_t) ((i34 % v32) / v36) * (uint32_t) v8) / v12));
      uint64_t temp_293 = v67.get_noc_addr(v69, v7);
      noc_async_read(temp_293, v68, v27);
      int32_t v70 = (int32_t) ((uint32_t) v68 + (uint32_t) v13);
      int32_t v71 = (int32_t) ((uint32_t) v69 + (uint32_t) v9);
      uint64_t temp_305 = v67.get_noc_addr(v71, v7);
      noc_async_read(temp_305, v70, v27);
      int32_t v72 = (int32_t) ((uint32_t) v68 + (uint32_t) v14);
      uint64_t temp_317 = v67.get_noc_addr((int32_t) ((uint32_t) v69 + (uint32_t) v33), v7);
      noc_async_read(temp_317, v72, v27);
      int32_t v73 = (int32_t) ((uint32_t) v68 + (uint32_t) v15);
      uint64_t temp_329 = v67.get_noc_addr((int32_t) ((uint32_t) v71 + (uint32_t) v33), v7);
      noc_async_read(temp_329, v73, v27);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v10);
    }
  }
  return;
}
