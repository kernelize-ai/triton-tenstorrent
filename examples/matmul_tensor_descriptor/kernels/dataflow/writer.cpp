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
  size_t v2 = get_absolute_logical_x();
  size_t v3 = 0;
  size_t v4 = 1;
  size_t v5 = 4;
  int32_t v6 = 63;
  int32_t v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 1;
  bool v10 = true;
  int32_t v11 = 4;
  int32_t v12 = 32;
  int32_t v13 = 2048;
  int32_t v14 = 4096;
  int32_t v15 = 6144;
  int32_t v16 = 8192;
  int32_t v17 = get_common_arg_val<uint32_t>(v3);
  int32_t v18 = get_common_arg_val<uint32_t>(2);
  int32_t v19 = get_common_arg_val<uint32_t>(20);
  int32_t v20 = get_common_arg_val<uint32_t>(22);
  int32_t v21 = get_common_arg_val<uint32_t>(30);
  int32_t v22 = get_common_arg_val<uint32_t>(31);
  int32_t v23 = get_common_arg_val<uint32_t>(32);
  DataFormat v24 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v25 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v26;
  v26.bank_base_address = v19;
  v26.page_size = v25;
  v26.data_format = v24;
  InterleavedAddrGenFast<true> v27 = v26;
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v30 = get_arg_val<uint32_t>(v4);
  int32_t v31 = get_arg_val<uint32_t>(v3);
  int32_t v32 = (int32_t) ((uint32_t) v22 + (uint32_t) v6) / v8;
  int32_t v33 = v20 != (int32_t) ((uint32_t) (v20 / v12) * (uint32_t) v12) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v12) + (uint32_t) v9) : v20 / v12;
  for (int32_t i34 = v31; i34 < v30; i34 += v9) {
    int32_t v35 = i34 / v32;
    int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) v6) / v8) - (uint32_t) v35) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v21 + (uint32_t) v6) / v8) - (uint32_t) v35) : v9;
    int32_t v37 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) (i34 % v36))) * (uint32_t) v8);
    for (int32_t j38 = v7; j38 < ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8); j38 += v9) {
      int32_t v39 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v40 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v3) {
        cb_reserve_back(get_compile_time_arg_val(0), v11);
        InterleavedAddrGenFast<true> v41;
        v41.bank_base_address = v17;
        v41.page_size = v29;
        v41.data_format = v28;
        InterleavedAddrGenFast<true> v42 = v41;
        int32_t v43 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v44 = v18 != (int32_t) ((uint32_t) (v18 / v12) * (uint32_t) v12) & v18 < v7 == v1 ? (int32_t) ((uint32_t) (v18 / v12) + (uint32_t) v9) : v18 / v12;
        int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v37 / v12) * (uint32_t) v44)) + (uint32_t) ((int32_t) ((uint32_t) j38 * (uint32_t) v8) / v12));
        uint64_t temp_249 = v42.get_noc_addr(v45, v7);
        noc_async_read(temp_249, v43, v29);
        int32_t v46 = (int32_t) ((uint32_t) v43 + (uint32_t) v13);
        int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v9);
        uint64_t temp_261 = v42.get_noc_addr(v47, v7);
        noc_async_read(temp_261, v46, v29);
        int32_t v48 = (int32_t) ((uint32_t) v43 + (uint32_t) v14);
        uint64_t temp_273 = v42.get_noc_addr((int32_t) ((uint32_t) v45 + (uint32_t) v44), v7);
        noc_async_read(temp_273, v48, v29);
        int32_t v49 = (int32_t) ((uint32_t) v43 + (uint32_t) v15);
        uint64_t temp_285 = v42.get_noc_addr((int32_t) ((uint32_t) v47 + (uint32_t) v44), v7);
        noc_async_read(temp_285, v49, v29);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        int32_t v50 = get_write_ptr(get_compile_time_arg_val(0));
        volatile tt_l1_ptr uint32_t* v51 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v39);
        noc_semaphore_wait(v51, v5);
        noc_semaphore_set(v51, v3);
        size_t v52 = experimental::convert_logical_y_to_translated(v5);
        size_t v53 = experimental::convert_logical_x_to_translated(v2);
        size_t v54 = experimental::convert_logical_y_to_translated(v3);
        size_t v55 = experimental::convert_logical_x_to_translated(v2);
        int64_t v56 = experimental::get_noc_multicast_addr(v55, v54, v53, v52, v50);
        noc_async_write_multicast(v50, v56, v16, v11);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v57 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v40);
        noc_semaphore_set(v57, v4);
        size_t v58 = experimental::convert_logical_y_to_translated(v5);
        size_t v59 = experimental::convert_logical_x_to_translated(v2);
        size_t v60 = experimental::convert_logical_y_to_translated(v3);
        size_t v61 = experimental::convert_logical_x_to_translated(v2);
        int64_t v62 = experimental::get_noc_multicast_addr(v61, v60, v59, v58, v40);
        noc_semaphore_set_multicast(v40, v62, v11);
      } else {
        cb_reserve_back(get_compile_time_arg_val(0), v11);
        size_t v63 = experimental::convert_logical_y_to_translated(v3);
        size_t v64 = experimental::convert_logical_x_to_translated(v2);
        int64_t v65 = get_noc_addr(v64, v63, v39);
        noc_semaphore_inc(v65, v4);
        volatile tt_l1_ptr uint32_t* v66 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v40);
        noc_semaphore_wait(v66, v4);
        noc_semaphore_set(v66, v3);
      }
      cb_push_back(get_compile_time_arg_val(0), v11);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v11);
    }
    int32_t v67 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v68 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v37 / v12) * (uint32_t) v33)) + (uint32_t) ((int32_t) ((uint32_t) ((i34 % v32) / v36) * (uint32_t) v8) / v12));
    uint64_t temp_186 = v27.get_noc_addr(v68, v7);
    noc_async_write(v67, temp_186, v25);
    int32_t v69 = (int32_t) ((uint32_t) v67 + (uint32_t) v13);
    int32_t v70 = (int32_t) ((uint32_t) v68 + (uint32_t) v9);
    uint64_t temp_198 = v27.get_noc_addr(v70, v7);
    noc_async_write(v69, temp_198, v25);
    int32_t v71 = (int32_t) ((uint32_t) v67 + (uint32_t) v14);
    uint64_t temp_210 = v27.get_noc_addr((int32_t) ((uint32_t) v68 + (uint32_t) v33), v7);
    noc_async_write(v71, temp_210, v25);
    int32_t v72 = (int32_t) ((uint32_t) v67 + (uint32_t) v15);
    uint64_t temp_222 = v27.get_noc_addr((int32_t) ((uint32_t) v70 + (uint32_t) v33), v7);
    noc_async_write(v72, temp_222, v25);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v11);
  }
  return;
}
