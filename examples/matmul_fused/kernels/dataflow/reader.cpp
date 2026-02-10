// matmul_kernel_fused__reader
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
  int32_t v19 = get_common_arg_val<uint32_t>(10);
  int32_t v20 = get_common_arg_val<uint32_t>(12);
  int32_t v21 = get_common_arg_val<uint32_t>(30);
  int32_t v22 = get_common_arg_val<uint32_t>(32);
  int32_t v23 = get_common_arg_val<uint32_t>(40);
  int32_t v24 = get_common_arg_val<uint32_t>(41);
  int32_t v25 = get_common_arg_val<uint32_t>(42);
  DataFormat v26 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v27 = get_tile_size(get_compile_time_arg_val(2));
  DataFormat v28 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v29 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v30 = get_arg_val<uint32_t>(v3);
  int32_t v31 = get_arg_val<uint32_t>(v2);
  int32_t v32 = (int32_t) ((uint32_t) v24 + (uint32_t) v6) / v8;
  int32_t v33 = v20 != (int32_t) ((uint32_t) (v20 / v12) * (uint32_t) v12) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v12) + (uint32_t) v9) : v20 / v12;
  for (int32_t i34 = v31; i34 < v30; i34 += v9) {
    int32_t v35 = i34 / v32;
    int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v35) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v35) : v9;
    int32_t v37 = (int32_t) ((uint32_t) ((i34 % v32) / v36) * (uint32_t) v8) / v12;
    for (int32_t j38 = v7; j38 < ((int32_t) ((uint32_t) v25 + (uint32_t) v6) / v8); j38 += v9) {
      cb_reserve_back(get_compile_time_arg_val(1), v10);
      InterleavedAddrGenFast<true> v39;
      v39.bank_base_address = v19;
      v39.page_size = v29;
      v39.data_format = v28;
      InterleavedAddrGenFast<true> v40 = v39;
      int32_t v41 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v42 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j38 * (uint32_t) v8) / v12) * (uint32_t) v33)) + (uint32_t) v37);
      uint64_t temp_408 = v40.get_noc_addr(v42, v7);
      noc_async_read(temp_408, v41, v29);
      int32_t v43 = (int32_t) ((uint32_t) v41 + (uint32_t) v13);
      int32_t v44 = (int32_t) ((uint32_t) v42 + (uint32_t) v9);
      uint64_t temp_420 = v40.get_noc_addr(v44, v7);
      noc_async_read(temp_420, v43, v29);
      int32_t v45 = (int32_t) ((uint32_t) v41 + (uint32_t) v14);
      uint64_t temp_432 = v40.get_noc_addr((int32_t) ((uint32_t) v42 + (uint32_t) v33), v7);
      noc_async_read(temp_432, v45, v29);
      int32_t v46 = (int32_t) ((uint32_t) v41 + (uint32_t) v15);
      uint64_t temp_444 = v40.get_noc_addr((int32_t) ((uint32_t) v44 + (uint32_t) v33), v7);
      noc_async_read(temp_444, v46, v29);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v10);
    }
    cb_reserve_back(get_compile_time_arg_val(2), v10);
    int32_t v47 = get_semaphore(get_compile_time_arg_val(4));
    int32_t v48 = get_semaphore(get_compile_time_arg_val(5));
    if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v2 & (ptrdiff_t) get_absolute_logical_x() == (ptrdiff_t) v2) {
      InterleavedAddrGenFast<true> v49;
      v49.bank_base_address = v21;
      v49.page_size = v27;
      v49.data_format = v26;
      InterleavedAddrGenFast<true> v50 = v49;
      int32_t v51 = get_write_ptr(get_compile_time_arg_val(2));
      int32_t v52 = v22 != (int32_t) ((uint32_t) (v22 / v12) * (uint32_t) v12) & v22 < v7 == v1 ? (int32_t) ((uint32_t) (v22 / v12) + (uint32_t) v9) : v22 / v12;
      int32_t v53 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) (i34 % v36))) * (uint32_t) v8) / v12) * (uint32_t) v52)) + (uint32_t) v37);
      uint64_t temp_396 = v50.get_noc_addr(v53, v7);
      noc_async_read(temp_396, v51, v27);
      int32_t v54 = (int32_t) ((uint32_t) v51 + (uint32_t) v13);
      int32_t v55 = (int32_t) ((uint32_t) v53 + (uint32_t) v9);
      uint64_t temp_408 = v50.get_noc_addr(v55, v7);
      noc_async_read(temp_408, v54, v27);
      int32_t v56 = (int32_t) ((uint32_t) v51 + (uint32_t) v14);
      uint64_t temp_420 = v50.get_noc_addr((int32_t) ((uint32_t) v53 + (uint32_t) v52), v7);
      noc_async_read(temp_420, v56, v27);
      int32_t v57 = (int32_t) ((uint32_t) v51 + (uint32_t) v15);
      uint64_t temp_432 = v50.get_noc_addr((int32_t) ((uint32_t) v55 + (uint32_t) v52), v7);
      noc_async_read(temp_432, v57, v27);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      volatile tt_l1_ptr uint32_t* v58 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v47);
      noc_semaphore_wait(v58, v17);
      noc_semaphore_set(v58, v2);
      size_t v59 = experimental::convert_logical_y_to_translated(v4);
      size_t v60 = experimental::convert_logical_x_to_translated(v5);
      size_t v61 = experimental::convert_logical_y_to_translated(v2);
      size_t v62 = experimental::convert_logical_x_to_translated(v2);
      int64_t v63 = experimental::get_noc_multicast_addr(v62, v61, v60, v59, v51);
      noc_async_write_multicast(v51, v63, v16, v18);
      {
      DeviceZoneScopedN("noc_async_write_barrier");
      noc_async_write_barrier();
      }
      volatile tt_l1_ptr uint32_t* v64 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v48);
      noc_semaphore_set(v64, v3);
      size_t v65 = experimental::convert_logical_y_to_translated(v4);
      size_t v66 = experimental::convert_logical_x_to_translated(v5);
      size_t v67 = experimental::convert_logical_y_to_translated(v2);
      size_t v68 = experimental::convert_logical_x_to_translated(v2);
      int64_t v69 = experimental::get_noc_multicast_addr(v68, v67, v66, v65, v48);
      noc_semaphore_set_multicast(v48, v69, v18);
    } else {
      size_t v70 = experimental::convert_logical_y_to_translated(v2);
      size_t v71 = experimental::convert_logical_x_to_translated(v2);
      int64_t v72 = get_noc_addr(v71, v70, v47);
      noc_semaphore_inc(v72, v3);
      volatile tt_l1_ptr uint32_t* v73 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v48);
      noc_semaphore_wait(v73, v3);
      noc_semaphore_set(v73, v2);
    }
    cb_push_back(get_compile_time_arg_val(2), v10);
  }
  return;
}

