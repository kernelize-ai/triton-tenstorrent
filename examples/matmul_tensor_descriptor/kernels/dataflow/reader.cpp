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
  size_t v1 = 0;
  size_t v2 = 1;
  size_t v3 = 4;
  size_t v4 = 7;
  int32_t v5 = 63;
  int32_t v6 = 0;
  int32_t v7 = 64;
  int32_t v8 = 1;
  int32_t v9 = 4;
  bool v10 = true;
  int32_t v11 = 32;
  int32_t v12 = 8192;
  size_t v13 = 39;
  int32_t v14 = 39;
  int32_t v15 = get_common_arg_val<uint32_t>(v1);
  int32_t v16 = get_common_arg_val<uint32_t>(2);
  int32_t v17 = get_common_arg_val<uint32_t>(30);
  int32_t v18 = get_common_arg_val<uint32_t>(31);
  int32_t v19 = get_common_arg_val<uint32_t>(32);
  DataFormat v20 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v22 = get_arg_val<uint32_t>(v2);
  int32_t v23 = get_arg_val<uint32_t>(v1);
  for (int32_t i24 = v23; i24 < v22; i24 += v8) {
    for (int32_t j25 = v6; j25 < ((int32_t) ((uint32_t) v19 + (uint32_t) v5) / v7); j25 += v8) {
      cb_reserve_back(get_compile_time_arg_val(0), v9);
      int32_t v26 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v27 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v1 & (ptrdiff_t) get_absolute_logical_x() == (ptrdiff_t) v1) {
        InterleavedAddrGenFast<true> v28;
        v28.bank_base_address = v15;
        v28.page_size = v21;
        v28.data_format = v20;
        InterleavedAddrGenFast<true> v29 = v28;
        int32_t v30 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v31 = v16 != (int32_t) ((uint32_t) (v16 / v11) * (uint32_t) v11) & v16 < v6 == false ? (int32_t) ((uint32_t) (v16 / v11) + (uint32_t) v8) : v16 / v11;
        int32_t v32 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) (i24 / ((int32_t) ((uint32_t) v18 + (uint32_t) v5) / v7)) + (uint32_t) (i24 % ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v5) / v7) - (uint32_t) (i24 / ((int32_t) ((uint32_t) v18 + (uint32_t) v5) / v7))) < v8 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v17 + (uint32_t) v5) / v7) - (uint32_t) (i24 / ((int32_t) ((uint32_t) v18 + (uint32_t) v5) / v7))) : v8)))) * (uint32_t) v7) / v11) * (uint32_t) v31)) + (uint32_t) ((int32_t) ((uint32_t) j25 * (uint32_t) v7) / v11));
        uint64_t temp_311 = v29.get_noc_addr(v32, v6);
        noc_async_read(temp_311, v30, v21);
        int32_t v33 = (int32_t) ((uint32_t) v30 + (uint32_t) 2048);
        int32_t v34 = (int32_t) ((uint32_t) v32 + (uint32_t) v8);
        uint64_t temp_323 = v29.get_noc_addr(v34, v6);
        noc_async_read(temp_323, v33, v21);
        int32_t v35 = (int32_t) ((uint32_t) v30 + (uint32_t) 4096);
        uint64_t temp_335 = v29.get_noc_addr((int32_t) ((uint32_t) v32 + (uint32_t) v31), v6);
        noc_async_read(temp_335, v35, v21);
        int32_t v36 = (int32_t) ((uint32_t) v30 + (uint32_t) 6144);
        uint64_t temp_347 = v29.get_noc_addr((int32_t) ((uint32_t) v34 + (uint32_t) v31), v6);
        noc_async_read(temp_347, v36, v21);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v37 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v26);
        noc_semaphore_wait(v37, v13);
        noc_semaphore_set(v37, v1);
        size_t v38 = experimental::convert_logical_y_to_translated(v3);
        size_t v39 = experimental::convert_logical_x_to_translated(v4);
        size_t v40 = experimental::convert_logical_y_to_translated(v1);
        size_t v41 = experimental::convert_logical_x_to_translated(v1);
        int64_t v42 = experimental::get_noc_multicast_addr(v41, v40, v39, v38, v30);
        noc_async_write_multicast(v30, v42, v12, v14);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v43 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v27);
        noc_semaphore_set(v43, v2);
        size_t v44 = experimental::convert_logical_y_to_translated(v3);
        size_t v45 = experimental::convert_logical_x_to_translated(v4);
        size_t v46 = experimental::convert_logical_y_to_translated(v1);
        size_t v47 = experimental::convert_logical_x_to_translated(v1);
        int64_t v48 = experimental::get_noc_multicast_addr(v47, v46, v45, v44, v27);
        noc_semaphore_set_multicast(v27, v48, v14);
      } else {
        size_t v49 = experimental::convert_logical_y_to_translated(v1);
        size_t v50 = experimental::convert_logical_x_to_translated(v1);
        int64_t v51 = get_noc_addr(v50, v49, v26);
        noc_semaphore_inc(v51, v2);
        volatile tt_l1_ptr uint32_t* v52 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v27);
        noc_semaphore_wait(v52, v2);
        noc_semaphore_set(v52, v1);
      }
      cb_push_back(get_compile_time_arg_val(0), v9);
    }
  }
  return;
}
