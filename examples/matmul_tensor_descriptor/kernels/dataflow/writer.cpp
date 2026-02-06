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
  size_t v2 = 0;
  size_t v3 = 1;
  size_t v4 = 4;
  size_t v5 = 7;
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
  size_t v17 = 39;
  int32_t v18 = 39;
  int32_t v19 = get_common_arg_val<uint32_t>(v2);
  int32_t v20 = get_common_arg_val<uint32_t>(2);
  int32_t v21 = get_common_arg_val<uint32_t>(20);
  int32_t v22 = get_common_arg_val<uint32_t>(22);
  int32_t v23 = get_common_arg_val<uint32_t>(30);
  int32_t v24 = get_common_arg_val<uint32_t>(31);
  int32_t v25 = get_common_arg_val<uint32_t>(32);
  DataFormat v26 = get_dataformat(get_compile_time_arg_val(2));
  int32_t v27 = get_tile_size(get_compile_time_arg_val(2));
  InterleavedAddrGenFast<true> v28;
  v28.bank_base_address = v21;
  v28.page_size = v27;
  v28.data_format = v26;
  InterleavedAddrGenFast<true> v29 = v28;
  DataFormat v30 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v31 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v32 = get_arg_val<uint32_t>(v3);
  int32_t v33 = get_arg_val<uint32_t>(v2);
  int32_t v34 = (int32_t) ((uint32_t) v24 + (uint32_t) v6) / v8;
  int32_t v35 = v22 != (int32_t) ((uint32_t) (v22 / v12) * (uint32_t) v12) & v22 < v7 == v1 ? (int32_t) ((uint32_t) (v22 / v12) + (uint32_t) v9) : v22 / v12;
  for (int32_t i36 = v33; i36 < v32; i36 += v9) {
    int32_t v37 = i36 / v34;
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v37) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v37) : v9;
    int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v37 + (uint32_t) (i36 % v38))) * (uint32_t) v8);
    for (int32_t j40 = v7; j40 < ((int32_t) ((uint32_t) v25 + (uint32_t) v6) / v8); j40 += v9) {
      cb_reserve_back(get_compile_time_arg_val(0), v11);
      int32_t v41 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v42 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v2 & (ptrdiff_t) get_absolute_logical_x() == (ptrdiff_t) v2) {
        InterleavedAddrGenFast<true> v43;
        v43.bank_base_address = v19;
        v43.page_size = v31;
        v43.data_format = v30;
        InterleavedAddrGenFast<true> v44 = v43;
        int32_t v45 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v46 = v20 != (int32_t) ((uint32_t) (v20 / v12) * (uint32_t) v12) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v12) + (uint32_t) v9) : v20 / v12;
        int32_t v47 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v39 / v12) * (uint32_t) v46)) + (uint32_t) ((int32_t) ((uint32_t) j40 * (uint32_t) v8) / v12));
        uint64_t temp_261 = v44.get_noc_addr(v47, v7);
        noc_async_read(temp_261, v45, v31);
        int32_t v48 = (int32_t) ((uint32_t) v45 + (uint32_t) v13);
        int32_t v49 = (int32_t) ((uint32_t) v47 + (uint32_t) v9);
        uint64_t temp_273 = v44.get_noc_addr(v49, v7);
        noc_async_read(temp_273, v48, v31);
        int32_t v50 = (int32_t) ((uint32_t) v45 + (uint32_t) v14);
        uint64_t temp_285 = v44.get_noc_addr((int32_t) ((uint32_t) v47 + (uint32_t) v46), v7);
        noc_async_read(temp_285, v50, v31);
        int32_t v51 = (int32_t) ((uint32_t) v45 + (uint32_t) v15);
        uint64_t temp_297 = v44.get_noc_addr((int32_t) ((uint32_t) v49 + (uint32_t) v46), v7);
        noc_async_read(temp_297, v51, v31);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v52 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v41);
        noc_semaphore_wait(v52, v17);
        noc_semaphore_set(v52, v2);
        size_t v53 = experimental::convert_logical_y_to_translated(v4);
        size_t v54 = experimental::convert_logical_x_to_translated(v5);
        size_t v55 = experimental::convert_logical_y_to_translated(v2);
        size_t v56 = experimental::convert_logical_x_to_translated(v2);
        int64_t v57 = experimental::get_noc_multicast_addr(v56, v55, v54, v53, v45);
        noc_async_write_multicast(v45, v57, v16, v18);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v58 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v42);
        noc_semaphore_set(v58, v3);
        size_t v59 = experimental::convert_logical_y_to_translated(v4);
        size_t v60 = experimental::convert_logical_x_to_translated(v5);
        size_t v61 = experimental::convert_logical_y_to_translated(v2);
        size_t v62 = experimental::convert_logical_x_to_translated(v2);
        int64_t v63 = experimental::get_noc_multicast_addr(v62, v61, v60, v59, v42);
        noc_semaphore_set_multicast(v42, v63, v18);
      } else {
        size_t v64 = experimental::convert_logical_y_to_translated(v2);
        size_t v65 = experimental::convert_logical_x_to_translated(v2);
        int64_t v66 = get_noc_addr(v65, v64, v41);
        noc_semaphore_inc(v66, v3);
        volatile tt_l1_ptr uint32_t* v67 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v42);
        noc_semaphore_wait(v67, v3);
        noc_semaphore_set(v67, v2);
      }
      cb_push_back(get_compile_time_arg_val(0), v11);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v11);
    }
    int32_t v68 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v69 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v39 / v12) * (uint32_t) v35)) + (uint32_t) ((int32_t) ((uint32_t) ((i36 % v34) / v38) * (uint32_t) v8) / v12));
    uint64_t temp_198 = v29.get_noc_addr(v69, v7);
    noc_async_write(v68, temp_198, v27);
    int32_t v70 = (int32_t) ((uint32_t) v68 + (uint32_t) v13);
    int32_t v71 = (int32_t) ((uint32_t) v69 + (uint32_t) v9);
    uint64_t temp_210 = v29.get_noc_addr(v71, v7);
    noc_async_write(v70, temp_210, v27);
    int32_t v72 = (int32_t) ((uint32_t) v68 + (uint32_t) v14);
    uint64_t temp_222 = v29.get_noc_addr((int32_t) ((uint32_t) v69 + (uint32_t) v35), v7);
    noc_async_write(v72, temp_222, v27);
    int32_t v73 = (int32_t) ((uint32_t) v68 + (uint32_t) v15);
    uint64_t temp_234 = v29.get_noc_addr((int32_t) ((uint32_t) v71 + (uint32_t) v35), v7);
    noc_async_write(v73, temp_234, v27);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v11);
  }
  return;
}
