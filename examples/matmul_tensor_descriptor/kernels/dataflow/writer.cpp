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
  int32_t v6 = 63;
  int32_t v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 1;
  bool v10 = true;
  int32_t v11 = 65535;
  int32_t v12 = 16;
  int32_t v13 = 4;
  int32_t v14 = 32;
  int32_t v15 = 2048;
  int32_t v16 = 4096;
  int32_t v17 = 6144;
  int32_t v18 = 8192;
  int32_t v19 = get_common_arg_val<uint32_t>(v4);
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
  int32_t v32 = get_arg_val<uint32_t>(v5);
  int32_t v33 = get_arg_val<uint32_t>(v4);
  int32_t v34 = (int32_t) ((uint32_t) v24 + (uint32_t) v6) / v8;
  int32_t v35 = get_arg_val<uint32_t>(3);
  int32_t v36 = get_arg_val<uint32_t>(4);
  int32_t v37 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 < v3 ? (uint32_t) v35 >> (uint32_t) v12 : v2)) & (uint32_t) v11);
  int32_t v38 = (int32_t) ((uint32_t) v35 & (uint32_t) v11);
  int32_t v39 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 < v3 ? (uint32_t) v36 >> (uint32_t) v12 : v2)) & (uint32_t) v11);
  int32_t v40 = (int32_t) ((uint32_t) v36 & (uint32_t) v11);
  int32_t v41 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v39 + (uint32_t) v9)) * (uint32_t) ((int32_t) ((uint32_t) v40 + (uint32_t) v9)))) - (uint32_t) v9);
  int32_t v42 = v22 != (int32_t) ((uint32_t) (v22 / v14) * (uint32_t) v14) & v22 < v7 == v1 ? (int32_t) ((uint32_t) (v22 / v14) + (uint32_t) v9) : v22 / v14;
  for (int32_t i43 = v33; i43 < v32; i43 += v9) {
    int32_t v44 = i43 / v34;
    int32_t v45 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v44) < v9 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v6) / v8) - (uint32_t) v44) : v9;
    int32_t v46 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v44 + (uint32_t) (i43 % v45))) * (uint32_t) v8);
    for (int32_t j47 = v7; j47 < ((int32_t) ((uint32_t) v25 + (uint32_t) v6) / v8); j47 += v9) {
      int32_t v48 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v49 = get_semaphore(get_compile_time_arg_val(4));
      if (v37 == (int32_t) ((ptrdiff_t) get_absolute_logical_x()) & v38 == (int32_t) ((ptrdiff_t) get_absolute_logical_y())) {
        cb_reserve_back(get_compile_time_arg_val(0), v13);
        InterleavedAddrGenFast<true> v50;
        v50.bank_base_address = v19;
        v50.page_size = v31;
        v50.data_format = v30;
        InterleavedAddrGenFast<true> v51 = v50;
        int32_t v52 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v53 = v20 != (int32_t) ((uint32_t) (v20 / v14) * (uint32_t) v14) & v20 < v7 == v1 ? (int32_t) ((uint32_t) (v20 / v14) + (uint32_t) v9) : v20 / v14;
        int32_t v54 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v46 / v14) * (uint32_t) v53)) + (uint32_t) ((int32_t) ((uint32_t) j47 * (uint32_t) v8) / v14));
        uint64_t temp_323 = v51.get_noc_addr(v54, v7);
        noc_async_read(temp_323, v52, v31);
        int32_t v55 = (int32_t) ((uint32_t) v52 + (uint32_t) v15);
        int32_t v56 = (int32_t) ((uint32_t) v54 + (uint32_t) v9);
        uint64_t temp_335 = v51.get_noc_addr(v56, v7);
        noc_async_read(temp_335, v55, v31);
        int32_t v57 = (int32_t) ((uint32_t) v52 + (uint32_t) v16);
        uint64_t temp_347 = v51.get_noc_addr((int32_t) ((uint32_t) v54 + (uint32_t) v53), v7);
        noc_async_read(temp_347, v57, v31);
        int32_t v58 = (int32_t) ((uint32_t) v52 + (uint32_t) v17);
        uint64_t temp_359 = v51.get_noc_addr((int32_t) ((uint32_t) v56 + (uint32_t) v53), v7);
        noc_async_read(temp_359, v58, v31);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        int32_t v59 = get_write_ptr(get_compile_time_arg_val(0));
        volatile tt_l1_ptr uint32_t* v60 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v48);
        noc_semaphore_wait(v60, v41);
        noc_semaphore_set(v60, v4);
        size_t v61 = experimental::convert_logical_y_to_translated(v40);
        size_t v62 = experimental::convert_logical_x_to_translated(v39);
        size_t v63 = experimental::convert_logical_y_to_translated(v38);
        size_t v64 = experimental::convert_logical_x_to_translated(v37);
        int64_t v65 = experimental::get_noc_multicast_addr(v64, v63, v62, v61, v59);
        noc_async_write_multicast(v59, v65, v18, v41);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v66 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v49);
        noc_semaphore_set(v66, v5);
        size_t v67 = experimental::convert_logical_y_to_translated(v40);
        size_t v68 = experimental::convert_logical_x_to_translated(v39);
        size_t v69 = experimental::convert_logical_y_to_translated(v38);
        size_t v70 = experimental::convert_logical_x_to_translated(v37);
        int64_t v71 = experimental::get_noc_multicast_addr(v70, v69, v68, v67, v49);
        noc_semaphore_set_multicast(v49, v71, v41);
      } else {
        cb_reserve_back(get_compile_time_arg_val(0), v13);
        size_t v72 = experimental::convert_logical_y_to_translated(v38);
        size_t v73 = experimental::convert_logical_x_to_translated(v37);
        int64_t v74 = get_noc_addr(v73, v72, v48);
        noc_semaphore_inc(v74, v5);
        volatile tt_l1_ptr uint32_t* v75 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v49);
        noc_semaphore_wait(v75, v5);
        noc_semaphore_set(v75, v4);
      }
      cb_push_back(get_compile_time_arg_val(0), v13);
    }
    {
    DeviceZoneScopedN("cb_wait_front");
    cb_wait_front(get_compile_time_arg_val(2), v13);
    }
    int32_t v76 = get_read_ptr(get_compile_time_arg_val(2));
    int32_t v77 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v46 / v14) * (uint32_t) v42)) + (uint32_t) ((int32_t) ((uint32_t) ((i43 % v34) / v45) * (uint32_t) v8) / v14));
    uint64_t temp_260 = v29.get_noc_addr(v77, v7);
    noc_async_write(v76, temp_260, v27);
    int32_t v78 = (int32_t) ((uint32_t) v76 + (uint32_t) v15);
    int32_t v79 = (int32_t) ((uint32_t) v77 + (uint32_t) v9);
    uint64_t temp_272 = v29.get_noc_addr(v79, v7);
    noc_async_write(v78, temp_272, v27);
    int32_t v80 = (int32_t) ((uint32_t) v76 + (uint32_t) v16);
    uint64_t temp_284 = v29.get_noc_addr((int32_t) ((uint32_t) v77 + (uint32_t) v42), v7);
    noc_async_write(v80, temp_284, v27);
    int32_t v81 = (int32_t) ((uint32_t) v76 + (uint32_t) v17);
    uint64_t temp_296 = v29.get_noc_addr((int32_t) ((uint32_t) v79 + (uint32_t) v42), v7);
    noc_async_write(v81, temp_296, v27);
    {
    DeviceZoneScopedN("noc_async_write_barrier");
    noc_async_write_barrier();
    }
    cb_pop_front(get_compile_time_arg_val(2), v13);
  }
  return;
}
