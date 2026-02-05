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
  size_t v5 = 4;
  int32_t v6 = 0;
  int32_t v7 = 64;
  int32_t v8 = 128;
  int32_t v9 = 32;
  int32_t v10 = 2;
  int32_t v11 = 1;
  bool v12 = true;
  int32_t v13 = 2048;
  int32_t v14 = 4096;
  int32_t v15 = 8;
  size_t v16 = 19;
  int32_t v17 = 19;
  int32_t v18 = get_common_arg_val<uint32_t>(v3);
  int32_t v19 = get_common_arg_val<uint32_t>(2);
  int32_t v20 = get_common_arg_val<uint32_t>(10);
  int32_t v21 = get_common_arg_val<uint32_t>(12);
  int32_t v22 = get_common_arg_val<uint32_t>(30);
  int32_t v23 = get_common_arg_val<uint32_t>(31);
  int32_t v24 = get_common_arg_val<uint32_t>(32);
  DataFormat v25 = get_dataformat(get_compile_time_arg_val(1));
  int32_t v26 = get_tile_size(get_compile_time_arg_val(1));
  DataFormat v27 = get_dataformat(get_compile_time_arg_val(0));
  int32_t v28 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v29 = get_arg_val<uint32_t>(v4);
  int32_t v30 = get_arg_val<uint32_t>(v3);
  int32_t v31 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) 127) / v8) * (uint32_t) v10);
  bool v32 = v2 < v5;
  size_t v33 = v32 ? v3 : v5;
  size_t v34 = v32 ? 3 : 7;
  int32_t v35 = v21 != (int32_t) ((uint32_t) (v21 / v9) * (uint32_t) v9) & v21 < v6 == v1 ? (int32_t) ((uint32_t) (v21 / v9) + (uint32_t) v11) : v21 / v9;
  for (int32_t i36 = v30; i36 < v29; i36 += v11) {
    int32_t v37 = (int32_t) ((uint32_t) (i36 / v31) * (uint32_t) v10);
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v22 + (uint32_t) 31) / v9) - (uint32_t) v37) < v10 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v22 + (uint32_t) 31) / v9) - (uint32_t) v37) : v10;
    for (int32_t j39 = v6; j39 < ((int32_t) ((uint32_t) v24 + (uint32_t) 63) / v7); j39 += v11) {
      int32_t v40 = (int32_t) ((uint32_t) j39 * (uint32_t) v7);
      cb_reserve_back(get_compile_time_arg_val(0), v10);
      int32_t v41 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v42 = get_semaphore(get_compile_time_arg_val(4));
      if ((ptrdiff_t) get_absolute_logical_y() == (ptrdiff_t) v3 & (v32 & (ptrdiff_t) v2 == (ptrdiff_t) v3 | v2 >= v5 & (ptrdiff_t) v2 == (ptrdiff_t) v5)) {
        InterleavedAddrGenFast<true> v43;
        v43.bank_base_address = v18;
        v43.page_size = v28;
        v43.data_format = v27;
        InterleavedAddrGenFast<true> v44 = v43;
        int32_t v45 = get_write_ptr(get_compile_time_arg_val(0));
        int32_t v46 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v37 + (uint32_t) (i36 % v38))) * (uint32_t) v9) / v9) * (uint32_t) (v19 != (int32_t) ((uint32_t) (v19 / v9) * (uint32_t) v9) & v19 < v6 == v1 ? (int32_t) ((uint32_t) (v19 / v9) + (uint32_t) v11) : v19 / v9))) + (uint32_t) (v40 / v9));
        uint64_t temp_410 = v44.get_noc_addr(v46, v6);
        noc_async_read(temp_410, v45, v28);
        int32_t v47 = (int32_t) ((uint32_t) v45 + (uint32_t) v13);
        uint64_t temp_422 = v44.get_noc_addr((int32_t) ((uint32_t) v46 + (uint32_t) v11), v6);
        noc_async_read(temp_422, v47, v28);
        {
        DeviceZoneScopedN("noc_async_read_barrier");
        noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* v48 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v41);
        noc_semaphore_wait(v48, v16);
        noc_semaphore_set(v48, v3);
        size_t v49 = experimental::convert_logical_y_to_translated(v5);
        size_t v50 = experimental::convert_logical_x_to_translated(v34);
        size_t v51 = experimental::convert_logical_y_to_translated(v3);
        size_t v52 = experimental::convert_logical_x_to_translated(v33);
        int64_t v53 = experimental::get_noc_multicast_addr(v52, v51, v50, v49, v45);
        noc_async_write_multicast(v45, v53, v14, v17);
        {
        DeviceZoneScopedN("noc_async_write_barrier");
        noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint32_t* v54 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v42);
        noc_semaphore_set(v54, v4);
        size_t v55 = experimental::convert_logical_y_to_translated(v5);
        size_t v56 = experimental::convert_logical_x_to_translated(v34);
        size_t v57 = experimental::convert_logical_y_to_translated(v3);
        size_t v58 = experimental::convert_logical_x_to_translated(v33);
        int64_t v59 = experimental::get_noc_multicast_addr(v58, v57, v56, v55, v42);
        noc_semaphore_set_multicast(v42, v59, v17);
      } else {
        size_t v60 = experimental::convert_logical_y_to_translated(v3);
        size_t v61 = experimental::convert_logical_x_to_translated(v33);
        int64_t v62 = get_noc_addr(v61, v60, v41);
        noc_semaphore_inc(v62, v4);
        volatile tt_l1_ptr uint32_t* v63 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v42);
        noc_semaphore_wait(v63, v4);
        noc_semaphore_set(v63, v3);
      }
      cb_push_back(get_compile_time_arg_val(0), v10);
      cb_reserve_back(get_compile_time_arg_val(1), v15);
      InterleavedAddrGenFast<true> v64;
      v64.bank_base_address = v20;
      v64.page_size = v26;
      v64.data_format = v25;
      InterleavedAddrGenFast<true> v65 = v64;
      int32_t v66 = get_write_ptr(get_compile_time_arg_val(1));
      int32_t v67 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v40 / v9) * (uint32_t) v35)) + (uint32_t) ((int32_t) ((uint32_t) ((i36 % v31) / v38) * (uint32_t) v8) / v9));
      uint64_t temp_348 = v65.get_noc_addr(v67, v6);
      noc_async_read(temp_348, v66, v26);
      int32_t v68 = (int32_t) ((uint32_t) v66 + (uint32_t) v13);
      int32_t v69 = (int32_t) ((uint32_t) v67 + (uint32_t) v11);
      uint64_t temp_360 = v65.get_noc_addr(v69, v6);
      noc_async_read(temp_360, v68, v26);
      int32_t v70 = (int32_t) ((uint32_t) v66 + (uint32_t) v14);
      int32_t v71 = (int32_t) ((uint32_t) v67 + (uint32_t) v10);
      uint64_t temp_372 = v65.get_noc_addr(v71, v6);
      noc_async_read(temp_372, v70, v26);
      int32_t v72 = (int32_t) ((uint32_t) v66 + (uint32_t) 6144);
      int32_t v73 = (int32_t) ((uint32_t) v67 + (uint32_t) 3);
      uint64_t temp_384 = v65.get_noc_addr(v73, v6);
      noc_async_read(temp_384, v72, v26);
      int32_t v74 = (int32_t) ((uint32_t) v66 + (uint32_t) 8192);
      uint64_t temp_396 = v65.get_noc_addr((int32_t) ((uint32_t) v67 + (uint32_t) v35), v6);
      noc_async_read(temp_396, v74, v26);
      int32_t v75 = (int32_t) ((uint32_t) v66 + (uint32_t) 10240);
      uint64_t temp_408 = v65.get_noc_addr((int32_t) ((uint32_t) v69 + (uint32_t) v35), v6);
      noc_async_read(temp_408, v75, v26);
      int32_t v76 = (int32_t) ((uint32_t) v66 + (uint32_t) 12288);
      uint64_t temp_420 = v65.get_noc_addr((int32_t) ((uint32_t) v71 + (uint32_t) v35), v6);
      noc_async_read(temp_420, v76, v26);
      int32_t v77 = (int32_t) ((uint32_t) v66 + (uint32_t) 14336);
      uint64_t temp_432 = v65.get_noc_addr((int32_t) ((uint32_t) v73 + (uint32_t) v35), v6);
      noc_async_read(temp_432, v77, v26);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_push_back(get_compile_time_arg_val(1), v15);
    }
  }
  return;
}
