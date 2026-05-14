// datamovement_kernel0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
  size_t v1 = 8192;
  size_t v2 = -1;
  size_t v3 = 12;
  size_t v4 = 0;
  int32_t v5 = 1;
  int32_t v6 = 64;
  int32_t v7 = 63;
  int32_t v8 = 4;
  int32_t v9 = 8192;
  experimental::CircularBuffer cb_ctarg_7(get_compile_time_arg_val(7));
  experimental::CircularBuffer cb_ctarg_8(get_compile_time_arg_val(8));
  int32_t v10 = (int32_t) ((uint32_t) ((int32_t) get_compile_time_arg_val(3)) + (uint32_t) v7) / v6;
  for (int32_t i11 = (int32_t) get_compile_time_arg_val(1); i11 < ((int32_t) get_compile_time_arg_val(0)); i11 += v5) {
    int32_t v12 = i11 / v10;
    int32_t v13 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) get_compile_time_arg_val(4)) + (uint32_t) v7) / v6) - (uint32_t) v12) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) get_compile_time_arg_val(4)) + (uint32_t) v7) / v6) - (uint32_t) v12) : v5;
    for (int32_t j14 = 0; j14 < ((int32_t) ((uint32_t) ((int32_t) get_compile_time_arg_val(2)) + (uint32_t) v7) / v6); j14 += v5) {
      size_t v15 = (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) j14 * (uint32_t) v6)));
      cb_ctarg_8.reserve_back(v8);
      int64_t v16 = get_noc_addr_from_bank_id<true>((int32_t) ((ptrdiff_t) ((ptrdiff_t) (((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) % v3) < (ptrdiff_t) v4 ? ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) % v3 + v3 : ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) % v3)), (int32_t) ((uint32_t) get_compile_time_arg_val(6) + (uint32_t) ((int32_t) ((ptrdiff_t) (((ptrdiff_t) ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) < (ptrdiff_t) v4 ? v2 - ((ptrdiff_t) ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) < (ptrdiff_t) v4 ? v2 - ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) : (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) / v3 : ((ptrdiff_t) ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) < (ptrdiff_t) v4 ? v2 - ((size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) : (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 + (uint32_t) (i11 % v13))) * (uint32_t) v6))) + v15) / v3) * v1)))));
      noc_async_read(v16, cb_ctarg_8.get_write_ptr(), v9);
      cb_ctarg_7.reserve_back(v8);
      int64_t v17 = get_noc_addr_from_bank_id<true>((int32_t) ((ptrdiff_t) ((ptrdiff_t) ((v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) % v3) < (ptrdiff_t) v4 ? (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) % v3 + v3 : (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) % v3)), (int32_t) ((uint32_t) get_compile_time_arg_val(5) + (uint32_t) ((int32_t) ((ptrdiff_t) (((ptrdiff_t) (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) < (ptrdiff_t) v4 ? v2 - ((ptrdiff_t) (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) < (ptrdiff_t) v4 ? v2 - (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) : v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) / v3 : ((ptrdiff_t) (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) < (ptrdiff_t) v4 ? v2 - (v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) : v15 + (size_t) ((ptrdiff_t) ((int32_t) ((uint32_t) ((i11 % v10) / v13) * (uint32_t) v6)))) / v3) * v1)))));
      noc_async_read(v17, cb_ctarg_7.get_write_ptr(), v9);
      {
      DeviceZoneScopedN("noc_async_read_barrier");
      noc_async_read_barrier();
      }
      cb_ctarg_8.push_back(v8);
      cb_ctarg_7.push_back(v8);
    }
  }
  return;
}
