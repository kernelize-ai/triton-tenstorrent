// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @matmul_kernel__reader()
  tt.func public @matmul_kernel__reader(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked2> 
    %c1_i32 = arith.constant 1 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %true = arith.constant true 
    %c32_i32 = arith.constant 32 : i32 
    %cst_4 = arith.constant dense<32> : tensor<32x32xi32, #blocked2> 
    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable> 
    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable> 
    %pid = tt.get_program_id x : i32 
    %num_pid_m = arith.addi %M, %c31_i32 : i32 
    %num_pid_m_5 = arith.divsi %num_pid_m, %c32_i32 : i32 
    %num_pid_n = arith.addi %N, %c31_i32 : i32 
    %num_pid_n_6 = arith.divsi %num_pid_n, %c32_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_n_6 : i32 
    %group_size_m = arith.subi %num_pid_m_5, %group_id : i32 
    %group_size_m_7 = arith.minsi %group_size_m, %c1_i32 : i32 
    %pid_m = arith.remsi %pid, %num_pid_n_6 : i32 
    %pid_m_8 = arith.remsi %pid_m, %group_size_m_7 : i32 
    %pid_m_9 = arith.addi %group_id, %pid_m_8 : i32 
    %pid_n = arith.divsi %pid_m, %group_size_m_7 : i32 
    %0 = arith.cmpi sge, %pid_m_9, %c0_i32 : i32
    llvm.intr.assume %0 : i1 
    %1 = arith.cmpi sge, %pid_n, %c0_i32 : i32 
    llvm.intr.assume %1 : i1 
    %2 = arith.cmpi sgt, %stride_am, %c0_i32 : i32 
    llvm.intr.assume %2 : i1 
    llvm.intr.assume %true : i1 
    llvm.intr.assume %true : i1 
    %3 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32 
    llvm.intr.assume %3 : i1 
    %4 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32 
    llvm.intr.assume %4 : i1 
    llvm.intr.assume %true : i1 
    %offs_am = arith.muli %pid_m_9, %c32_i32 : i32 
    %offs_am_10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_am_12 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_am_13 = tt.splat %offs_am : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> 
    %offs_am_14 = arith.addi %offs_am_13, %offs_am_10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> 
    %offs_am_15 = tt.splat %M : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> 
    %offs_am_16 = arith.remsi %offs_am_14, %offs_am_15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> 
    %offs_bn = arith.muli %pid_n, %c32_i32 : i32 
    %offs_bn_17 = tt.splat %offs_bn : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> 
    %offs_bn_18 = arith.addi %offs_bn_17, %offs_am_11 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> 
    %offs_bn_19 = tt.splat %N : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> 
    %offs_bn_20 = arith.remsi %offs_bn_18, %offs_bn_19 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> 
    %a_ptrs = tt.expand_dims %offs_am_16 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2> 
    %a_ptrs_21 = tt.splat %stride_am : i32 -> tensor<32x1xi32, #blocked2> 
    %a_ptrs_22 = arith.muli %a_ptrs, %a_ptrs_21 : tensor<32x1xi32, #blocked2> 
    %a_ptrs_23 = tt.expand_dims %offs_am_12 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2> 
    %a_ptrs_24 = tt.broadcast %a_ptrs_22 : tensor<32x1xi32, #blocked2> -> tensor<32x32xi32, #blocked2> 
    %a_ptrs_25 = tt.broadcast %a_ptrs_23 : tensor<1x32xi32, #blocked2> -> tensor<32x32xi32, #blocked2> 
    %a_ptrs_26 = arith.addi %a_ptrs_24, %a_ptrs_25 : tensor<32x32xi32, #blocked2> 
    %a_ptrs_27 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked2> 
    %a_ptrs_28 = tt.addptr %a_ptrs_27, %a_ptrs_26 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2> 
    %b_ptrs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> 
    %b_ptrs_29 = tt.expand_dims %b_ptrs {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1> 
    %b_ptrs_30 = tt.splat %stride_bk : i32 -> tensor<32x1xi32, #blocked1> 
    %b_ptrs_31 = arith.muli %b_ptrs_29, %b_ptrs_30 : tensor<32x1xi32, #blocked1> 
    %b_ptrs_32 = tt.expand_dims %offs_bn_20 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1> 
    %b_ptrs_33 = tt.broadcast %b_ptrs_31 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1> 
    %b_ptrs_34 = tt.broadcast %b_ptrs_32 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1> 
    %b_ptrs_35 = arith.addi %b_ptrs_33, %b_ptrs_34 : tensor<32x32xi32, #blocked1> 
    %b_ptrs_36 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1> 
    %b_ptrs_37 = tt.addptr %b_ptrs_36, %b_ptrs_35 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1> 
    %5 = arith.addi %K, %c31_i32 : i32 
    %6 = arith.divsi %5, %c32_i32 : i32 
    %accumulator:3 = scf.for %accumulator_38 = %c0_i32 to %6 step %c1_i32 iter_args(%accumulator_39 = %cst_1, %a_ptrs_40 = %a_ptrs_28, %b_ptrs_41 = %b_ptrs_37) -> (tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %a_42 = arith.muli %accumulator_38, %c32_i32 : i32 
      %a_43 = arith.subi %K, %a_42 : i32 
      %a_44 = tt.splat %a_43 : i32 -> tensor<1x32xi32, #blocked2>
      %a_45 = arith.cmpi slt, %a_ptrs_23, %a_44 : tensor<1x32xi32, #blocked2> 
      %a_46 = tt.broadcast %a_45 : tensor<1x32xi1, #blocked2> -> tensor<32x32xi1, #blocked2> 
      %a_47 = tt.load %a_ptrs_40, %a_46, %cst_3 : tensor<32x32x!tt.ptr<f16>, #blocked2> 
      ttg.local_store %a_47, %a : tensor<32x32xf16, #blocked2> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable> 
      %b_48 = tt.splat %a_43 : i32 -> tensor<32x1xi32, #blocked1> 
      %b_49 = arith.cmpi slt, %b_ptrs_29, %b_48 : tensor<32x1xi32, #blocked1> 
      %b_50 = tt.broadcast %b_49 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1> 
      %b_51 = tt.load %b_ptrs_41, %b_50, %cst_2 : tensor<32x32x!tt.ptr<f16>, #blocked1> 
      ttg.local_store %b_51, %b : tensor<32x32xf16, #blocked1> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable> 
      %accumulator_52 = tt.dot %cst_0, %cst, %accumulator_39 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked> 
      %a_ptrs_53 = tt.addptr %a_ptrs_40, %cst_4 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2> 
      %b_ptrs_54 = arith.muli %stride_bk, %c32_i32 : i32 
      %b_ptrs_55 = tt.splat %b_ptrs_54 : i32 -> tensor<32x32xi32, #blocked1> 
      %b_ptrs_56 = tt.addptr %b_ptrs_41, %b_ptrs_55 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1> 
      scf.yield %accumulator_52, %a_ptrs_53, %b_ptrs_56 : tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1> 
    }
    // CHECK: return
    tt.return 
  } 
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: func.func public @matmul_kernel__compute()
    tt.func public @matmul_kernel__compute(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
        %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked> 
        %c31_i32 = arith.constant 31 : i32 
        %c1_i32 = arith.constant 1 : i32 
        %c0_i32 = arith.constant 0 : i32 
        %true = arith.constant true 
        %c32_i32 = arith.constant 32 : i32 
        %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %pid = tt.get_program_id x : i32 
        %num_pid_m = arith.addi %M, %c31_i32 : i32 
        %num_pid_m_0 = arith.divsi %num_pid_m, %c32_i32 : i32 
        %num_pid_n = arith.addi %N, %c31_i32 : i32 
        %num_pid_n_1 = arith.divsi %num_pid_n, %c32_i32 : i32 
        %group_id = arith.divsi %pid, %num_pid_n_1 : i32 
        %group_size_m = arith.subi %num_pid_m_0, %group_id : i32 
        %group_size_m_2 = arith.minsi %group_size_m, %c1_i32 : i32 
        %pid_m = arith.remsi %pid, %num_pid_n_1 : i32 
        %pid_m_3 = arith.remsi %pid_m, %group_size_m_2 : i32 
        %pid_m_4 = arith.addi %group_id, %pid_m_3 : i32 
        %pid_n = arith.divsi %pid_m, %group_size_m_2 : i32 
        %1 = arith.cmpi sge, %pid_m_4, %c0_i32 : i32 
        llvm.intr.assume %1 : i1 
        %2 = arith.cmpi sge, %pid_n, %c0_i32 : i32 
        llvm.intr.assume %2 : i1 
        %3 = arith.cmpi sgt, %stride_am, %c0_i32 : i32 
        llvm.intr.assume %3 : i1 
        llvm.intr.assume %true : i1 
        llvm.intr.assume %true : i1 
        %4 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32 
        llvm.intr.assume %4 : i1 
        %5 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32 
        llvm.intr.assume %5 : i1 
        llvm.intr.assume %true : i1 
        %6 = arith.addi %K, %c31_i32 : i32 
        %7 = arith.divsi %6, %c32_i32 : i32 
        %accumulator = scf.for %accumulator_5 = %c0_i32 to %7 step %c1_i32 iter_args(%arg10 = %cst) -> (tensor<32x32xf32, #blocked>)  : i32 {
        %a_6 = ttg.local_load %a : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> 
        %b_7 = ttg.local_load %b : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> 
        %accumulator_8 = tt.dot %a_6, %b_7, %arg10 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked> 
        scf.yield %accumulator_8 : tensor<32x32xf32, #blocked> 
        } 
        %c = arith.truncf %accumulator : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked> 
        ttg.local_store %c, %0 : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable> 
        // CHECK: return
        tt.return
  } 
}
