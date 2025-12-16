// RUN: triton-opt %s -split-input-file --tritontenstorrent-convert-tensor-desc -canonicalize | FileCheck %s

module {
  // CHECK: @matmul_kernel_tma
  tt.func public @matmul_kernel_tma(%a_desc: !tt.tensordesc<tensor<32x32xf16>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<32x32xf16>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc: !tt.tensordesc<tensor<32x32xf16>>, %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %accumulator = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    // CHECK-DAG: %[[cst:.*]] = arith.constant dense<32> : tensor<32x32xi32>
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[c1024_i32:.*]] = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c31_i32 : i32
    %num_pid_m_12 = arith.divsi %num_pid_m, %c32_i32 : i32
    %num_pid_n = arith.addi %N, %c31_i32 : i32
    %num_pid_n_13 = arith.divsi %num_pid_n, %c32_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_13, %c8_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_in_group : i32
    %first_pid_m = arith.muli %group_id, %c8_i32 : i32
    %group_size_m = arith.subi %num_pid_m_12, %first_pid_m : i32
    %group_size_m_14 = arith.minsi %group_size_m, %c8_i32 : i32
    %pid_m = arith.remsi %pid, %group_size_m_14 : i32
    %pid_m_15 = arith.addi %first_pid_m, %pid_m : i32
    %pid_n = arith.remsi %pid, %num_pid_in_group : i32
    %pid_n_16 = arith.divsi %pid_n, %group_size_m_14 : i32
    %k_tiles = arith.addi %K, %c31_i32 : i32
    %k_tiles_17 = arith.divsi %k_tiles, %c32_i32 : i32
    // CHECK: %[[num_pid_in_group:.*]] = arith.muli
    // CHECK: %[[first_pid_m:.*]] = arith.muli
    // CHECK: %[[offs_am:.*]] = arith.muli
    // CHECK: %[[offs_bn:.*]] = arith.muli
    %offs_am = arith.muli %pid_m_15, %c32_i32 : i32
    %offs_bn = arith.muli %pid_n_16, %c32_i32 : i32
    %accumulator_18 = scf.for %k = %c0_i32 to %k_tiles_17 step %c1_i32 iter_args(%accumulator_19 = %accumulator) -> (tensor<32x32xf32>)  : i32 {
    // CHECK: scf.for {{.*}} {
      %offs_k = arith.muli %k, %c32_i32 : i32
      // CHECK: %[[offs_k:.*]] = arith.muli

      // COM: A matrix row and column tile offsets
      // CHECK: %[[A_ROW_BASE:.*]] = tt.splat %[[offs_am]] : i32
      // CHECK: %[[RANGE0:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32}
      // CHECK: %[[A_ROW_IDX:.*]] = arith.addi %[[A_ROW_BASE]], %[[RANGE0]]
      // CHECK: %[[A_ROW_IDX_EXP:.*]] = tt.expand_dims %[[A_ROW_IDX]] {axis = 1 : i32}

      // CHECK: %[[A_COL_BASE:.*]] = tt.splat %[[offs_k]] : i32
      // CHECK: %[[RANGE1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32}
      // CHECK: %[[A_COL_IDX:.*]] = arith.addi %[[A_COL_BASE]], %[[RANGE1]]
      // CHECK: %[[A_COL_IDX_EXP:.*]] = tt.expand_dims %[[A_COL_IDX]] {axis = 0 : i32}

      // COM: A matrix Tile ID
      // CHECK: %[[A_TILE_M:.*]] = arith.divsi %[[offs_am]], %[[c32_i32]]
      // CHECK: %[[A_TILE_K:.*]] = arith.divsi %[[offs_k]], %[[c32_i32]]
      // CHECK: %[[A_NTILES:.*]] = arith.ceildivsi %{{.*}}, %[[c32_i32]]
      // CHECK: %[[A_TILE_ROW_OFFSET:.*]] = arith.muli %[[A_TILE_M]], %[[A_NTILES]]
      // CHECK: %[[A_TILE_ROWCOL_OFFSET:.*]] = arith.addi %[[A_TILE_ROW_OFFSET]], %[[A_TILE_K]]
      // CHECK: %[[A_TILE_BASE_ELEMENTS:.*]] = arith.muli %[[A_TILE_ROWCOL_OFFSET]], %[[c1024_i32]]

      // COM: A matrix intra-tile offsets
      // CHECK: %[[A_ROW_BCAST:.*]] = tt.broadcast %[[A_ROW_IDX_EXP]]
      // CHECK: %[[A_ROW_INTRA:.*]] = arith.remui %[[A_ROW_BCAST]], %[[cst]]
      // CHECK: %[[A_COL_BCAST:.*]] = tt.broadcast %[[A_COL_IDX_EXP]]
      // CHECK: %[[A_COL_INTRA:.*]] = arith.remui %[[A_COL_BCAST]], %[[cst]]
      // CHECK: %[[A_ROW_STRIDE:.*]] = arith.muli %[[A_ROW_INTRA]], %[[cst]]
      // CHECK: %[[A_INTRA:.*]] = arith.addi %[[A_ROW_STRIDE]], %[[A_COL_INTRA]]

      // COM: A matrix load
      // CHECK: %[[A_BASE_SPLAT:.*]] = tt.splat %[[A_TILE_BASE_ELEMENTS]] : i32 -> tensor<32x32xi32>
      // CHECK: %[[A_ELEM_OFFSET:.*]] = arith.addi %[[A_BASE_SPLAT]], %[[A_INTRA]]
      // CHECK: %[[A_PTR_BASE:.*]] = tt.splat %{{.*}} : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
      // CHECK: %[[A_PTRS:.*]] = tt.addptr %[[A_PTR_BASE]], %[[A_ELEM_OFFSET]]
      // CHECK: %[[A_LOAD:.*]] = tt.load %[[A_PTRS]]

      // CHECK-NOT: tt.descriptor_load
      // CHECK-NOT: tt.descriptor_load
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] : !tt.tensordesc<tensor<32x32xf16>> -> tensor<32x32xf16>
      %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] : !tt.tensordesc<tensor<32x32xf16>> -> tensor<32x32xf16>
      %accumulator_20 = tt.dot %a, %b, %accumulator_19 : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      scf.yield %accumulator_20 : tensor<32x32xf32>
      // CHECK: scf.yield
    // CHECK: }
    }
    %c = arith.truncf %accumulator_18 : tensor<32x32xf32> to tensor<32x32xf16>
    %offs_cm = arith.muli %pid_m_15, %c32_i32 : i32
    %offs_cn = arith.muli %pid_n_16, %c32_i32 : i32
    // CHECK-NOT: tt.descriptor_store

    // CHECK: %[[M_TILE:.*]] = arith.divsi {{.*}}, %[[c32_i32]]
    // CHECK: %[[N_TILE:.*]] = arith.divsi {{.*}}, %[[c32_i32]]
    // CHECK: %[[M_TILES:.*]] = arith.ceildivsi {{.*}}, %[[c32_i32]]
    // CHECK: %[[C_M_OFFSET:.*]] = arith.muli %[[M_TILE]], %[[M_TILES]]
    // CHECK: %[[C_TILE_ID:.*]] = arith.addi %[[C_M_OFFSET]], %[[N_TILE]]
    // CHECK: %[[C_TILE_ELEM_BASE:.*]] = arith.muli %[[C_TILE_ID]], %[[c1024_i32]]

    // CHECK: %[[C_INTRA_OFFSET:.*]] = arith.addi

    // CHECK: %[[C_TILE_ID_TENSOR:.*]] = tt.splat %[[C_TILE_ELEM_BASE]]
    // CHECK: %[[C_OFFSET:.*]] = arith.addi %[[C_TILE_ID_TENSOR]], %[[C_INTRA_OFFSET]]

    // CHECK: %[[BASE_PTR:.*]] = tt.splat %{{.*}}
    // CHECK: %[[OUT_PTRS:.*]] = tt.addptr %[[BASE_PTR]], %[[C_OFFSET]]
    // CHECK: tt.store %[[OUT_PTRS]], {{.*}}
    tt.descriptor_store %c_desc[%offs_cm, %offs_cn], %c : !tt.tensordesc<tensor<32x32xf16>>, tensor<32x32xf16>
    tt.return
  }
}
