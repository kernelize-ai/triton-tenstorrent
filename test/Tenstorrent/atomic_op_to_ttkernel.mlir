// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel -canonicalize | FileCheck %s

// Scalar atomic_add: reads/writes a single 32-bit element through a software
// spinlock built from the one real hardware atomic (noc_semaphore_inc +
// reading its return value back), since noc_semaphore_inc itself cannot
// target DRAM and cannot express add/cas/max/min/and/or/xor directly.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: func.func public @atomic_add_kernel__reader
  tt.func public @atomic_add_kernel__reader(%ptr: !tt.ptr<i32> {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %val = arith.constant 1 : i32
    // Lock acquire: fetch-and-add the lock word, barrier, then read the
    // pre-increment value back from the fixed local L1 return-value slot.
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: ttkernel.noc_semaphore_inc
    // CHECK: ttkernel.noc_async_atomic_barrier
    // CHECK: ttkernel.reinterpret_cast
    // CHECK: ttkernel.load_from_l1
    // Read the current value of the target element, compute the new value,
    // write it back -- all still inside the lock.
    // CHECK: ttkernel.noc_async_read_tile
    // CHECK: ttkernel.noc_async_read_barrier
    // CHECK: arith.addi
    // CHECK: ttkernel.store_to_l1
    // CHECK: ttkernel.noc_async_write_tile
    // CHECK: ttkernel.noc_async_write_barrier
    // Lock release.
    // CHECK: ttkernel.noc_semaphore_inc
    // CHECK: ttkernel.noc_async_atomic_barrier
    %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}

// -----

// Scalar atomic_cas: pure bit-pattern compare + select, no float arithmetic
// needed even for float element types.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: func.func public @atomic_cas_kernel__reader
  tt.func public @atomic_cas_kernel__reader(%ptr: !tt.ptr<i32> {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %cmp = arith.constant 0 : i32
    %val = arith.constant 1 : i32
    // CHECK: ttkernel.load_from_l1
    // CHECK: arith.cmpi eq
    // CHECK: arith.select
    // CHECK: ttkernel.store_to_l1
    %old = tt.atomic_cas relaxed, gpu, %ptr, %cmp, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

// Masked scalar atomic: the whole acquire/critical-section/release sequence
// is guarded by a single scf.if on the (already-scalar) mask.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: func.func public @atomic_add_masked_kernel__reader
  tt.func public @atomic_add_masked_kernel__reader(%ptr: !tt.ptr<i32> {tt.divisibility = 8 : i32}, %mask: i1) attributes {noinline = false} {
    %val = arith.constant 1 : i32
    // CHECK: scf.if
    // CHECK: ttkernel.noc_semaphore_inc
    %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

