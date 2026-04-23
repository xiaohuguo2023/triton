// Tests the PerfModel LDS capacity guard in LowerLoops.
//
// The guard is inside initSchedule() and fires when estimated LDS usage
// exceeds the device limit, preventing silent miscompile from aliased
// pipeline buffers.
//
// Reachability: initSchedule() is only called when tritonamdgpu-pipeline
// finds pipelined tt.load ops (loadToInfo non-empty), which requires the
// full compilation pipeline (AccelerateAMDMatmul → ScheduleLoops → Pipeline).
// The overflow path is tested via the C++ unit test AMDPerfModel
// (IsValidConfig.ExcessiveNumStagesExceedsLds / LargeBlockSizeWithManyStagesExceedsLds).
//
// This lit test verifies the guard does not fire spuriously on a valid config
// (no false-positive), using post-MFMA IR that goes through the full pipeline.
//
// RUN: triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-schedule-loops="num_stages=2" \
// RUN:   -tritonamdgpu-pipeline="use_async_copy=0" \
// RUN:   2>&1 | FileCheck %s

// 128x64 FP16 kernel on gfx942, 2 pipeline stages.
// LDS estimate = 2*(128*64 + 64*64)*2 B ~ 49 KB < 64 KB.
// Guard must not emit a false-positive error.
// CHECK-NOT: [PerfModel] LDS usage
// CHECK-LABEL: @lds_guard_no_false_positive
#blocked  = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma    = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#shared  = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem   = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @lds_guard_no_false_positive(
      %lb: i32, %ub: i32, %step: i32,
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>) -> tensor<128x64xf32, #mma> {
    %cst   = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c0    = arith.constant 0 : i32
    %cst_a = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %cst_b = arith.constant dense<64> : tensor<64x64xi32, #blocked1>
    %pa  = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %pb  = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked1>
    %sa  = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared,  #smem, mutable>
    %sb  = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16,  #shared1, #smem, mutable>
    %s0a = ttg.memdesc_index %sa[%c0] : !ttg.memdesc<1x128x64xf16, #shared,  #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared,  #smem, mutable>
    %s0b = ttg.memdesc_index %sb[%c0] : !ttg.memdesc<1x64x64xf16,  #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16,  #shared1, #smem, mutable>
    %res:5 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %cst, %a = %pa, %b = %pb, %sla = %s0a, %slb = %s0b) ->
        (tensor<128x64xf32, #mma>,
         tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>,
         !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
         !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>) : i32 {
      %la  = tt.load %a  : tensor<128x64x!tt.ptr<f16>, #blocked>
      %lb2 = tt.load %b  : tensor<64x64x!tt.ptr<f16>, #blocked1>
      %da  = ttg.local_load %sla : !ttg.memdesc<128x64xf16, #shared,  #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %db  = ttg.local_load %slb : !ttg.memdesc<64x64xf16,  #shared1, #smem, mutable> -> tensor<64x64xf16,  #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %c   = tt.dot %da, %db, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      %na  = tt.addptr %a, %cst_a : tensor<128x64x!tt.ptr<f16>, #blocked>,  tensor<128x64xi32, #blocked>
      %nb  = tt.addptr %b, %cst_b : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
      ttg.local_store %la,  %sla : tensor<128x64xf16, #blocked>  -> !ttg.memdesc<128x64xf16, #shared,  #smem, mutable>
      ttg.local_store %lb2, %slb : tensor<64x64xf16,  #blocked1> -> !ttg.memdesc<64x64xf16,  #shared1, #smem, mutable>
      scf.yield %c, %na, %nb, %sla, %slb :
        tensor<128x64xf32, #mma>,
        tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>,
        !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
        !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    }
    ttg.local_dealloc %sa : !ttg.memdesc<1x128x64xf16, #shared,  #smem, mutable>
    ttg.local_dealloc %sb : !ttg.memdesc<1x64x64xf16,  #shared1, #smem, mutable>
    tt.return %res#0 : tensor<128x64xf32, #mma>
  }
}
