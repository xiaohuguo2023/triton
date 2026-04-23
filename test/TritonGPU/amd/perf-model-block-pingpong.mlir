// Tests that the PerfModel integration in BlockPingpong correctly gates
// ping-pong scheduling based on resource feasibility.
//
// When the PerfModel reports the config is not feasible (e.g. VGPR spill
// predicted), BlockPingpong must skip ping-pong and emit no setprio ops.
// When the config is feasible, ping-pong proceeds normally.
//
// RUN: triton-opt %s -split-input-file \
// RUN:   --tritonamdgpu-block-pingpong="num-stages=2" \
// RUN:   | FileCheck %s

// Feasible config: 64x64 FP16 tile, 4 warps on gfx942.
// VGPR estimate = 96 < 256 => model says valid => ping-pong proceeds.
// CHECK-LABEL: @pingpong_feasible
// CHECK: rocdl.s.setprio
#blocked  = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma    = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared  = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem   = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_feasible(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>,
      %arg3: i32, %arg4: i32) {
    %cst  = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c0   = arith.constant 0 : i32
    %c1   = arith.constant 1 : i32
    %c64  = arith.constant 64 : i32
    %cst_a = arith.constant dense<64> : tensor<64x64xi32, #blocked1>
    %cst_b = arith.constant dense<64> : tensor<64x64xi32, #blocked>
    %a0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %b0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %ra = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %rb = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %xa = tt.expand_dims %ra {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %xb = tt.expand_dims %rb {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %ya = tt.broadcast %xa : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %yb = tt.broadcast %xb : tensor<64x1xi32, #blocked>  -> tensor<64x64xi32, #blocked>
    %pa = tt.broadcast %a0 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1>
    %pb = tt.broadcast %b0 : tensor<64x1x!tt.ptr<f16>, #blocked>  -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %qa = tt.addptr %pa, %ya : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
    %qb = tt.addptr %pb, %yb : tensor<64x64x!tt.ptr<f16>, #blocked>,  tensor<64x64xi32, #blocked>
    %sa = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared,  #smem, mutable>
    %sb = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    %s0a = ttg.memdesc_index %sa[%c0] : !ttg.memdesc<1x64x64xf16, #shared,  #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared,  #smem, mutable>
    %s0b = ttg.memdesc_index %sb[%c0] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    %res:6 = scf.for %iv = %c0 to %c64 step %c1
        iter_args(%acc = %cst, %pa2 = %qa, %pb2 = %qb, %idx = %c0,
                  %sla = %s0a, %slb = %s0b) ->
        (tensor<64x64xf32, #mma>,
         tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>,
         i32,
         !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
         !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>) : i32 {
      %na = tt.addptr %pa2, %cst_a : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
      %nb = tt.addptr %pb2, %cst_b : tensor<64x64x!tt.ptr<f16>, #blocked>,  tensor<64x64xi32, #blocked>
      %la = tt.load %na : tensor<64x64x!tt.ptr<f16>, #blocked1>
      %lb = tt.load %nb : tensor<64x64x!tt.ptr<f16>, #blocked>
      %da = ttg.local_load %sla : !ttg.memdesc<64x64xf16, #shared,  #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %db = ttg.local_load %slb : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %c  = tt.dot %da, %db, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
      %ni = arith.addi %idx, %c1 : i32
      %nr = arith.cmpi slt, %ni, %c1 : i32
      %nx = arith.select %nr, %ni, %c0 : i32
      %nsa = ttg.memdesc_index %sa[%nx] : !ttg.memdesc<1x64x64xf16, #shared,  #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared,  #smem, mutable>
      %nsb = ttg.memdesc_index %sb[%nx] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      ttg.local_store %la, %nsa : tensor<64x64xf16, #blocked1> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.local_store %lb, %nsb : tensor<64x64xf16, #blocked>  -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      scf.yield %c, %na, %nb, %nx, %nsa, %nsb :
        tensor<64x64xf32, #mma>,
        tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>,
        i32,
        !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
        !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    }
    ttg.local_dealloc %sa : !ttg.memdesc<1x64x64xf16, #shared,  #smem, mutable>
    ttg.local_dealloc %sb : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    tt.return
  }
}
