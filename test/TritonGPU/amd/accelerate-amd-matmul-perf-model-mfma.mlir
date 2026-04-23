// Tests that selectMfmaNonKDim follows Origami's throughput-based selection:
//   throughput = (M * N * K) / (throughputCycles / numSimdPerCU)
// Tie-breaking rule: prefer 16x16 over 32x32 when throughput is equal.
//
// On CDNA3 (gfx942) and CDNA2 (gfx90a), 32x32 and 16x16 always tie in
// throughput, so the model must select 16x16 (not 32x32) when
// matrix-instruction-size=0 (auto).
//
// RUN: triton-opt %s -split-input-file \
// RUN:   --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=0" \
// RUN:   | FileCheck %s --check-prefix=GFX942-AUTO
//
// RUN: triton-opt %s -split-input-file \
// RUN:   --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=32" \
// RUN:   | FileCheck %s --check-prefix=GFX942-32
//
// RUN: triton-opt %s -split-input-file \
// RUN:   --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx90a matrix-instruction-size=0" \
// RUN:   | FileCheck %s --check-prefix=GFX90A-AUTO

// gfx942 FP16 -> FP32, large tile (128x128).
// throughput(32x32x8)  = 32*32*8 / (64/4) = 512
// throughput(16x16x16) = 16*16*16 / (32/4) = 512  => tie => 16x16 wins.
// GFX942-AUTO: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
// GFX942-32:   #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [32, 32, 8], isTransposed = true}>
// GFX942-AUTO-LABEL: fp16_128x128
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp16_128x128(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %result = tt.dot %a, %b, %c : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

// gfx942 BF16 -> FP32, large tile (128x128).
// throughput(32x32x4)  = 32*32*4 / (64/4) = 256
// throughput(16x16x8)  = 16*16*8 / (32/4) = 256  => tie => 16x16 wins.
// GFX942-AUTO: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
// GFX942-AUTO-LABEL: bf16_128x128
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bf16_128x128(
      %a: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %result = tt.dot %a, %b, %c : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

// gfx942 FP32 -> FP32, large tile (128x128).
// Only 16x16x4 exists for FP32 on CDNA3 => 16x16.
// GFX942-AUTO: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 4], isTransposed = true}>
// GFX942-AUTO-LABEL: fp32_128x128
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp32_128x128(
      %a: tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %result = tt.dot %a, %b, %c : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

// gfx942 FP16, small tile (32x16): blockN == 16 < 32.
// 32x32 requires both blockM >= 32 and blockN >= 32; blockN=16 fails => 16x16.
// GFX942-AUTO: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
// GFX942-AUTO-LABEL: fp16_32x16
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp16_32x16(
      %a: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked> {
    %result = tt.dot %a, %b, %c : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x16xf32, #blocked>
    tt.return %result : tensor<32x16xf32, #blocked>
  }
}

// -----

// gfx90a (CDNA2) FP16 -> FP32, large tile (128x128).
// throughput(32x32x8)  = 32*32*8 / (64/4) = 512
// throughput(16x16x16) = 16*16*16 / (32/4) = 512  => tie => 16x16 wins.
// GFX90A-AUTO: #mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
// GFX90A-AUTO-LABEL: gfx90a_fp16_128x128
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @gfx90a_fp16_128x128(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %result = tt.dot %a, %b, %c : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}
