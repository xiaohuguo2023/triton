//===-- PerfModel.cpp - Triton AMD analytical performance model -----------===//
//
// Implementation of the analytical performance model declared in PerfModel.h.
//
// Modelling approach
// ──────────────────
// The model is a three-layer stack:
//
//  Layer 1 – Hardware database
//    Static tables of per-architecture constants: CU count, SIMD width, VGPR
//    budget, LDS capacity, memory bandwidth, and MFMA reciprocal throughput.
//    Values are derived from AMD ISA documentation and calibrated against
//    published peak TFLOPS figures.
//
//  Layer 2 – Resource accounting
//    VGPR model:  accumulator tile + A/B register fragments + misc overhead.
//    LDS model:   numBuffers × (A tile + B tile) with 8-element row padding.
//    Occupancy:   min(VGPR-limited waves, LDS-limited CTAs) × waveSize.
//
//  Layer 3 – Roofline + wave quantisation
//    Per-tile cycles = max(compute_cycles, (1 - pipeline_overlap)*mem_cycles).
//    Total cycles    = per_tile_cycles × numWaves.
//    Pipeline overlap approximates the fraction of memory latency hidden by
//    software pipelining as a function of numStages.
//    Wave quantisation captures the "tail wave" inefficiency when the number
//    of output tiles is not a multiple of the CU count.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/PerfModel.h"

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mlir::triton::AMD::perf {

//===----------------------------------------------------------------------===//
// 1. Hardware database
//===----------------------------------------------------------------------===//

Arch archFromString(llvm::StringRef s) {
  // Accept both exact names and prefixes ("gfx942a" → CDNA3).
  if (s == "gfx908")
    return Arch::CDNA1;
  if (s == "gfx90a")
    return Arch::CDNA2;
  if (s.starts_with("gfx940") || s.starts_with("gfx941") ||
      s.starts_with("gfx942"))
    return Arch::CDNA3;
  if (s.starts_with("gfx950"))
    return Arch::CDNA4;
  if (s == "gfx1250")
    return Arch::GFX1250;
  if (s.starts_with("gfx12"))
    return Arch::RDNA4;
  if (s.starts_with("gfx11"))
    return Arch::RDNA3;
  return Arch::Unknown;
}

// Peak FP16→FP32 MFMA throughput via the 32×32×8 instruction.
//   FLOP/cycle/CU = (32*32*8*2 FLOPs) / (throughput_cycles) * numSimdPerCU
//
// Validated against published peak TFLOPS:
//   MI210 (CDNA2, 104 CUs, 1.7 GHz):
//     1024 FLOP/cycle/CU × 1.7e9 × 104 ≈ 181 TFLOPS FP16  ✓
double HardwareInfo::peakMfmaFlopsPerCycleCU() const {
  // FLOPs per 32x32x8 F16 MFMA instruction
  constexpr double mfmaFlops = 32.0 * 32.0 * 8.0 * 2.0; // = 16384
  // Reciprocal throughput for 32x32 MFMA on each arch (cycles per instr/SIMD)
  double throughputCycles = 64.0; // default (CDNA1/2/3)
  switch (arch) {
  case Arch::CDNA4:
    throughputCycles = 64.0; // same throughput, but wider VGPR file
    break;
  case Arch::RDNA3:
  case Arch::RDNA4:
  case Arch::GFX1250:
    // RDNA uses WMMA 16×16; express equivalent MFMA peak for roofline.
    // WMMA_F32_16x16x16F16: 16*16*16*2 / 32 cycles = 256 FLOP/cycle/SIMD
    // numSimdPerCU = 2, so 512 FLOP/cycle/CU.
    return 512.0;
  default:
    break;
  }
  return (mfmaFlops / throughputCycles) * numSimdPerCU;
}

HardwareInfo HardwareInfo::get(llvm::StringRef archStr) {
  return get(archFromString(archStr));
}

HardwareInfo HardwareInfo::get(Arch arch) {
  HardwareInfo hw;
  hw.arch = arch;

  switch (arch) {
  // ── CDNA1  gfx908  MI100 ────────────────────────────────────────────────
  case Arch::CDNA1:
    hw.numCUs = 120;
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 256;
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 65536;         // 64 KB
    hw.l2SizeBytes = 8 << 20;    // 8 MB total, monolithic die
    hw.mallSizeBytes = 0;
    hw.numXCDs = 1;
    hw.clockMHz = 1500.0;
    hw.peakMemBwBytesPerCycle = 800.0;
    hw.peakL2BwBytesPerCycle = 0.0;  // TODO: calibrate
    break;

  // ── CDNA2  gfx90a  MI200 (MI210 / MI250) ────────────────────────────────
  case Arch::CDNA2:
    hw.numCUs = 104;           // MI210; MI250X has 110 CUs
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 256;
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 65536;         // 64 KB
    hw.l2SizeBytes = 8 << 20;    // 8 MB per die; 2 dies in MI250X
    hw.mallSizeBytes = 0;
    hw.numXCDs = 2;              // MI250X is 2-die; MI210 is 1-die (conservative)
    hw.clockMHz = 1700.0;
    hw.peakMemBwBytesPerCycle = 941.0;
    hw.peakL2BwBytesPerCycle = 0.0;  // TODO: calibrate
    break;

  // ── CDNA3  gfx940/941/942  MI300 ────────────────────────────────────────
  case Arch::CDNA3:
    hw.numCUs = 228;           // MI300X; 228 CUs across 8 XCDs
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 256;
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 65536;         // 64 KB
    // Origami: NUM_XCD=8, L2_capacity read from HIP at runtime.
    // Estimate: MI300X has ~256 MB L2 across 8 XCDs → 32 MB per XCD.
    hw.l2SizeBytes = 32 << 20;   // 32 MB per XCD
    hw.mallSizeBytes = 0;
    hw.numXCDs = 8;
    hw.clockMHz = 2100.0;
    hw.peakMemBwBytesPerCycle = 2524.0;
    hw.peakL2BwBytesPerCycle = 0.0;  // TODO: calibrate
    break;

  // ── CDNA4  gfx950  MI350 (MI355X) ──────────────────────────────────────
  case Arch::CDNA4:
    hw.numCUs = 256;             // MI355X: 256 CUs
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 512;        // Doubled VGPR file vs CDNA3
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 163840;        // 160 KB (from TargetInfo.cpp)
    // Origami uses NUM_XCD=8 for gfx950 (same as gfx942 MI300X).
    // L2 capacity per XCD: device L2 / 8 XCDs.
    hw.l2SizeBytes = 32 << 20;   // 32 MB per XCD (256 MB total / 8 XCDs)
    hw.mallSizeBytes = 0;
    hw.numXCDs = 8;              // Origami: get_default_num_xcds(gfx950) = 8
    hw.clockMHz = 2400.0;
    hw.peakMemBwBytesPerCycle = 3000.0;  // ~7.2 TB/s / 2.4 GHz
    // Calibrated value from testtmp/calibrate_l2_bw.py: ~17900 bytes/cycle
    // (peak effective BW ~43 TB/s observed at M=2048 BK=128 on MI355X).
    // Set to 0 until BK candidate capping is implemented to prevent the L2
    // model from over-preferring large BK at problem sizes where it hurts.
    hw.peakL2BwBytesPerCycle = 0.0;  // re-enable after BK cap fix
    break;

  // ── RDNA3  gfx1100/1101/1102 ────────────────────────────────────────────
  case Arch::RDNA3:
    hw.numCUs = 60;            // RX 7900 XTX
    hw.numSimdPerCU = 2;
    hw.waveSize = 32;
    hw.vgprPerSimd = 1536;     // 48 KB VGPR file per SIMD, 32-bit lanes
    hw.vgprAllocGranule = 8;
    hw.maxWavesPerSimd = 16;
    hw.ldsPerCU = 65536;        // 64 KB
    hw.l2SizeBytes = 6 << 20;   // 6 MB
    hw.mallSizeBytes = 96 << 20; // 96 MB Infinity Cache
    hw.clockMHz = 2500.0;
    // Peak BW: ~960 GB/s  →  ≈ 384 bytes/cycle
    hw.peakMemBwBytesPerCycle = 384.0;
    break;

  // ── RDNA4  gfx1200/1201 ─────────────────────────────────────────────────
  case Arch::RDNA4:
    hw.numCUs = 64;
    hw.numSimdPerCU = 2;
    hw.waveSize = 32;
    hw.vgprPerSimd = 1536;
    hw.vgprAllocGranule = 8;
    hw.maxWavesPerSimd = 16;
    hw.ldsPerCU = 65536;
    hw.l2SizeBytes = 8 << 20;
    hw.mallSizeBytes = 128 << 20;
    hw.clockMHz = 3000.0;
    hw.peakMemBwBytesPerCycle = 512.0;
    break;

  // ── GFX1250 ─────────────────────────────────────────────────────────────
  case Arch::GFX1250:
    hw.numCUs = 40;
    hw.numSimdPerCU = 2;
    hw.waveSize = 32;
    hw.vgprPerSimd = 1536;
    hw.vgprAllocGranule = 8;
    hw.maxWavesPerSimd = 16;
    hw.ldsPerCU = 327680;       // 320 KB (from TargetInfo.cpp)
    hw.l2SizeBytes = 4 << 20;
    hw.mallSizeBytes = 64 << 20;
    hw.clockMHz = 2900.0;
    hw.peakMemBwBytesPerCycle = 480.0;
    break;

  default:
    // Leave all fields zeroed for Unknown arch.
    break;
  }
  return hw;
}

//===----------------------------------------------------------------------===//
// 2. MFMA throughput table
//===----------------------------------------------------------------------===//

int elemKindBits(ElemKind k) {
  switch (k) {
  case ElemKind::FP64: return 64;
  case ElemKind::FP32: return 32;
  case ElemKind::TF32: return 32;
  case ElemKind::FP16: return 16;
  case ElemKind::BF16: return 16;
  case ElemKind::FP8:  return 8;
  case ElemKind::FP6:  return 6;
  case ElemKind::FP4:  return 4;
  case ElemKind::I8:   return 8;
  default:             return 0;
  }
}

ElemKind elemKindFromBits(int bits, bool isFloat, bool isBF) {
  if (!isFloat) {
    if (bits == 8) return ElemKind::I8;
    return ElemKind::Unknown;
  }
  switch (bits) {
  case 64: return ElemKind::FP64;
  case 32: return ElemKind::FP32;
  case 16: return isBF ? ElemKind::BF16 : ElemKind::FP16;
  case 8:  return ElemKind::FP8;
  case 6:  return ElemKind::FP6;
  case 4:  return ElemKind::FP4;
  default: return ElemKind::Unknown;
  }
}

// Reciprocal throughput table: cycles per MFMA instruction per SIMD unit.
//
// Sources:
//   AMD ISA documentation for each GFX generation.
//   Cross-checked against Origami's INSTRUCTION_MAP in hardware.hpp and
//   against published peak TFLOPS (see validation comment in
//   peakMfmaFlopsPerCycleCU above).
//
// Table layout: {arch, mDim, nDim, kDim, throughputCycles, aKind, cKind}.
// Only representative entries are listed; getMfmaInstrInfo() matches on
// (arch, mDim, nDim, aKind, cKind) and ignores kDim for look-up purposes
// (kDim is included as an informational field in the returned descriptor).
struct ThroughputEntry {
  Arch arch;
  int mDim, nDim, kDim;
  int throughputCycles;
  ElemKind aKind, cKind;
};

// clang-format off
static constexpr ThroughputEntry kMfmaThroughputTable[] = {
  // ── CDNA1  (gfx908) ──────────────────────────────────────────────────────
  {Arch::CDNA1, 32, 32,  8, 64, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA1, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA1,  4,  4,  4,  8, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA1, 32, 32,  8, 64, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA1, 16, 16, 16, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA1, 16, 16,  4, 64, ElemKind::FP64,  ElemKind::FP64},
  {Arch::CDNA1, 32, 32,  8, 64, ElemKind::I8,    ElemKind::I8},
  {Arch::CDNA1, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── CDNA2  (gfx90a) ──────────────────────────────────────────────────────
  {Arch::CDNA2, 32, 32,  8, 64, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA2, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA2,  4,  4,  4,  8, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA2, 32, 32,  4, 64, ElemKind::BF16,  ElemKind::FP32}, // packed
  {Arch::CDNA2, 16, 16,  8, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA2, 16, 16,  4, 64, ElemKind::FP64,  ElemKind::FP64},
  {Arch::CDNA2, 32, 32,  8, 64, ElemKind::I8,    ElemKind::I8},
  {Arch::CDNA2, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── CDNA3  (gfx940/941/942) ───────────────────────────────────────────────
  {Arch::CDNA3, 32, 32,  8, 64, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA3, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA3, 32, 32,  4, 64, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA3, 16, 16,  8, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA3, 32, 32,  8, 64, ElemKind::TF32,  ElemKind::FP32},
  {Arch::CDNA3, 16, 16, 16, 32, ElemKind::TF32,  ElemKind::FP32},
  // FP8 (any E4M3/E5M2 variant)
  {Arch::CDNA3, 32, 32, 16, 64, ElemKind::FP8,   ElemKind::FP32},
  {Arch::CDNA3, 16, 16, 32, 32, ElemKind::FP8,   ElemKind::FP32},
  {Arch::CDNA3, 16, 16,  4, 64, ElemKind::FP64,  ElemKind::FP64},
  {Arch::CDNA3, 32, 32,  8, 64, ElemKind::I8,    ElemKind::I8},
  {Arch::CDNA3, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── CDNA4  (gfx950) ───────────────────────────────────────────────────────
  // gfx950 introduces wider MFMA variants (mfma_f32_32x32x16_f16,
  // mfma_f32_16x16x32_f16) with double the kDim vs CDNA3. Same latency in
  // cycles but processes 2× more K elements per instruction, halving the
  // number of MFMA ops per tile. Source: MfmaGroup.cpp TRITON_MFMA_v4_2case.
  {Arch::CDNA4, 32, 32, 16, 64, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA4, 16, 16, 32, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA4, 32, 32, 16, 64, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA4, 16, 16, 32, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA4, 32, 32, 16, 64, ElemKind::FP8,   ElemKind::FP32},
  {Arch::CDNA4, 16, 16, 32, 32, ElemKind::FP8,   ElemKind::FP32},
  {Arch::CDNA4, 32, 32, 32, 64, ElemKind::FP6,   ElemKind::FP32},
  {Arch::CDNA4, 16, 16, 64, 32, ElemKind::FP6,   ElemKind::FP32},
  {Arch::CDNA4, 32, 32, 64, 64, ElemKind::FP4,   ElemKind::FP32},
  {Arch::CDNA4, 16, 16,128, 32, ElemKind::FP4,   ElemKind::FP32},
  {Arch::CDNA4, 16, 16,  4, 64, ElemKind::FP64,  ElemKind::FP64},
  {Arch::CDNA4, 32, 32,  8, 64, ElemKind::I8,    ElemKind::I8},
  {Arch::CDNA4, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── RDNA3  (gfx11xx)  –  WMMA 16×16 only ─────────────────────────────────
  {Arch::RDNA3, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::RDNA3, 16, 16, 16, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::RDNA3, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── RDNA4  (gfx12xx) ─────────────────────────────────────────────────────
  {Arch::RDNA4, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::RDNA4, 16, 16, 16, 32, ElemKind::BF16,  ElemKind::FP32},
  {Arch::RDNA4, 16, 16, 16, 32, ElemKind::FP8,   ElemKind::FP32},
  {Arch::RDNA4, 16, 16, 16, 32, ElemKind::I8,    ElemKind::I8},

  // ── GFX1250 ──────────────────────────────────────────────────────────────
  {Arch::GFX1250, 16, 16,  16, 32, ElemKind::FP16, ElemKind::FP32},
  {Arch::GFX1250, 16, 16,  16, 32, ElemKind::BF16, ElemKind::FP32},
  {Arch::GFX1250, 16, 16,  16, 32, ElemKind::FP8,  ElemKind::FP32},
  // Scaled WMMA (16×16×128 for FP4)
  {Arch::GFX1250, 16, 16, 128, 32, ElemKind::FP4,  ElemKind::FP32},
};
// clang-format on

std::optional<MfmaInstrInfo> getMfmaInstrInfo(Arch arch, int mDim, int nDim,
                                               ElemKind aKind, ElemKind cKind) {
  for (const auto &e : kMfmaThroughputTable) {
    if (e.arch == arch && e.mDim == mDim && e.nDim == nDim &&
        e.aKind == aKind && e.cKind == cKind) {
      return MfmaInstrInfo{mDim, nDim, e.kDim, e.throughputCycles, aKind,
                           cKind};
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// 3. MFMA instruction size selection
//===----------------------------------------------------------------------===//
//
// Placed before the resource-accounting section because deriveKWidth() (below)
// calls selectMfmaNonKDim() when cfg.kWidth == 0 and cfg.mfmaNonKDim == 0.
// Defining it here removes any ambiguity about definition order.

int selectMfmaNonKDim(const GemmProblem &prob, const TritonGemmConfig &cfg,
                      const HardwareInfo &hw) {
  // Tiny tiles: no standard 16x16 or 32x32 MFMA available.
  if (std::min(cfg.blockM, cfg.blockN) < 16)
    return 4;

  // Iterate over kMfmaThroughputTable and pick the square MFMA instruction
  // with the highest throughput, matching Origami's
  // get_recommended_matrix_instruction() logic:
  //
  //   throughput = (M * N * K) / (throughputCycles / numSimdPerCU)
  //
  // Tie-breaking rule: prefer 16x16 over 32x32 when throughput is equal.
  // On CDNA3/4 (gfx942/gfx950), 32x32 and 16x16 always tie, so 16x16 wins.
  // On CDNA2 (gfx90a) with BF16, 32x32 genuinely has higher throughput.
  int bestDim = 0;
  double bestThroughput = -1.0;

  for (const auto &e : kMfmaThroughputTable) {
    if (e.arch != hw.arch || e.aKind != prob.aKind || e.cKind != prob.cKind)
      continue;
    // Only consider square MFMA tiles (mDim == nDim).
    if (e.mDim != e.nDim)
      continue;
    // Block must be at least as large as the instruction tile.
    if (cfg.blockM < e.mDim || cfg.blockN < e.nDim)
      continue;

    int effectiveCycles = e.throughputCycles / hw.numSimdPerCU;
    if (effectiveCycles <= 0)
      continue;

    double throughput =
        static_cast<double>(e.mDim * e.nDim * e.kDim) / effectiveCycles;

    bool isBetter = throughput > bestThroughput;
    bool isTiePrefer16 =
        (throughput == bestThroughput) && (e.mDim == 16) && (bestDim != 16);

    if (isBetter || isTiePrefer16) {
      bestThroughput = throughput;
      bestDim = e.mDim;
    }
  }

  // Fallback: unknown arch or dtype not in table — use 16x16.
  if (bestDim == 0)
    return 16;

  return bestDim;
}

//===----------------------------------------------------------------------===//
// 4. Resource accounting
//===----------------------------------------------------------------------===//

int estimateNumBuffers(const TritonGemmConfig &cfg) {
  // Mirror LowerLoops.cpp::initSchedule() logic:
  //   async copy pipeline  →  numBuffers = numStages
  //   synchronous pipeline →  numBuffers = max(1, numStages - 1)
  if (cfg.bypassLds)
    return 1; // operands live in registers; only the C tile needs buffering
  if (cfg.useAsyncCopy)
    return cfg.numStages;
  return std::max(1, cfg.numStages - 1);
}

/// Derive the effective kWidth from the MFMA throughput table when the caller
/// has not set cfg.kWidth explicitly (cfg.kWidth == 0).
///
/// This mirrors AccelerateAMDMatmul.cpp's own derivation:
///   kBase = MfmaIntrinsic.kBase  (elements per thread for one MFMA issue)
///   kWidth = kBase * kPack        (kPack ∈ {1, 2}, pass option)
///
/// kBase itself is determined by the intrinsic shape and wave size.  From the
/// AccelerateAMDMatmul.cpp comments (lines 679-704):
///   mfma_32x32: kBase = kDim / 2
///   mfma_16x16: kBase = kDim / 4
///   mfma_4x4:   kBase = kDim / 16
///
/// These ratios express how many k-elements each of the 64 lanes in a wave64
/// wavefront receives when the kDim elements are distributed across threads.
static int deriveKWidth(const GemmProblem &prob, const TritonGemmConfig &cfg,
                        const HardwareInfo &hw) {
  if (cfg.kWidth > 0)
    return cfg.kWidth; // caller provided the exact value (e.g. from
                       // DotOperandEncodingAttr::getKWidth())

  const int mDim = cfg.mfmaNonKDim > 0
                       ? cfg.mfmaNonKDim
                       : selectMfmaNonKDim(prob, cfg, hw);

  auto infoOpt = getMfmaInstrInfo(hw.arch, mDim, mDim, prob.aKind, prob.cKind);
  if (!infoOpt)
    return 8; // safe fallback for unknown intrinsics

  // kBase: elements per thread per MFMA issue.
  // From AccelerateAMDMatmul.cpp (MfmaGroup::kBase field):
  //   mfma_32x32: kBase = kDim / 2
  //   mfma_16x16: kBase = kDim / 4
  //   mfma_4x4:   kBase = kDim / 16
  // This matches the TRITON_MFMA_v4_2case entries in MfmaGroup.cpp where
  // kBase is explicitly listed (e.g. 32x32x16 has kBase=8 = 16/2).
  const int divisor = (mDim >= 32) ? 2 : (mDim >= 16) ? 4 : 16;
  const int kBase = std::max(1, infoOpt->kDim / divisor);
  return kBase * std::max(1, cfg.kPack);
}

int estimateVgpr(const GemmProblem &prob, const TritonGemmConfig &cfg,
                 const HardwareInfo &hw) {
  // Each VGPR holds 4 bytes (32-bit register).
  constexpr int bytesPerVgpr = 4;
  const int ws = hw.waveSize;

  // Accumulator tile: BLOCK_M × BLOCK_N elements of the C type, distributed
  // across all lanes across all warps in the CTA.
  // Each warp covers (BLOCK_M × BLOCK_N / numWarps) output elements,
  // distributed across waveSize lanes: vgprAccum per warp = that / waveSize.
  const int cBytes = (prob.cBits + 7) / 8;
  const int numWarps = std::max(1, cfg.numWarps);
  const int vgprAccum =
      (cfg.blockM * cfg.blockN * cBytes + numWarps * ws * bytesPerVgpr - 1) /
      (numWarps * ws * bytesPerVgpr);

  // A and B register fragments: kWidth elements per lane, each of width aBits.
  // kWidth is derived from the MFMA intrinsic table when not explicitly set,
  // so callers do not need to know kBase × kPack to get a good VGPR estimate.
  const int kWidth = deriveKWidth(prob, cfg, hw);
  const int aBytes = (prob.aBits + 7) / 8;
  const int bBytes = (prob.bBits + 7) / 8;
  const int vgprAFrag =
      (cfg.blockM * kWidth * aBytes + ws * bytesPerVgpr - 1) /
      (ws * bytesPerVgpr);
  const int vgprBFrag =
      (cfg.blockN * kWidth * bBytes + ws * bytesPerVgpr - 1) /
      (ws * bytesPerVgpr);

  // Miscellaneous overhead: base pointers, loop induction variables, predicates
  // and a small stack frame.  28 is an empirical constant calibrated against
  // AMDGCN assembly of representative Triton GEMM kernels.
  constexpr int vgprMisc = 28;

  const int total = vgprAccum + vgprAFrag + vgprBFrag + vgprMisc;
  // Round up to the allocation granularity.
  return ((total + hw.vgprAllocGranule - 1) / hw.vgprAllocGranule) *
         hw.vgprAllocGranule;
}

// Padding helpers matching TensorAtlas's hardware.py formulas, which in turn
// mirror Triton's PaddedSharedEncodingAttr::getPaddedSize().

// [[32, 4]] padding pattern: 4 padding elements per 32-element block.
static int64_t ldspadded32x4(int64_t n) {
  int64_t p = (n >> 5) << 2;
  if ((n & 31) == 0 && p >= 4)
    p -= 4;
  return n + p;
}

// [[interval, 8]] padding pattern: 8 padding elements per interval-element
// block.  interval must be a power of 2.
static int64_t ldspaddedDim8(int64_t n, int interval) {
  int log2Interval = __builtin_ctz(static_cast<unsigned>(interval));
  int64_t p = (n >> log2Interval) << 3; // * 8
  if (n % interval == 0 && p >= 8)
    p -= 8;
  return n + p;
}

int estimateLdsBytes(const GemmProblem &prob, const TritonGemmConfig &cfg,
                     const HardwareInfo &hw) {
  if (cfg.bypassLds)
    return 0;

  const int numBuf = estimateNumBuffers(cfg);
  const int aBytes = (prob.aBits + 7) / 8;
  const int bBytes = (prob.bBits + 7) / 8;

  if (cfg.useAsyncCopy) {
    // Async-copy path uses PaddedSharedEncoding — apply TensorAtlas's formula:
    //   padded = max(_padded32x4(elem), _paddedDim8(elem, dim))
    // matching Triton's composePaddedLayoutForAsyncCopyCDNA4 intent.
    int64_t elemA = (int64_t)cfg.blockM * cfg.blockK;
    int64_t elemB = (int64_t)cfg.blockN * cfg.blockK;

    int64_t paddedA = ldspadded32x4(elemA);
    int64_t paddedB = ldspadded32x4(elemB);

    // blockK power-of-2: also check [[blockK, 8]] for A.
    if (cfg.blockK > 0 && (cfg.blockK & (cfg.blockK - 1)) == 0) {
      int64_t pa = ldspaddedDim8(elemA, cfg.blockK);
      if (pa > paddedA)
        paddedA = pa;
    }
    // blockN power-of-2: also check [[blockN, 8]] for B.
    if (cfg.blockN > 0 && (cfg.blockN & (cfg.blockN - 1)) == 0) {
      int64_t pb = ldspaddedDim8(elemB, cfg.blockN);
      if (pb > paddedB)
        paddedB = pb;
    }

    return static_cast<int>(numBuf * (paddedA * aBytes + paddedB * bBytes));
  }

  // Synchronous copy: conservative 8-element row padding.
  const int ldsA = numBuf * cfg.blockM * (cfg.blockK + 8) * aBytes;
  const int ldsB = numBuf * cfg.blockN * (cfg.blockK + 8) * bBytes;
  return ldsA + ldsB;
}

//===----------------------------------------------------------------------===//
// 4. Origami-derived cache reuse and WGM (GROUP_SIZE_M) models
//===----------------------------------------------------------------------===//
//
// These functions port Origami's predict_workgroup_mapping, compute_mall_tiles,
// compute_l2_tiles, and estimate_l2_hit directly to Triton's tile vocabulary.
// See origami/src/origami/gemm.cpp for the reference implementation.

// Origami: compute_mall_tiles
// Returns (mall_m, mall_n): number of unique M/N tiles visible to all active
// CUs sharing the MALL (or the per-XCD L2 when called with cuPerXcd).
static std::pair<int64_t, int64_t>
computeMallTiles(int64_t gridM, int64_t gridN, int64_t activeCUs, int64_t wgm) {
  if (gridM == 0 || gridN == 0 || activeCUs == 0)
    return {1, 1};
  const int64_t W         = std::max(wgm, int64_t(1));
  const int64_t slabTiles = gridM * std::min(W, gridN);
  const int64_t fullSlabs = std::min(activeCUs / std::max(slabTiles, int64_t(1)),
                                     gridN / std::min(W, gridN));
  const int64_t mallN =
      std::min(std::max((fullSlabs + 1) * std::min(W, gridN), int64_t(1)), gridN);
  const int64_t denom = std::max(mallN / std::min(W, gridN), int64_t(1)) * std::min(W, gridN);
  const int64_t mallM = std::min((activeCUs + denom - 1) / denom, gridM);
  return {std::max(mallM, int64_t(1)), std::max(mallN, int64_t(1))};
}

// Origami: compute_l2_tiles
// Returns (l2_m, l2_n): unique tiles visible within one XCD's L2, after
// capacity-shrinking to fit l2SizeBytes.
static std::pair<int64_t, int64_t>
computeL2Tiles(int64_t gridM, int64_t gridN, int64_t activeCUs,
               int64_t wgm, int64_t numXCDs, int64_t l2SizeBytes,
               int64_t blockM, int64_t blockK, int aBytes,
               int64_t blockN, int bBytes) {
  if (gridM == 0 || gridN == 0 || activeCUs == 0)
    return {1, 1};
  const int64_t cuPerXcd = std::max(activeCUs / std::max(numXCDs, int64_t(1)), int64_t(1));
  const int64_t mnPerXcd = (gridM * gridN + numXCDs - 1) / numXCDs;
  const int64_t effPerXcd = std::min(cuPerXcd, mnPerXcd);
  auto [mallM, mallN] = computeMallTiles(gridM, gridN, effPerXcd, wgm);

  // Capacity shrink: A tile = blockM × blockK bytes, B tile = blockN × blockK bytes
  const double aBytesPerTile = static_cast<double>(blockM) * blockK * aBytes;
  const double bBytesPerTile = static_cast<double>(blockN) * blockK * bBytes;
  const double l2Cap = static_cast<double>(l2SizeBytes);

  int64_t l2M = mallM, l2N = mallN;
  while (l2M * aBytesPerTile + l2N * bBytesPerTile > l2Cap && (l2M > 1 || l2N > 1)) {
    if (l2M * aBytesPerTile > l2N * bBytesPerTile && l2M > 1)
      --l2M;
    else if (l2N > 1)
      --l2N;
    else
      --l2M;
  }
  return {std::max(l2M, int64_t(1)), std::max(l2N, int64_t(1))};
}

// Origami: estimate_l2_hit
// L2 hit rate = (total_elements - unique_elements) / total_elements
// where total = uA*l2_n + uB*l2_m, unique = uA + uB,
//       uA = l2_m * blockM * blockK,  uB = l2_n * blockN * blockK.
static double estimateL2HitRate(int64_t l2M, int64_t l2N,
                                 int64_t blockM, int64_t blockK,
                                 int64_t blockN) {
  const int64_t uA    = l2M * blockM * blockK;
  const int64_t uB    = l2N * blockN * blockK;
  const int64_t total = std::max(uA * l2N + uB * l2M, int64_t(1));
  const int64_t cached = total - (uA + uB);
  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Origami: predict_workgroup_mapping (fast path)
// Selects the GROUP_SIZE_M (WGM slab width) that minimises the L2 working-set
// cost for the last XCD in the first scheduling timestep.
int selectGroupSizeM(const GemmProblem &prob, const TritonGemmConfig &cfg,
                     const HardwareInfo &hw) {
  const int64_t gridM  = (prob.M + cfg.blockM - 1) / cfg.blockM;
  const int64_t gridN  = (prob.N + cfg.blockN - 1) / cfg.blockN;
  const int64_t numMTs = gridM * gridN;
  const int64_t N_CU   = hw.numCUs;
  const int64_t numXCD = std::max(hw.numXCDs, 1);
  const int64_t cuPerXcd = N_CU / numXCD;

  // Trivial cases
  if (gridM <= 1 || gridN <= 1 || numMTs <= numXCD)
    return 1;

  // Large grids: solution is insensitive → use sqrt(N_CU/numXCD)
  const int64_t gridThreshold = static_cast<int64_t>(std::sqrt(static_cast<double>(N_CU)));
  if (gridM > gridThreshold && gridN > gridThreshold)
    return static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N_CU / numXCD))));

  const int64_t wgsPerXcd = std::min((numMTs + numXCD - 1) / numXCD, cuPerXcd);

  // Enough work + small N → use grid_n directly
  if (wgsPerXcd >= cuPerXcd / 2 && gridN <= 8)
    return static_cast<int>(gridN);

  // Build candidate set: {1, 4, 6} ∪ divisors(wgm_cap)
  const int64_t wgmCap = std::min(gridN, wgsPerXcd / 2);
  if (wgmCap <= 0)
    return 1;

  // Use bitmask for candidates (Origami approach; capped at 64)
  uint64_t cmask = 0;
  for (int64_t v : {int64_t(1), int64_t(4), int64_t(6)})
    if (v <= wgmCap && v < 64)
      cmask |= (1ULL << v);
  for (int64_t i = 1; i * i <= wgmCap && i < 64; ++i) {
    if (wgmCap % i == 0) {
      cmask |= (1ULL << i);
      if (wgmCap / i < 64)
        cmask |= (1ULL << (wgmCap / i));
    }
  }

  // Evaluate L2 working-set cost for last XCD in first timestep
  const int64_t total        = numMTs;
  const int64_t lastXcd      = numXCD - 2; // Origami uses NUM_XCD-2
  const int64_t groupSize    = total >= numXCD ? total / numXCD : total;
  const int64_t tileThisXcd  = std::min(cuPerXcd, groupSize);
  const int64_t start        = lastXcd * groupSize;
  const int64_t count        = (start < total)
                                   ? std::min(tileThisXcd, total - start)
                                   : int64_t(0);

  const double aCost = static_cast<double>(cfg.blockM) * ((prob.aBits + 7) / 8);
  const double bCost = static_cast<double>(cfg.blockN) * ((prob.bBits + 7) / 8);

  int   bestWgm  = 1;
  double bestCost = std::numeric_limits<double>::max();

  for (uint64_t m = cmask; m; m &= m - 1) {
    const int64_t wgm       = static_cast<int64_t>(__builtin_ctzll(m));
    const int64_t slabTiles = gridM * wgm;
    const int64_t firstSlab = start / slabTiles;
    const int64_t lastSlab  = count > 0 ? (start + count - 1) / slabTiles : firstSlab;
    const int64_t firstRow  = (start % slabTiles) / wgm;
    const int64_t lastRow   = count > 0
                                  ? ((start + count - 1) % slabTiles) / wgm
                                  : firstRow;

    int64_t uniqueRows, uniqueCols;
    if (firstSlab == lastSlab) {
      uniqueRows = lastRow - firstRow + 1;
      uniqueCols = (uniqueRows > 1) ? wgm
                                    : std::min(count, wgm);
    } else {
      uniqueRows = (lastSlab - firstSlab > 1)
                       ? gridM
                       : std::min(gridM, (gridM - firstRow) + (lastRow + 1));
      uniqueCols = std::min((lastSlab - firstSlab + 1) * wgm, gridN);
    }
    uniqueRows = std::min(uniqueRows, gridM);
    uniqueCols = std::min(uniqueCols, gridN);

    const double cost = uniqueRows * aCost + uniqueCols * bCost;
    if (cost < bestCost) {
      bestCost = cost;
      bestWgm  = static_cast<int>(wgm);
    }
  }
  return bestWgm;
}

//===----------------------------------------------------------------------===//
// 5. Full performance estimate
//===----------------------------------------------------------------------===//

PerfEstimate estimatePerf(const GemmProblem &prob, const TritonGemmConfig &cfg,
                          const HardwareInfo &hw) {
  PerfEstimate est;

  if (hw.arch == Arch::Unknown || hw.numCUs == 0) {
    est.isValid = false;
    return est;
  }

  // ── Step 1: Resource accounting ──────────────────────────────────────────

  est.numBuffers = estimateNumBuffers(cfg);
  est.vgprCount  = estimateVgpr(prob, cfg, hw);
  est.ldsBytes   = estimateLdsBytes(prob, cfg, hw);

  // LDS feasibility check.
  est.ldsExceeded = (est.ldsBytes > hw.ldsPerCU);

  // VGPR-limited waves per SIMD.
  est.wavesPerSimd = std::min(hw.maxWavesPerSimd,
                              hw.vgprPerSimd / std::max(1, est.vgprCount));
  // Heuristic spill threshold: if < 1 wave fits, we hard-spill.
  // Light spill threshold scales with the VGPR file size: 75% of vgprPerSimd.
  // On CDNA3 (256 VGPRs) this is ~192; on CDNA4 (512 VGPRs) this is ~384.
  const int spillThreshold = (hw.vgprPerSimd * 3) / 4;
  est.likelySpills = (est.vgprCount > hw.vgprPerSimd) ||
                     (est.wavesPerSimd == 1 && est.vgprCount > spillThreshold);

  // CTA-limited occupancy from LDS.
  int ctasFromLds = (est.ldsBytes > 0)
                        ? std::max(1, hw.ldsPerCU / est.ldsBytes)
                        : hw.maxWavesPerSimd; // no LDS → not LDS-limited
  int wavesFromLds = ctasFromLds * cfg.numWarps;

  // Effective waves per CU (min of VGPR and LDS constraints).
  int maxWavesPerCU = hw.numSimdPerCU * hw.maxWavesPerSimd;
  int wavesFromVgpr = est.wavesPerSimd * hw.numSimdPerCU;
  int wavesPerCU = std::min({wavesFromVgpr, wavesFromLds, maxWavesPerCU});

  est.ctasPerCU = wavesPerCU / std::max(1, cfg.numWarps);
  est.occupancy = static_cast<double>(wavesPerCU) / maxWavesPerCU;

  // For the occupancy penalty in the roofline, use VGPR-limited occupancy only.
  // LDS-limited occupancy (from larger pipeline buffers) should not be penalised:
  // more LDS per CTA improves compute-memory overlap rather than hurting throughput.
  // This allows the BK tiebreak to correctly prefer BK=64 over BK=32 even when
  // BK=64 uses more LDS (lower LDS occupancy) but achieves the same pipeline depth.
  const double vgprOccupancy =
      static_cast<double>(std::min(wavesFromVgpr, maxWavesPerCU)) / maxWavesPerCU;

  // ── Step 2: Wave quantisation ─────────────────────────────────────────────

  auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

  est.totalOutputTiles =
      ceildiv(prob.M, cfg.blockM) * ceildiv(prob.N, cfg.blockN) * prob.batchSize;

  // Each CTA covers one output tile (no Stream-K in Triton by default).
  est.numWaves =
      static_cast<int>(ceildiv(est.totalOutputTiles, hw.numCUs));

  // Tail-wave efficiency: if outputTiles is not a multiple of numCUs the last
  // wave runs with fewer active CUs.
  // Special case: when totalOutputTiles < numCUs, only that many CUs are
  // active — the rest are idle. Model this as waveEfficiency < 1.
  int64_t fullWaveCUs = est.totalOutputTiles % hw.numCUs;
  est.waveEfficiency =
      (fullWaveCUs == 0)
          ? 1.0
          : (static_cast<double>(est.totalOutputTiles) /
             (static_cast<double>(est.numWaves) * hw.numCUs));

  // ── Step 3: Roofline across the full K reduction ──────────────────────────
  //
  // A critical fix vs a naive per-tile roofline: each output tile requires
  // ceil(K / BLOCK_K) K-loop iterations.  A small BLOCK_K processes fewer
  // elements per iteration but runs many more iterations, making it slower
  // than a large BLOCK_K despite lower per-iteration cost.  This matches
  // Origami's num_iter term in compute_total_latency().
  //
  //   numKIter          = K / BLOCK_K  (exact, not ceiling)
  //   numMfmaPerKBlock  = (BLOCK_M/mDim) * (BLOCK_N/nDim) * (BLOCK_K/kDim)
  //   computeCycles     = numMfmaPerKBlock * numKIter * throughputCycles
  //                       / numSimdPerCU
  //
  // We use exact (floating-point) K/BLOCK_K rather than ceil(K/BLOCK_K) for
  // numKIter. When K is not divisible by BLOCK_K (e.g. K=2880, BK=128),
  // ceil gives 23 but the effective work is 22.5 iterations. Using ceil
  // unfairly penalises larger BK values in ranking: BK=128 appears to do
  // 23/22.5 = 2.2% more work than BK=64 (45 exact iterations), making the
  // model prefer BK=64. With exact division both configurations perform the
  // same total work and rank equally, so the BK tiebreak correctly selects
  // the larger value. (Triton itself runs ceil iterations with K-masking for
  // the partial last block — this is a ranking approximation, not a correctness
  // issue.)

  int mDim = cfg.mfmaNonKDim > 0
                 ? cfg.mfmaNonKDim
                 : selectMfmaNonKDim(prob, cfg, hw);

  const double numKIter =
      (cfg.blockK > 0) ? static_cast<double>(prob.K) / cfg.blockK : 1.0;

  auto infoOpt = getMfmaInstrInfo(hw.arch, mDim, mDim, prob.aKind, prob.cKind);
  if (!infoOpt) {
    infoOpt = getMfmaInstrInfo(hw.arch, 16, 16, ElemKind::FP16, ElemKind::FP32);
  }

  if (infoOpt) {
    const MfmaInstrInfo &info = *infoOpt;
    int64_t numMfmaPerKBlock =
        static_cast<int64_t>(ceildiv(cfg.blockM, info.mDim)) *
        ceildiv(cfg.blockN, info.nDim) *
        ceildiv(cfg.blockK, info.kDim);
    est.computeCycles =
        static_cast<double>(numMfmaPerKBlock) * numKIter
        * info.throughputCycles / hw.numSimdPerCU;
  } else {
    // Fallback: full GEMM FLOPs for this output tile.
    est.computeCycles =
        (2.0 * cfg.blockM * cfg.blockN * prob.K) / hw.peakMfmaFlopsPerCycleCU();
  }

  // Memory cycles: total A/B traffic across all K iterations.
  const int aBytes = (prob.aBits + 7) / 8;
  const int bBytes = (prob.bBits + 7) / 8;
  const int cBytes = (prob.cBits + 7) / 8;

  // Per K-block fetch (one stage worth of A+B).
  double tileBytesABperK =
      static_cast<double>(cfg.blockM * cfg.blockK * aBytes +
                          cfg.blockN * cfg.blockK * bBytes);
  // Total A/B traffic: numKIter fetches of the K-block (exact, not ceil).
  double tileBytesAB = tileBytesABperK * numKIter;
  double tileBytesC  = static_cast<double>(cfg.blockM * cfg.blockN * cBytes);
  double tileBytesTotal = tileBytesAB + tileBytesC;

  // Memory cycles: model cache hierarchy using Origami's L2 hit-rate formula.
  // L2 hit rate reduces effective DRAM traffic; hits are served at L2 bandwidth.
  const int64_t gridM = ceildiv(prob.M, cfg.blockM);
  const int64_t gridN = ceildiv(prob.N, cfg.blockN);
  const int wgm = (cfg.groupSizeM > 0) ? cfg.groupSizeM
                                        : selectGroupSizeM(prob, cfg, hw);

  double l2HitRate = 0.0;
  if (hw.l2SizeBytes > 0 && hw.numCUs > 0 && gridM > 0 && gridN > 0) {
    auto [l2M, l2N] = computeL2Tiles(
        gridM, gridN, hw.numCUs, wgm, hw.numXCDs, hw.l2SizeBytes,
        cfg.blockM, cfg.blockK, aBytes, cfg.blockN, bBytes);
    l2HitRate = estimateL2HitRate(l2M, l2N, cfg.blockM, cfg.blockK, cfg.blockN);
  }

  // Effective bandwidth: DRAM for misses, L2 for hits.
  // When peakL2BwBytesPerCycle == 0 (uncalibrated), L2 hits are treated as
  // free (served at DRAM bandwidth) — equivalent to disabling the L2 model.
  // Set peakL2BwBytesPerCycle after calibration to enable accurate L2 reuse.
  const double dramBwPerCU = hw.peakMemBwBytesPerCycle / std::max(hw.numCUs, 1);
  double memCycles;
  if (hw.peakL2BwBytesPerCycle > 0.0) {
    const double l2BwPerCU   = hw.peakL2BwBytesPerCycle / std::max(hw.numCUs, 1);
    const double dramTraffic = tileBytesAB * (1.0 - l2HitRate);
    const double l2Traffic   = tileBytesAB * l2HitRate;
    memCycles = (dramBwPerCU > 0.0 ? dramTraffic / dramBwPerCU : 0.0) +
                (l2BwPerCU   > 0.0 ? l2Traffic   / l2BwPerCU   : 0.0);
  } else {
    // Uncalibrated: use DRAM bandwidth for all traffic (conservative baseline).
    memCycles = (dramBwPerCU > 0.0) ? tileBytesAB / dramBwPerCU : 0.0;
  }
  est.memoryCycles = memCycles;

  // ── Step 4: Software-pipeline overlap ─────────────────────────────────────
  //
  // With numStages software pipeline stages the compiler issues global-memory
  // reads numStages−1 iterations ahead of the compute that consumes them.
  // When there are enough waves (occupancy) to fill the pipeline depth the
  // memory latency becomes fully hidden.
  //
  // Simple model:
  //   overlap = min(1, (numStages - 1) / pipelineDepthNeeded)
  // where pipelineDepthNeeded ≈ memoryCycles / computeCycles.
  //
  // If compute already dominates (memoryCycles < computeCycles) there is
  // nothing to hide and overlap is irrelevant.
  double depthNeeded = (est.computeCycles > 0.0)
                           ? est.memoryCycles / est.computeCycles
                           : 1.0;
  double stageFactor = (depthNeeded > 0.0)
                           ? static_cast<double>(cfg.numStages - 1) / depthNeeded
                           : 1.0;
  est.pipelineOverlap = std::min(1.0, std::max(0.0, stageFactor));

  // Effective tile cycles: compute dominates, memory visible only to the
  // extent it is not hidden by pipelining.
  double hiddenMemCycles = est.memoryCycles * est.pipelineOverlap;
  est.effectiveTileCycles =
      std::max(est.computeCycles, est.memoryCycles - hiddenMemCycles);

  // isComputeBound based on EFFECTIVE cycles (after pipeline overlap), not raw.
  // A kernel pipelined enough to hide memory latency is compute-bound even if
  // raw memoryCycles > computeCycles.
  est.isComputeBound = (est.effectiveTileCycles <= est.computeCycles * 1.05);

  // ── Step 5: Predicted throughput ──────────────────────────────────────────
  //
  // Total cycles = effectiveTileCycles × numWaves / waveEfficiency
  //
  // Occupancy penalty: low occupancy limits the GPU's ability to hide
  // instruction-level and memory latency with other wavefronts.  However,
  // when the kernel is compute-bound (MFMA pipeline is the bottleneck),
  // occupancy does not limit throughput — the MFMA units are saturated
  // regardless of how many other waves are resident.
  //
  // Occupancy penalty: for compute-bound kernels, no penalty (MFMA saturated).
  // For memory-bound kernels, use VGPR-only occupancy (vgprOccupancy) rather
  // than the combined est.occupancy. LDS-limited occupancy reflects larger
  // pipeline buffers — beneficial for overlap, not a performance constraint.
  // Cap at 2× (min effective occupancy = 0.5) to avoid over-penalising.
  double occupancyPenalty = est.isComputeBound
                                ? 1.0
                                : (1.0 / std::max(vgprOccupancy, 0.5));

  double totalCycles = est.effectiveTileCycles * est.numWaves * occupancyPenalty;
  // Apply wave-tail efficiency.
  totalCycles /= std::max(1e-6, est.waveEfficiency);

  double totalFlops = 2.0 * prob.M * prob.N * prob.K * prob.batchSize;

  // TFLOPS = totalFlops / (totalCycles / clockMHz * 1e6) / 1e12
  if (totalCycles > 0.0 && hw.clockMHz > 0.0) {
    double totalSeconds = totalCycles / (hw.clockMHz * 1e6);
    est.predictedTflops = totalFlops / totalSeconds / 1e12;
  }

  // Arithmetic intensity: FLOPs per byte of A+B traffic (per K-block, consistent
  // with the numerator using cfg.blockK rather than prob.K).
  est.arithmeticIntensity =
      (tileBytesABperK > 0.0)
          ? (2.0 * cfg.blockM * cfg.blockN * cfg.blockK) / tileBytesABperK
          : 0.0;

  // ── Validity ──────────────────────────────────────────────────────────────
  est.isValid = !est.ldsExceeded &&
                !est.likelySpills &&
                (cfg.numWarps > 0) &&
                ((cfg.numWarps & (cfg.numWarps - 1)) == 0) && // power of two
                (cfg.blockK > 0) && (cfg.blockM > 0) && (cfg.blockN > 0);

  return est;
}

//===----------------------------------------------------------------------===//
// 5. Config validation and ranking
//===----------------------------------------------------------------------===//

bool isValidConfig(const GemmProblem &prob, const TritonGemmConfig &cfg,
                   const HardwareInfo &hw) {
  if (cfg.numWarps <= 0 || (cfg.numWarps & (cfg.numWarps - 1)) != 0)
    return false;
  if (cfg.blockM <= 0 || cfg.blockN <= 0 || cfg.blockK <= 0)
    return false;

  const int ldsBytes = estimateLdsBytes(prob, cfg, hw);
  if (ldsBytes > hw.ldsPerCU)
    return false;

  const int vgpr = estimateVgpr(prob, cfg, hw);
  if (vgpr > hw.vgprPerSimd) // hard spill
    return false;

  // blockK must be divisible by the chosen MFMA kDim so that K iterations
  // are integer multiples.
  int mDim = cfg.mfmaNonKDim > 0 ? cfg.mfmaNonKDim
                                  : selectMfmaNonKDim(prob, cfg, hw);
  auto infoOpt =
      getMfmaInstrInfo(hw.arch, mDim, mDim, prob.aKind, prob.cKind);
  if (infoOpt && (cfg.blockK % infoOpt->kDim) != 0)
    return false;

  return true;
}

std::vector<TritonGemmConfig>
rankConfigs(const GemmProblem &prob, llvm::ArrayRef<TritonGemmConfig> configs,
            const HardwareInfo &hw, size_t topK) {
  // Optimization 1: store index + estimate, not a copy of TritonGemmConfig.
  // Sorting moves ScoredIdx (one size_t + PerfEstimate) rather than copying
  // the full TritonGemmConfig struct. Matches Origami's reference_wrapper
  // technique.
  struct ScoredIdx {
    size_t idx;
    PerfEstimate est;
  };

  std::vector<ScoredIdx> scored;
  scored.reserve(configs.size());

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto &cfg = configs[i];

    // Optimization 2: LDS pre-filter — check the cheap constraint first and
    // skip the expensive roofline computation for configs that cannot fit.
    // Mirrors Origami's check_lds_capacity pre-filter in rank_configs().
    if (estimateLdsBytes(prob, cfg, hw) > hw.ldsPerCU)
      continue;

    scored.push_back({i, estimatePerf(prob, cfg, hw)});
  }

  // Comparator: higher TFLOPS first; tie-break sequence:
  //  1. arithmetic intensity (larger tile → better cache reuse)
  //  2. blockK (larger → fewer K-loop iterations, better pipeline fill)
  //  3. numWarps (larger → enables pingpong scheduling on CDNA3/4)
  //  4. blockM (Origami convention)
  auto cmp = [&](const ScoredIdx &a, const ScoredIdx &b) {
    if (a.est.isValid != b.est.isValid)
      return a.est.isValid > b.est.isValid;
    if (std::abs(a.est.predictedTflops - b.est.predictedTflops) > 1e-3)
      return a.est.predictedTflops > b.est.predictedTflops;
    if (std::abs(a.est.arithmeticIntensity - b.est.arithmeticIntensity) > 1e-3)
      return a.est.arithmeticIntensity > b.est.arithmeticIntensity;
    if (configs[a.idx].blockK != configs[b.idx].blockK)
      return configs[a.idx].blockK > configs[b.idx].blockK;
    if (configs[a.idx].numWarps != configs[b.idx].numWarps)
      return configs[a.idx].numWarps > configs[b.idx].numWarps;
    // Prefer fewer stages: simpler pipeline with less LDS/VGPR pressure.
    // Autotune consistently picks nS=2 on gfx950; nS=3 rarely wins in practice.
    if (configs[a.idx].numStages != configs[b.idx].numStages)
      return configs[a.idx].numStages < configs[b.idx].numStages;
    return configs[a.idx].blockM > configs[b.idx].blockM;
  };

  // Optimization 3: use partial_sort when only top-K results are needed.
  // O(N log K) vs O(N log N) — ~7× faster for N=924, K=5.
  const size_t k =
      (topK == 0 || topK >= scored.size()) ? scored.size() : topK;
  if (k < scored.size())
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(), cmp);
  else
    std::stable_sort(scored.begin(), scored.end(), cmp);

  std::vector<TritonGemmConfig> result;
  result.reserve(k);
  for (size_t i = 0; i < k; ++i)
    result.push_back(configs[scored[i].idx]);
  return result;
}

//===----------------------------------------------------------------------===//
// 8. Candidate config generation
//===----------------------------------------------------------------------===//

std::vector<TritonGemmConfig>
generateCandidates(const GemmProblem &prob, const HardwareInfo &hw) {
  // Step 1: pick the MFMA instruction dimension using the throughput model.
  // Use a probe config (blockM=blockN=128) to let selectMfmaNonKDim run.
  // The result is arch+dtype specific and independent of tile size.
  TritonGemmConfig probe;
  probe.blockM = 128;
  probe.blockN = 128;
  probe.blockK = 32;
  probe.mfmaNonKDim = 0; // let model choose
  const int mfmaDim = selectMfmaNonKDim(prob, probe, hw);

  // Step 2: determine valid blockK values — multiples of the MFMA kDim.
  // Look up kDim from the throughput table for this arch+dtype combination.
  int mfmaKDim = 16; // safe default
  if (auto info = getMfmaInstrInfo(hw.arch, mfmaDim, mfmaDim,
                                   prob.aKind, prob.cKind))
    mfmaKDim = info->kDim;

  // blockK candidates: 1×, 2×, 4× the MFMA kDim.
  // Cap at 4×kDim: from empirical calibration (testtmp/calibrate_l2_bw.py),
  // BK=8×kDim (=256 for gfx950) is fastest for M≤1024 (fits in L2) but
  // drops for M≥2048 when the working set exceeds L2 per XCD. Since we
  // don't yet model this M-dependent L2 capacity constraint, conservatively
  // cap at 4×kDim (=128 for gfx950) which performs well across all M.
  const int blockKCandidates[] = {
      mfmaKDim,
      mfmaKDim * 2,
      mfmaKDim * 4,
  };

  // Step 3: generate (blockM, blockN) pairs using Origami's wave-based formula:
  //   blockM = mfmaDim × waveTileM × waveCountM
  //   blockN = mfmaDim × waveTileN × waveCountN
  // waveTile sweeps {1, 2, 4}, waveCount sweeps {1, 2, 4}.
  // minTile = 2 × mfmaDim: single-instruction tiles (blockM == mfmaDim) are
  // too small — kernel-launch overhead dominates and MFMA utilisation is poor.
  const int waveTiles[]  = {1, 2, 4};
  const int waveCounts[] = {1, 2, 4};
  const int maxTile = 256;
  const int minTile = 2 * mfmaDim; // e.g. 32 for mfmaDim=16

  std::vector<std::pair<int, int>> mnPairs;
  for (int wtM : waveTiles) {
    for (int wcM : waveCounts) {
      int bM = mfmaDim * wtM * wcM;
      if (bM < minTile || bM > maxTile)
        continue;
      for (int wtN : waveTiles) {
        for (int wcN : waveCounts) {
          int bN = mfmaDim * wtN * wcN;
          if (bN < minTile || bN > maxTile)
            continue;
          mnPairs.emplace_back(bM, bN);
        }
      }
    }
  }
  // Deduplicate (blockM, blockN) pairs.
  std::sort(mnPairs.begin(), mnPairs.end());
  mnPairs.erase(std::unique(mnPairs.begin(), mnPairs.end()), mnPairs.end());

  // Step 4: sweep numWarps and numStages.
  // On CDNA4 (gfx950), Triton AMD GEMM kernels use num_warps=8 exclusively:
  // the pingpong scheduler splits 8 warps into compute and memory clusters.
  // Using fewer warps disables pingpong and misses the key gfx950 optimization.
  // For CDNA1-3 and RDNA we sweep the full range.
  // On CDNA4 (gfx950): num_warps=8 required for pingpong; num_stages=2
  // consistently wins across all problem sizes (empirically validated vs
  // autotuner). Use a single value to avoid model picking nS=3 due to
  // over-optimistic pipeline overlap estimates.
  // On other archs: sweep the full range.
  const bool cdna4Only = (hw.arch == Arch::CDNA4);
  const int numWarpsCandidates[]  = {4, 8};
  const int numStagesCandidates[] = {1, 2, 3};

  // Step 5: enumerate all combinations and keep feasible ones.
  // Reserve the theoretical upper bound to avoid reallocation.
  std::vector<TritonGemmConfig> candidates;
  candidates.reserve(mnPairs.size() * std::size(blockKCandidates) *
                     std::size(numWarpsCandidates) *
                     std::size(numStagesCandidates));
  for (auto [bM, bN] : mnPairs) {
    for (int bK : blockKCandidates) {
      if (bK > 256)
        continue;
      for (int nW : numWarpsCandidates) {
        if (cdna4Only && nW != 8) // gfx950 requires nW=8 for pingpong
          continue;
        for (int nS : numStagesCandidates) {
          if (cdna4Only && nS != 2) // gfx950: nS=2 wins empirically
            continue;
          TritonGemmConfig cfg;
          cfg.blockM      = bM;
          cfg.blockN      = bN;
          cfg.blockK      = bK;
          cfg.numWarps    = nW;
          cfg.numStages   = nS;
          cfg.mfmaNonKDim = mfmaDim;
          cfg.kWidth      = 0;         // let estimateVgpr derive it
          cfg.bypassLds   = false;
          cfg.useAsyncCopy = true;
          cfg.kPack       = 1;
          cfg.wavesPerEu  = 0;
          cfg.groupSizeM  = selectGroupSizeM(prob, cfg, hw);

          if (isValidConfig(prob, cfg, hw))
            candidates.push_back(cfg);
        }
      }
    }
  }
  return candidates;
}

} // namespace mlir::triton::AMD::perf
