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
    hw.ldsPerCU = 65536;        // 64 KB
    hw.l2SizeBytes = 8 << 20;   // 8 MB
    hw.mallSizeBytes = 0;
    hw.clockMHz = 1500.0;
    // Peak BW: ~1.2 TB/s  →  1.2e12 / (1.5e9) ≈ 800 bytes/cycle
    hw.peakMemBwBytesPerCycle = 800.0;
    break;

  // ── CDNA2  gfx90a  MI200 (MI210 / MI250) ────────────────────────────────
  case Arch::CDNA2:
    hw.numCUs = 104;           // MI210; MI250X has 110 CUs
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 256;
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 65536;        // 64 KB
    hw.l2SizeBytes = 8 << 20;   // 8 MB
    hw.mallSizeBytes = 0;
    hw.clockMHz = 1700.0;
    // Peak BW: ~1.6 TB/s  →  ≈ 941 bytes/cycle
    hw.peakMemBwBytesPerCycle = 941.0;
    break;

  // ── CDNA3  gfx940/941/942  MI300 ────────────────────────────────────────
  case Arch::CDNA3:
    hw.numCUs = 228;           // MI300X; MI300A has 228 CUs as well
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 256;
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 65536;        // 64 KB
    hw.l2SizeBytes = 256 << 20; // 256 MB (HBM3, shared across 8 XCDs)
    hw.mallSizeBytes = 0;
    hw.clockMHz = 2100.0;
    // Peak BW: ~5.3 TB/s for MI300X  →  ≈ 2524 bytes/cycle
    hw.peakMemBwBytesPerCycle = 2524.0;
    break;

  // ── CDNA4  gfx950  MI350 ────────────────────────────────────────────────
  case Arch::CDNA4:
    hw.numCUs = 256;
    hw.numSimdPerCU = 4;
    hw.waveSize = 64;
    hw.vgprPerSimd = 512;      // Doubled VGPR file vs CDNA3
    hw.vgprAllocGranule = 4;
    hw.maxWavesPerSimd = 10;
    hw.ldsPerCU = 163840;       // 160 KB (from TargetInfo.cpp)
    hw.l2SizeBytes = 256 << 20;
    hw.mallSizeBytes = 0;
    hw.clockMHz = 2400.0;
    hw.peakMemBwBytesPerCycle = 3000.0;
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
  // Same instruction latencies as CDNA3, but 512 VGPRs per SIMD allows higher
  // occupancy.  Scaled MFMA (F8F6F4) adds new instruction shapes.
  {Arch::CDNA4, 32, 32,  8, 64, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA4, 16, 16, 16, 32, ElemKind::FP16,  ElemKind::FP32},
  {Arch::CDNA4, 32, 32,  4, 64, ElemKind::BF16,  ElemKind::FP32},
  {Arch::CDNA4, 16, 16,  8, 32, ElemKind::BF16,  ElemKind::FP32},
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
  // The ratio kDim / mDim encodes the relationship documented in
  // AccelerateAMDMatmul.cpp: larger mDim → fewer k-elements per thread.
  const int kBase = std::max(1, infoOpt->kDim / mDim);
  return kBase * std::max(1, cfg.kPack);
}

int estimateVgpr(const GemmProblem &prob, const TritonGemmConfig &cfg,
                 const HardwareInfo &hw) {
  // Each VGPR holds 4 bytes (32-bit register).
  constexpr int bytesPerVgpr = 4;
  const int ws = hw.waveSize;

  // Accumulator tile: BLOCK_M × BLOCK_N elements of the C type, distributed
  // across all lanes in the wavefront.
  const int cBytes = (prob.cBits + 7) / 8;
  const int vgprAccum =
      (cfg.blockM * cfg.blockN * cBytes + ws * bytesPerVgpr - 1) /
      (ws * bytesPerVgpr);

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

int estimateLdsBytes(const GemmProblem &prob, const TritonGemmConfig &cfg,
                     const HardwareInfo &hw) {
  if (cfg.bypassLds)
    return 0;

  const int numBuf = estimateNumBuffers(cfg);
  const int aBytes = (prob.aBits + 7) / 8;
  const int bBytes = (prob.bBits + 7) / 8;

  // 8-element padding per row avoids the most common LDS bank-conflict pattern
  // without requiring architecture-specific alignment analysis.
  // (AccelerateAMDMatmul.cpp::composePaddedLayout handles the exact version.)
  constexpr int paddingElems = 8;

  const int ldsA =
      numBuf * cfg.blockM * (cfg.blockK + paddingElems) * aBytes;
  const int ldsB =
      numBuf * cfg.blockN * (cfg.blockK + paddingElems) * bBytes;

  return ldsA + ldsB;
}

//===----------------------------------------------------------------------===//
// 4. Full performance estimate
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
  // If waves_per_simd is 1 and vgprCount > 192, light spilling is likely.
  est.likelySpills = (est.vgprCount > hw.vgprPerSimd) ||
                     (est.wavesPerSimd == 1 && est.vgprCount > 192);

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

  // ── Step 2: Wave quantisation ─────────────────────────────────────────────

  auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

  est.totalOutputTiles =
      ceildiv(prob.M, cfg.blockM) * ceildiv(prob.N, cfg.blockN) * prob.batchSize;

  // Each CTA covers one output tile (no Stream-K in Triton by default).
  est.numWaves =
      static_cast<int>(ceildiv(est.totalOutputTiles, hw.numCUs));

  // Tail-wave efficiency: if outputTiles is not a multiple of numCUs the last
  // wave runs with fewer active CUs.
  int64_t fullWaveCUs = est.totalOutputTiles % hw.numCUs;
  est.waveEfficiency =
      (fullWaveCUs == 0)
          ? 1.0
          : (static_cast<double>(est.totalOutputTiles) /
             (static_cast<double>(est.numWaves) * hw.numCUs));

  // ── Step 3: Roofline per tile ─────────────────────────────────────────────

  // Compute cycles: how many MFMA cycles does one output tile require on a
  // single CU?
  //
  //   numMfma = (BLOCK_M/mDim) * (BLOCK_N/nDim) * (BLOCK_K/kDim)
  //   computeCycles = numMfma * throughputCycles / numSimdPerCU
  //
  // We resolve the chosen mDim/nDim from cfg or fall back to the model's
  // selection.  If no matching intrinsic exists, use a synthetic estimate.
  int mDim = cfg.mfmaNonKDim > 0
                 ? cfg.mfmaNonKDim
                 : selectMfmaNonKDim(prob, cfg, hw);

  auto infoOpt = getMfmaInstrInfo(hw.arch, mDim, mDim, prob.aKind, prob.cKind);
  if (!infoOpt) {
    // Fall back to 16x16 FP16→FP32 as a representative entry.
    infoOpt = getMfmaInstrInfo(hw.arch, 16, 16, ElemKind::FP16, ElemKind::FP32);
  }

  if (infoOpt) {
    const MfmaInstrInfo &info = *infoOpt;
    // Number of MFMA instructions to cover one output tile.
    int64_t numMfma = static_cast<int64_t>(ceildiv(cfg.blockM, info.mDim)) *
                      ceildiv(cfg.blockN, info.nDim) *
                      ceildiv(cfg.blockK, info.kDim);
    // All SIMDs in the CU work in parallel (one wave per SIMD for the warp
    // group), so divide by numSimdPerCU.
    est.computeCycles =
        static_cast<double>(numMfma * info.throughputCycles) / hw.numSimdPerCU;
  } else {
    // Synthetic fallback: derive from peak FLOP rate.
    double tileFlops =
        2.0 * cfg.blockM * cfg.blockN * cfg.blockK;
    est.computeCycles = tileFlops / hw.peakMfmaFlopsPerCycleCU();
  }

  // Memory cycles: time to stream A and B tiles from global memory.
  // We attribute all A/B reads to global memory (worst case; L2/MALL hits
  // improve this in practice but require knowledge of the access pattern).
  const int aBytes = (prob.aBits + 7) / 8;
  const int bBytes = (prob.bBits + 7) / 8;
  const int cBytes = (prob.cBits + 7) / 8;

  double tileBytesAB =
      static_cast<double>(cfg.blockM * cfg.blockK * aBytes +
                          cfg.blockN * cfg.blockK * bBytes);
  double tileBytesC =
      static_cast<double>(cfg.blockM * cfg.blockN * cBytes);
  double tileBytesTotal = tileBytesAB + tileBytesC;

  // Each CU receives an equal share of total memory bandwidth.
  double bwPerCU = hw.peakMemBwBytesPerCycle / hw.numCUs;
  est.memoryCycles = (bwPerCU > 0.0) ? (tileBytesAB / bwPerCU) : 0.0;

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

  est.isComputeBound = (est.computeCycles >= est.memoryCycles);

  // ── Step 5: Predicted throughput ──────────────────────────────────────────

  // Total cycles for the whole GEMM:
  //   effectiveTileCycles × numWaves  (wave quantisation)
  // scaled by occupancy inefficiency.
  double totalCycles =
      est.effectiveTileCycles * est.numWaves / std::max(1e-6, est.occupancy);
  // Apply wave-tail efficiency.
  totalCycles /= std::max(1e-6, est.waveEfficiency);

  double totalFlops =
      2.0 * prob.M * prob.N * prob.K * prob.batchSize;

  // TFLOPS = totalFlops / (totalCycles / clockMHz * 1e6) / 1e12
  if (totalCycles > 0.0 && hw.clockMHz > 0.0) {
    double totalSeconds = totalCycles / (hw.clockMHz * 1e6);
    est.predictedTflops = totalFlops / totalSeconds / 1e12;
  }

  // Arithmetic intensity (compute / memory roof of the roofline).
  est.arithmeticIntensity =
      (tileBytesTotal > 0.0)
          ? (2.0 * cfg.blockM * cfg.blockN * cfg.blockK) / tileBytesTotal
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
            const HardwareInfo &hw) {
  struct ScoredConfig {
    TritonGemmConfig cfg;
    PerfEstimate est;
  };

  std::vector<ScoredConfig> scored;
  scored.reserve(configs.size());
  for (const auto &cfg : configs)
    scored.push_back({cfg, estimatePerf(prob, cfg, hw)});

  std::stable_sort(scored.begin(), scored.end(),
                   [](const ScoredConfig &a, const ScoredConfig &b) {
    // Invalid configs go last.
    if (a.est.isValid != b.est.isValid)
      return a.est.isValid > b.est.isValid;
    // Among valid configs, prefer higher predicted TFLOPS.
    if (std::abs(a.est.predictedTflops - b.est.predictedTflops) > 1e-3)
      return a.est.predictedTflops > b.est.predictedTflops;
    // Tie-break 1: higher arithmetic intensity (better compute efficiency).
    if (std::abs(a.est.arithmeticIntensity - b.est.arithmeticIntensity) > 1e-3)
      return a.est.arithmeticIntensity > b.est.arithmeticIntensity;
    // Tie-break 2: larger blockM (mirrors Origami's convention).
    return a.cfg.blockM > b.cfg.blockM;
  });

  std::vector<TritonGemmConfig> result;
  result.reserve(scored.size());
  for (const auto &s : scored)
    result.push_back(s.cfg);
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

  // blockK candidates: 1×, 2×, 4×, 8× the MFMA kDim, capped at 256.
  // Larger blockK improves arithmetic intensity but inflates LDS.
  const int blockKCandidates[] = {
      mfmaKDim,
      mfmaKDim * 2,
      mfmaKDim * 4,
      mfmaKDim * 8,
  };

  // Step 3: generate (blockM, blockN) pairs using Origami's wave-based formula:
  //   blockM = mfmaDim × waveTileM × waveCountM
  //   blockN = mfmaDim × waveTileN × waveCountN
  // waveTile sweeps {1, 2, 4}, waveCount sweeps {1, 2, 4}.
  // Cap at 256 to avoid unrealistically large tiles.
  const int waveTiles[]  = {1, 2, 4};
  const int waveCounts[] = {1, 2, 4};
  const int maxTile = 256;

  std::vector<std::pair<int, int>> mnPairs;
  for (int wtM : waveTiles) {
    for (int wcM : waveCounts) {
      int bM = mfmaDim * wtM * wcM;
      if (bM < mfmaDim || bM > maxTile)
        continue;
      for (int wtN : waveTiles) {
        for (int wcN : waveCounts) {
          int bN = mfmaDim * wtN * wcN;
          if (bN < mfmaDim || bN > maxTile)
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
  const int numWarpsCandidates[]  = {1, 2, 4, 8};
  const int numStagesCandidates[] = {1, 2, 3, 4};

  // Step 5: enumerate all combinations and keep feasible ones.
  std::vector<TritonGemmConfig> candidates;
  for (auto [bM, bN] : mnPairs) {
    for (int bK : blockKCandidates) {
      if (bK > 256)
        continue;
      for (int nW : numWarpsCandidates) {
        for (int nS : numStagesCandidates) {
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

          if (isValidConfig(prob, cfg, hw))
            candidates.push_back(cfg);
        }
      }
    }
  }
  return candidates;
}

} // namespace mlir::triton::AMD::perf
