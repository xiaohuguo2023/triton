//===-- PerfModel.h - Triton AMD analytical performance model ---*- C++ -*-===//
//
// Analytical performance model for AMD GPU GEMM kernels compiled by Triton.
//
// Inspired by Origami (AMD hipBLASLT), but built specifically for Triton's
// execution model:  software-pipelined loops, async global→LDS copies,
// MFMA/WMMA matrix cores, and the wave-quantisation effects of Triton's
// static tile mapping.
//
// The model answers three practical questions for each candidate config:
//   1. Is the config *feasible*?  (LDS fits, VGPR budget not blown, tile
//      dimensions divide the problem shape evenly enough.)
//   2. What resources does it consume?  (VGPR count, LDS bytes, occupancy,
//      number of scheduling waves.)
//   3. How fast is it?  (Predicted TFLOPS from a roofline + wave-quant model.)
//
// Callers in AccelerateAMDMatmul.cpp, BlockPingpong.cpp and LowerLoops.cpp can
// use the lightweight free functions; no class instantiation is required.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODEL_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODEL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::triton::AMD::perf {

//===----------------------------------------------------------------------===//
// 1. Hardware description
//===----------------------------------------------------------------------===//

/// AMD GPU architecture families supported by this model.
enum class Arch {
  CDNA1,   ///< gfx908  – MI100
  CDNA2,   ///< gfx90a  – MI200 (MI210 / MI250)
  CDNA3,   ///< gfx940 / gfx941 / gfx942  – MI300
  CDNA4,   ///< gfx950  – MI350
  RDNA3,   ///< gfx1100 / gfx1101 / gfx1102  – RX 7000 series
  RDNA4,   ///< gfx1200 / gfx1201  – RX 9000 series
  GFX1250, ///< gfx1250  – Radeon AI Pro
  Unknown,
};

/// Parse a target-triple arch string (e.g. "gfx942", "gfx1100") to Arch.
Arch archFromString(llvm::StringRef archStr);

/// Static hardware constants for one AMD GPU architecture.
///
/// All bandwidth / clock figures represent typical peak ("boost") values.
/// Use HardwareInfo::get() to obtain a pre-populated instance.
struct HardwareInfo {
  Arch arch = Arch::Unknown;

  // ── Compute ──────────────────────────────────────────────────────────────
  int numCUs = 0;          ///< Total compute units on the device
  int numSimdPerCU = 4;    ///< SIMD units per CU  (4 on CDNA, 2 on RDNA3/4)
  int waveSize = 64;       ///< Wavefront width in lanes (64 CDNA, 32 RDNA3/4)
  int vgprPerSimd = 256;   ///< Total VGPR slots per SIMD unit
  int vgprAllocGranule = 4;///< VGPR allocation rounded up to this multiple
  int maxWavesPerSimd = 10;///< Hardware-enforced max wavefronts per SIMD

  // ── Memory hierarchy ─────────────────────────────────────────────────────
  int ldsPerCU = 65536;    ///< LDS (shared memory) bytes per CU
  int l2SizeBytes = 0;     ///< L2 cache size per XCD in bytes
  int mallSizeBytes = 0;   ///< Infinity Cache / MALL size in bytes (0 = none)
  int numXCDs = 1;         ///< Number of XCDs (chiplets); 1 for monolithic dies

  // ── Bandwidth / clocks ────────────────────────────────────────────────────
  /// Peak global memory bandwidth in bytes per clock cycle
  /// (= peak_BW_GBps / clockMHz * 1000).
  double peakMemBwBytesPerCycle = 0.0;
  /// Peak L2 bandwidth in bytes per clock cycle (device-wide).
  /// Much higher than DRAM; used to compute effective bandwidth with L2 hits.
  double peakL2BwBytesPerCycle = 0.0;
  double clockMHz = 0.0;   ///< Typical boost clock in MHz

  // ── Derived helpers ───────────────────────────────────────────────────────

  /// Peak FP16→FP32 MFMA throughput via 32×32 tiles, in FLOP/cycle/CU.
  /// Used as the compute roof of the roofline model.
  double peakMfmaFlopsPerCycleCU() const;

  /// Populate a HardwareInfo from an arch string.  Returns an Unknown instance
  /// (with zeroed fields) for unrecognised architectures.
  static HardwareInfo get(llvm::StringRef archStr);
  static HardwareInfo get(Arch arch);
};

//===----------------------------------------------------------------------===//
// 2. MFMA instruction throughput table
//===----------------------------------------------------------------------===//

/// Coarse element-type category used for MFMA throughput look-up.
/// Finer distinctions (E4M3 vs E5M2 FP8, etc.) don't affect throughput.
enum class ElemKind {
  FP64,
  FP32,
  TF32,  ///< XFloat32  (CDNA3+ only)
  FP16,
  BF16,
  FP8,   ///< Any 8-bit float variant
  FP6,   ///< Any 6-bit float variant
  FP4,   ///< Any 4-bit float variant
  I8,
  Unknown,
};

/// Bits per element for each ElemKind (FP4 = 4, FP6 = 6, …).
int elemKindBits(ElemKind k);

/// Convert an element bit-width to its ElemKind (FP variants only).
/// Returns ElemKind::Unknown for unrecognised widths.
ElemKind elemKindFromBits(int bits, bool isFloat = true, bool isBF = false);

/// Per-instruction MFMA throughput descriptor.
struct MfmaInstrInfo {
  int mDim;              ///< Output M tile size
  int nDim;              ///< Output N tile size
  int kDim;              ///< K consumed per instruction
  int throughputCycles;  ///< Reciprocal throughput: cycles per instr per SIMD
  ElemKind aKind;        ///< Input A element type category
  ElemKind cKind;        ///< Accumulator element type category
};

/// Look up the MFMA throughput descriptor for the given arch, instruction
/// tile shape and element types.  Returns std::nullopt if unsupported.
std::optional<MfmaInstrInfo> getMfmaInstrInfo(Arch arch, int mDim, int nDim,
                                               ElemKind aKind, ElemKind cKind);

//===----------------------------------------------------------------------===//
// 3. GEMM problem and kernel configuration
//===----------------------------------------------------------------------===//

/// Describes the mathematical GEMM problem being compiled.
struct GemmProblem {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t batchSize = 1;

  ElemKind aKind = ElemKind::FP16;
  ElemKind bKind = ElemKind::FP16;
  ElemKind cKind = ElemKind::FP32; ///< Accumulator / output element type

  int aBits = 16;  ///< sizeof(a_elem) * 8  (4 for FP4, 8 for FP8, …)
  int bBits = 16;
  int cBits = 32;
};

/// Describes a candidate Triton GEMM kernel configuration.
///
/// These correspond directly to Triton autotuner parameters plus compiler
/// decisions made inside AccelerateAMDMatmul.cpp and LowerLoops.cpp.
///
/// Population guide
/// ────────────────
/// Before AccelerateAMDMatmul runs (e.g. for pre-selection of mfmaNonKDim):
///   blockM/N/K   ← RankedTensorType::getShape() on the tt.dot result / A operand
///   numWarps     ← ttg::lookupNumWarps(dotOp)          [reads "ttg.num-warps"]
///   kWidth       ← leave 0; estimateVgpr() will derive it from mfmaNonKDim
///                  and the MFMA throughput table (kBase field)
///   mfmaNonKDim  ← 0 to let selectMfmaNonKDim() choose, or set explicitly
///   numStages    ← pass option
///   useAsyncCopy ← detect ttg::AsyncCopyGlobalToLocalOp in parent scf.for
///   bypassLds    ← detect absence of LDS ops in parent scf.for
///
/// After AccelerateAMDMatmul runs (e.g. for LDS guard in LowerLoops):
///   kWidth       ← DotOperandEncodingAttr::getKWidth() on A or B operand type
///   mfmaNonKDim  ← AMDMfmaEncodingAttr::getInstrShape()[0] on result type
///   numWarps     ← AMDMfmaEncodingAttr::getWarpsPerCTA() product on result type
struct TritonGemmConfig {
  int blockM = 128;         ///< BLOCK_M  (output tile M)
  int blockN = 128;         ///< BLOCK_N  (output tile N)
  int blockK = 32;          ///< BLOCK_K  (reduction tile K)
  int numStages = 2;        ///< Software pipeline stages
  int numWarps = 4;         ///< Wavefronts per CTA  (ttg::lookupNumWarps)
  /// Shared-memory load vectorisation width in elements per thread.
  /// When 0, estimateVgpr() derives kWidth from the MFMA kBase stored in
  /// MfmaInstrInfo (= kDim / waveSize fraction that each thread holds).
  /// Set explicitly from DotOperandEncodingAttr::getKWidth() once the pass
  /// has run, or leave 0 for pre-pass model queries.
  int kWidth = 0;
  int mfmaNonKDim = 0;      ///< Chosen MFMA mDim/nDim (0 = let model choose)
  bool bypassLds = false;   ///< True if operands go directly to MFMA (no LDS)
  bool useAsyncCopy = true; ///< True if global→LDS copies are async
  int wavesPerEu = 0;       ///< Requested waves per EU hint (0 = unset)
  int kPack = 1;            ///< LDS vector multiplier (1 or 2; pass option kPack)
  int groupSizeM = 8;       ///< Tile scheduling slab width (Triton GROUP_SIZE_M)
};

//===----------------------------------------------------------------------===//
// 4. Performance estimate result
//===----------------------------------------------------------------------===//

/// Full performance estimate returned by estimatePerf().
struct PerfEstimate {
  // ── Resource usage ────────────────────────────────────────────────────────
  int vgprCount = 0;       ///< Estimated VGPRs per wavefront (after rounding)
  int ldsBytes = 0;        ///< Estimated LDS bytes per CTA
  int numBuffers = 0;      ///< LDS double-buffer count (= effective num_stages)
  int wavesPerSimd = 0;    ///< Resident wavefronts per SIMD from VGPR limit
  int ctasPerCU = 0;       ///< Resident CTAs per CU (min of VGPR and LDS)

  // ── Wave quantisation ─────────────────────────────────────────────────────
  int64_t totalOutputTiles = 0; ///< ceil(M/BM) * ceil(N/BN) * batch
  int numWaves = 0;        ///< Scheduling waves = ceil(outputTiles / numCUs)
  double waveEfficiency = 0.0;  ///< Utilisation of the last wave  [0..1]

  // ── Latency breakdown (cycles, per tile, on one CU) ───────────────────────
  double computeCycles = 0.0;   ///< Cycles in MFMA instructions
  double memoryCycles = 0.0;    ///< Cycles waiting for global memory
  double pipelineOverlap = 0.0; ///< Memory hidden by SW pipelining  [0..1]
  double effectiveTileCycles = 0.0; ///< max(compute, (1-overlap)*memory)

  // ── Derived metrics ────────────────────────────────────────────────────────
  double occupancy = 0.0;           ///< Wavefront slot utilisation  [0..1]
  double arithmeticIntensity = 0.0; ///< FLOP per global-memory byte
  bool isComputeBound = false;
  bool ldsExceeded = false;   ///< Config requires more LDS than available
  bool likelySpills = false;  ///< VGPR count likely causes register spilling
  bool isValid = false;       ///< Config is feasible on this hardware

  // ── Predicted throughput ──────────────────────────────────────────────────
  double predictedTflops = 0.0;
};

//===----------------------------------------------------------------------===//
// 5. Public API
//===----------------------------------------------------------------------===//

/// Estimate the number of LDS ping-pong buffers used by the pipeline.
///
/// Mirrors the logic in LowerLoops.cpp::initSchedule():
///   numBuffers = numStages  (for async copy pipelines)
///   numBuffers = numStages - 1  (for synchronous pipelines, min 1)
int estimateNumBuffers(const TritonGemmConfig &cfg);

/// Estimate per-wavefront VGPR usage.
///
/// Model:
///   vgpr_accum  = ceil(BLOCK_M * BLOCK_N * cBits/8 / (waveSize * 4))
///   vgpr_a_frag = ceil(BLOCK_M * kWidth * aBits/8  / (waveSize * 4))
///   vgpr_b_frag = ceil(BLOCK_N * kWidth * bBits/8  / (waveSize * 4))
///   vgpr_misc   = empirical overhead (~28 VGPRs for pointers, indices, masks)
///   total       = roundUp(vgpr_accum + vgpr_a_frag + vgpr_b_frag + vgpr_misc,
///                         hw.vgprAllocGranule)
int estimateVgpr(const GemmProblem &prob, const TritonGemmConfig &cfg,
                 const HardwareInfo &hw);

/// Estimate per-CTA LDS bytes consumed by operand buffers.
///
/// Model:
///   For each operand X in {A, B}:
///     lds_X = numBuffers * BLOCK_M_or_N * (BLOCK_K + padding) * xBits/8
///   padding = 8 elements  (heuristic; avoids the most common bank-conflict
///             pattern without needing architecture-specific alignment)
///
/// bypassLds = true  →  returns 0 (operands stay in registers).
int estimateLdsBytes(const GemmProblem &prob, const TritonGemmConfig &cfg,
                     const HardwareInfo &hw);

/// Full analytical performance estimate for problem + config + hardware.
///
/// The model combines:
///   • VGPR and LDS resource accounting → occupancy
///   • Roofline (peak MFMA FLOP/cycle vs peak memory BW)
///   • Software-pipeline overlap factor  (function of numStages)
///   • Wave-quantisation penalty         (tail wave under-utilisation)
PerfEstimate estimatePerf(const GemmProblem &prob, const TritonGemmConfig &cfg,
                          const HardwareInfo &hw);

/// Quick feasibility check.  Returns false if:
///   • LDS usage exceeds hw.ldsPerCU
///   • blockK is not divisible by the chosen MFMA kDim
///   • Estimated VGPR count exceeds hw.vgprPerSimd  (hard spill)
///   • numWarps is 0 or not a power of two
bool isValidConfig(const GemmProblem &prob, const TritonGemmConfig &cfg,
                   const HardwareInfo &hw);

/// Sort a list of candidate configs by predicted throughput (best first).
///
/// Configs that exceed the LDS capacity are excluded from the output entirely
/// (pre-filtered before the expensive roofline computation). Configs that fail
/// other validity checks (VGPR, kDim alignment) are pushed to the end.
///
/// When topK > 0, only the top-K results are returned and std::partial_sort is
/// used instead of std::stable_sort — O(N log K) vs O(N log N). Pass topK=0
/// (default) to return all valid configs in ranked order.
///
/// Among equally-ranked configs, prefer higher arithmetic intensity, then
/// larger blockM (matches Origami's tie-breaking convention).
std::vector<TritonGemmConfig>
rankConfigs(const GemmProblem &prob, llvm::ArrayRef<TritonGemmConfig> configs,
            const HardwareInfo &hw, size_t topK = 0);

/// Select the optimal GROUP_SIZE_M (tile scheduling slab width) using
/// Origami's predict_workgroup_mapping algorithm: iterate over candidates
/// {1,4,6} ∪ divisors(wgm_cap), evaluate L2 working-set cost for the last
/// XCD in the first timestep, return the candidate with minimum cost.
///
/// This is equivalent to Origami's `predict_workgroup_mapping` and controls
/// the same parameter as Triton's GROUP_SIZE_M in the matmul kernel.
int selectGroupSizeM(const GemmProblem &prob, const TritonGemmConfig &cfg,
                     const HardwareInfo &hw);

/// Analytically select the best MFMA mDim / nDim for a problem.
///
/// Returns 32 when the tile fits and the instruction is available, 16
/// otherwise.  Falls back to 4 for very small M or N (< 16).
///
/// This replaces the pure-threshold heuristic in
/// AccelerateAMDMatmul.cpp::chooseMfmaInstruction() (lines 172-196) with a
/// model-driven decision that considers occupancy and compute efficiency.
int selectMfmaNonKDim(const GemmProblem &prob, const TritonGemmConfig &cfg,
                      const HardwareInfo &hw);

/// Kernel-flavor selector for generateCandidates() / rankConfigs().
///
/// The standard triton matmul kernel is compiler-pipelined and supports a wide
/// range of (numWarps, numStages, blockM/N/K) combinations. Gluon kernels
/// (e.g. v9_beyond_hotloop) have structural constraints fixed at the layout
/// level: warps_per_cta=[2,2] requires numWarps=4, the 4-quadrant accumulator
/// requires blockM/blockN to be multiples of 128, and the loop-unrolled-by-2
/// pipeline requires K to be divisible by 2*blockK.
enum class KernelType {
  Standard, ///< Compiler-pipelined triton matmul (full sweep)
  Gluon,    ///< Hand-tuned gluon kernel with v9-style 4-quadrant + 2x unroll
};

/// Generate a candidate set of TritonGemmConfig tuples for a given problem
/// and hardware target.
///
/// Tile size generation follows Origami's wave-based approach:
///   blockM = mfmaDim × waveTileM × waveCountM
///   blockN = mfmaDim × waveTileN × waveCountN
///   blockK = multiple of the MFMA kDim from the throughput table
///
/// For KernelType::Standard:
///   numWarps swept over {4, 8}; numStages swept over {1, 2, 3}.
///   On CDNA4 (gfx950): forced to numWarps=8, numStages=2 (pingpong scheduler).
///
/// For KernelType::Gluon:
///   numWarps fixed to 4 (warps_per_cta=[2,2] structural requirement).
///   numStages fixed to 2 (gluon's hand-tuned 8-deep async pipeline).
///   Tile constraints: blockM/blockN ≥ 128, divisible by 128 (4-quadrant);
///   blockK % 32 == 0 (gfx950 MFMA kDim); K % (2*blockK) == 0; K/blockK ≥ 4.
///
/// All candidates are filtered through isValidConfig — only feasible configs
/// (LDS fits, VGPR fits, kDim alignment) are returned.
///
/// The returned list is unranked; pass it to rankConfigs() to sort by
/// predicted TFLOPS.
std::vector<TritonGemmConfig>
generateCandidates(const GemmProblem &prob, const HardwareInfo &hw,
                   KernelType kernelType = KernelType::Standard);

} // namespace mlir::triton::AMD::perf

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODEL_H_
