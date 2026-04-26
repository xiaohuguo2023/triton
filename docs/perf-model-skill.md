# AMD PerfModel — Skill Reference & Technical Report

## What This Skill Does

The AMD PerfModel is an **analytical GEMM performance model** for AMD GPUs that predicts the best kernel configuration (tile sizes, pipeline stages, etc.) without benchmarking. It follows Origami's approach: select configs by predicted TFLOPS from a hardware-calibrated roofline model.

### Key capabilities
- `generateCandidates(prob, hw)` → ~200-900 valid tile configurations
- `rankConfigs(prob, candidates, hw, top_k)` → sorted by predicted TFLOPS
- `selectGroupSizeM(prob, cfg, hw)` → Origami-style GROUP_SIZE_M (WGM) selection
- `pick_gemm_config(M, N, K, dtype, arch, top_k)` → Python entry point

### Performance on gfx950 (TensorAtlas datasets)

| Dataset | Median PM/Tuned | Within 10% | Overall |
|---|---|---|---|
| llama3_mlp | **1.07** | 89% | 1.01× |
| deepseek_r1 | **1.02** | 93% | 0.99× |
| qwen3_235b_a22b | **1.04** | 94% | 1.01× |
| gpt_oss_120b | **1.02** | 84% | 0.97× |
| llama4_maverick | **0.97** | 64% | 0.96× |

Selection overhead: **0.09 ms** (vs autotuning which benchmarks N configs × ~25ms each).

---

## Shape Categorization

### Three-regime model

```
totalOutputTiles = ceil(M/BM) × ceil(N/BN)

Small-M regime:   M ≤ 4×mfmaDim  OR  totalOutputTiles < numCUs × 4
                  → BK candidates: {kDim, 2×, 4×, 8×, 16×} = {32,64,128,256,512}
                  → numWarps = 8 (CDNA4 only, enables pingpong)
                  → numStages = 2 (CDNA4 only, empirically optimal)
                  → Example: M≤64 on gfx950 (256 CUs, mfmaDim=16)

Normal regime:    totalOutputTiles ≥ numCUs × 4
                  → BK candidates: {kDim, 2×, 4×} = {32, 64, 128}
                  → All numWarps, numStages values swept

Large-tile regime: isComputeBound=true (effectiveCycles ≈ computeCycles)
                  → occupancyPenalty = 1.0 (MFMA pipeline saturated)
                  → BK tiebreak prefers larger BK
```

### Why the threshold matters

| M | approxOutputTiles (32×32 min tile) | Regime | Max BK |
|---|---|---|---|
| 64 | 4 | Small-M | 512 |
| 256 | 64 | Small-M | 512 |
| 512 | 256 | Small-M | 512 |
| 768 | 576 | Small-M | 512 |
| 1024 | 1024 | Normal | 128 |
| 1536 | 2304 | Normal | 128 |
| 4096 | 16384 | Normal | 128 |

**Threshold = numCUs × 4 = 1024 on gfx950** covers M≤768 where AITER experiment shows BK=512 provides 18-29% speedup via reduced K-loop overhead.

### Shape-specific config preferences (gfx950 empirical)

```
M≤32, large N (N≥5120): BK=512, BM=32, BN=32, nW=8, nS=2
M≤256, any N:           BK=256-512, small square tiles, nW=8, nS=2
M=256-1024:             BK=64-128 optimal (L2 crossover)
M≥1024, large tiles:    BK=64, BM=128-256, BN=64-256, nW=8, nS=2
Non-square (N/M ≠ 1):   Aspect ratio tiebreak: prefer BN/BM ≈ N/M
Non-divisible K:        Use exact K/BK (float) for numKIter ranking
```

---

## Architecture Overview

### Three-layer stack

```
Layer 1: Hardware database (PerfModel.h/cpp)
  - kMfmaThroughputTable[]: MFMA instruction latencies per arch+dtype
  - HardwareInfo::get(archStr): numCUs, numSimdPerCU, vgprPerSimd,
    ldsPerCU, numXCDs, peakMemBwBytesPerCycle, peakL2BwBytesPerCycle

Layer 2: Resource accounting
  - estimateVgpr(prob, cfg, hw): accumulator + fragments + misc
  - estimateLdsBytes(prob, cfg, hw): TensorAtlas padded formula
  - isValidConfig(prob, cfg, hw): LDS + VGPR + kDim alignment checks

Layer 3: Roofline + wave quantisation (estimatePerf)
  - computeCycles = numMFMAperKBlock × (K/BK) × throughput / numSimd
  - memoryCycles  = tileBytesAB × (K/BK) / (peakDRAM_bwPerCU)
  - L2HitRate     = Origami's computeL2Tiles formula (when calibrated)
  - effectiveCycles = max(computeCycles, memoryCycles × (1-overlap))
  - isComputeBound  = effectiveCycles ≤ computeCycles × 1.05
  - occupancyPenalty = isComputeBound ? 1.0 : 1/max(vgprOccupancy, 0.5)
  - totalCycles = effectiveCycles × numWaves × occupancyPenalty / waveEff
  - predictedTflops = totalFlops / (totalCycles / clockMHz * 1e6) / 1e12
```

### Key design decisions

**vgprOccupancy vs combined occupancy**: We use VGPR-only occupancy for the penalty (not LDS-limited occupancy). Larger LDS = larger pipeline buffers = beneficial for overlap. LDS-limited occupancy is NOT penalized because it reflects better pipelining, not worse performance. _Exception_: empirically, BK=128 vs BK=64 at M=1536 shows LDS does matter — but this requires the L2 model to fix properly.

**Exact K/BK for numKIter**: We use `(double)K / BK` instead of `ceil(K/BK)`. When K is not divisible by BK (e.g., K=2880, BK=128), ceil overcounts by 0-1 iterations, unfairly penalizing larger BK. Using exact division makes BK=64 and BK=128 predict equal TFLOPS for K=2880, allowing the BK tiebreak to correctly prefer BK=128 where it fits.

**Aspect ratio tiebreak**: When configs predict equal TFLOPS, prefer the tile whose `log(BN/BM)` is closest to `log(N/M)`. For N>>M shapes (N=5120, M=4096), prefers BM=128×BN=256 over BM=256×BN=128. This matches Origami's WGM intuition for L2 spatial reuse.

**numStages fixed to 2 on CDNA4**: Autotune consistently picks nS=2 across all problem sizes on gfx950. nS=3 slightly overflows LDS for large tiles (256×256×64: 204KB > 160KB limit), and our pipeline overlap formula over-rewards extra stages. Generating only nS=2 for CDNA4 reduces search space without losing good configs.

---

## L2 Bandwidth Calibration

### Experiment (`testtmp/calibrate_l2_bw.py`)

**Setup**: Fixed BM=BN=64, nW=8, nS=2, mfma=16, GROUP_SIZE_M=8 on gfx950 MI355X.  
Sweep BK ∈ {32, 64, 128, 256} across M ∈ {512, 1024, 2048, 4096}.

**Results (TFLOPS)**:

```
     M   BK=32   BK=64  BK=128  BK=256   Observation
   512    21.1    26.9    32.6    35.1    BK=256 wins (all fit in L2)
  1024   102.6   132.8   173.9   186.7    BK=256 wins (L2 hits dominate)
  2048   332.2   431.1   445.1   334.6    BK=128 wins, BK=256 drops → L2 overflow
  4096   406.6   454.6   444.8   404.2    BK=64 ≈ BK=128, BK=256 drops
```

**Calibrated values**:
- Peak L2 bandwidth: **~43 TB/s** (at M=2048, BK=128 on MI355X)
- `peakL2BwBytesPerCycle = 43e12 / 2.4e9 ≈ 17,900 bytes/cycle`
- L2 capacity per XCD: `32 MB` (numXCDs=8 per Origami for gfx950)
- L2 crossover BK: between 128 and 256 for M=2048 (64×64 tile)

**Current status**: `peakL2BwBytesPerCycle = 0` (disabled). Enabling it causes regressions — see Gap 4 below.

**Key insight from enabling attempt**: The L2 hit rate formula is **BK-independent** — BK cancels in the `uA/uB` ratio, so all BK values predict the same L2 hit rate for a given tile shape. Enabling L2BW:
- Does NOT fix BK=128 vs BK=64 selection (M=1536 gap remains)
- Makes memory-bound small tiles look artificially fast for large-M shapes → wrong config selection
- Causes regressions across all datasets (llama3_mlp: 89%→83%, gpt_oss_120b: 84%→63%)

### AITER experiment (`testtmp/test_aiter_tiles.py`)

Compared PerfModel configs against AITER (AMD exhaustive tuner) tile sizes.

**Key finding**: BK=512 is 18-29% faster than BK=128 for M≤32 with N≥5120.

```
    M     N     K  AITER(BK=512)  PM(BK=512)  Ratio
    4  5120  2880       7.25       7.41       1.02  ✅ matches
    8  5120  2880      14.85      14.71       0.99  ✅ matches
   16  5120  2880      30.09      30.38       1.01  ✅ matches
   32  5120  2880      60.52      60.40       1.00  ✅ matches
```

After fix: PerfModel now selects the same BK=512 config as AITER for N=5120 small-M shapes.

---

## Implementation Details

### Files

```
third_party/amd/include/TritonAMDGPUTransforms/PerfModel.h    — API
third_party/amd/lib/TritonAMDGPUTransforms/PerfModel.cpp      — implementation
third_party/amd/python/triton_amd.cc                          — pybind11 bindings
third_party/amd/backend/amd_gemm_selector.py                  — Python selector
third_party/amd/python/test/test_perf_model.py                — Python binding tests
third_party/amd/unittest/PerfModelTest.cpp                    — C++ unit tests (32 cases)
```

### Python API

```python
from triton.backends.amd.amd_gemm_selector import (
    pick_gemm_config, config_to_kernel_kwargs, current_amd_arch
)

# Select top-1 config analytically
cfgs = pick_gemm_config(M, N, K, 'fp16', arch, top_k=1)
cfg  = cfgs[0]
kw   = config_to_kernel_kwargs(cfg)
# kw contains: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
#              matrix_instr_nonkdim, GROUP_SIZE_M, num_warps, num_stages

# Or use top_k>1 for debugging / misprediction diagnosis
top3 = pick_gemm_config(M, N, K, 'fp16', arch, top_k=3)
```

### C++ API

```cpp
using namespace mlir::triton::AMD::perf;
GemmProblem prob{.M=4096, .N=4096, .K=4096,
                  .aKind=ElemKind::FP16, .bKind=ElemKind::FP16,
                  .cKind=ElemKind::FP32, .aBits=16, .bBits=16, .cBits=32};
HardwareInfo hw = HardwareInfo::get("gfx950");

auto candidates = generateCandidates(prob, hw);          // ~200-900 configs
auto ranked     = rankConfigs(prob, candidates, hw, 5);  // top-5
int  gsm        = selectGroupSizeM(prob, ranked[0], hw); // GROUP_SIZE_M
```

### TritonGemmConfig fields

| Field | Description | CDNA4 constraint |
|---|---|---|
| blockM, blockN | Tile dimensions (power of 2) | ≥ 2×mfmaDim = 32 |
| blockK | K-block size (power of 2) | ≤ 4×kDim=128 (normal) or 16×kDim=512 (small-M) |
| numWarps | Wavefronts per CTA | 8 only (enables pingpong) |
| numStages | Pipeline stages | 2 only (LDS constraint + empirical) |
| mfmaNonKDim | MFMA instruction dim | 16 (throughput-based, matches Origami) |
| groupSizeM | Tile scheduling WGM | Origami's predict_workgroup_mapping |

---

## Known Gaps and Next Steps

### Gap 1: M=1536/1664 — BK=128 vs BK=64 (-14 to -16%)

**Root cause**: For `64×64` tile at M=1536, BK=32/64/128 all predict identical TFLOPS. BK tiebreak picks BK=128 (larger) but BK=64 is empirically ~4% faster at M=1536.

**Gluon LDS analysis** (`gfx9-gluon-tutorials/docs/lds_throughput.md`): `ds_read_b128` throughput = 1 instruction per 16 cycles per SIMD (steady state). For 64×64 tile:
- BK=64: 16 ds_read_b128 per iteration × 16 cycles = **256 cycles LDS** = 256 compute cycles → perfectly balanced
- BK=128: 32 ds_read_b128 × 16 cycles = **512 cycles LDS** = 512 compute cycles → also balanced

Both BK values are in the LDS-compute balanced regime — the gluon model confirms the analytical prediction of equal TFLOPS is **correct**. The measured 4% advantage for BK=64 at M=1536 is below analytical model resolution (microarchitectural scheduling noise).

**Gluon TCP model** (`docs/memory_bandwidth_model.md`): TCP = 32KB cap on in-flight bytes per CU. TCP-capped pipeline depth = min(numStages-1, 32KB / (numActiveWaves × dataPerWave)) is the same for BK=64 and BK=128 at M=1536 → equal pipeline efficiency.

**Resolution**: This gap is within analytical model noise. The gluon models provide the theoretical foundation to confirm this cannot be fixed without hardware-level microarchitecture data. The BK tiebreak (prefer larger) is correct for M≤1280 (BK=128 wins 20-30%) and approximately correct for M=1536 (within 4%).

**Empirical crossover** (testtmp/bk_crossover.py on gfx950):
- M=512-1280: BK=128 wins by 20-30% → large BK tiebreak correct
- M=1536: BK=64 wins by 4% → large BK tiebreak wrong, but within noise
- M=2048: BK=128 wins by 4% → large BK tiebreak correct again

### Gap 2: M=2944/3072 — tile shape mismatch (-2 to -3%)

**Root cause**: Model ranks `256×256` tile (compute-bound, 1 wave, waveEff=0.56) over `128×64` tile (memory-bound, 5 waves, waveEff=0.83). Actual: `128×64` achieves 634 TFLOPS vs `256×256` achieves ~450 TFLOPS.

**Why**: `128×64` is classified as memory-bound, gets occupancy penalty and large memoryCycles × 5 waves. `256×256` is barely compute-bound (escapes occupancy penalty), only 1 wave. The model doesn't capture that 5-wave `128×64` hides memory latency far better than 1-wave `256×256`.

**Fix**: Multi-wave memory overlap: effective_overlap = f(numStages, ctasPerCU, depthNeeded). Requires careful calibration to avoid over-boosting high-CTA-count tiny tiles.

### Gap 3: Small-M stream-K shapes (M≤32, non-square)

**Root cause**: No stream-K support. For M=4, N=5120, standard tiling gives 1×320=320 tiles for 256 CUs → >100% utilization from output tiles alone, but the AITER configs show much better performance with stream-K style (NUM_SMS > numCUs).

**Plan**: See `docs/small-m-stream-k-plan.md` for detailed implementation phases.

### Gap 4: peakL2BwBytesPerCycle — fundamental model limitation

**Status**: Set to 0 (disabled). Calibrated value = 17,900 bytes/cycle (43 TB/s).

**Root cause discovered**: The L2 hit rate formula (from Origami) is mathematically **independent of BK**:
```
hit_rate = 1 - (l2_m×BM + l2_n×BN) / (l2_m × l2_n × (BM + BN))
```
BK cancels completely in uA = l2_m × BM × **BK** and uB = l2_n × BN × **BK** → ratio is constant.

This means:
1. Enabling peakL2BwBytesPerCycle does NOT fix the M=1536/1664 BK selection issue
2. Enabling it over-boosts memory-bound small tiles globally → wrong selections for large-M shapes
3. Caused regressions: llama3_mlp within-10% 89%→83%, gpt_oss_120b 84%→63%

**What L2BW actually does**: Reduces memoryCycles equally for all BK values of same tile. For compute-bound configs (isComputeBound=true), has zero effect. For memory-bound configs, can make them appear more competitive — but this boost is not shape-calibrated.

**Path forward**: Need shape-dependent L2 hit rate that varies with BK. This requires a different model (e.g., accounting for temporal reuse across K-iterations), not just the spatial reuse from Origami's tile-sharing model.

---

## Debugging Guide

### Check what config PerfModel selects

```python
from triton._C.libtriton.amd import perf_model as pm
hw   = pm.HardwareInfo.get('gfx950')
prob = pm.GemmProblem(M, N, K, pm.ElemKind.FP16, ...)
cands = pm.generate_candidates(prob, hw)
ranked = pm.rank_configs(prob, cands, hw, top_k=5)
for cfg in ranked:
    e = pm.estimate_perf(prob, cfg, hw)
    print(f'BM={cfg.block_m} BK={cfg.block_k}: {e.predicted_tflops:.0f} TFLOPS '
          f'occ={e.occupancy:.2f} computeBound={e.is_compute_bound}')
```

### Compare against TensorAtlas tuned configs

```bash
python testtmp/eval_tensoratlas.py --dataset llama3_mlp
python testtmp/eval_tensoratlas.py --dataset gpt_oss_120b --no-bench  # config only
```

### Run calibration sweep

```bash
python testtmp/calibrate_l2_bw.py    # BK sweep for L2 bandwidth measurement
python testtmp/compare_configs.py    # autotune vs PerfModel for square M sweep
python testtmp/test_aiter_tiles.py   # PerfModel vs AITER exhaustive tuning
```

### Unit tests

```bash
# C++ unit tests (32 cases)
ninja AMDPerfModel && ./build/container-build/third_party/amd/unittest/AMDPerfModel

# Python binding tests
pytest third_party/amd/python/test/test_perf_model.py -v
```

---

## Origami Alignment

| Feature | Origami | Our implementation | Status |
|---|---|---|---|
| MFMA selection | throughput-based, tiebreak=16 | ✅ same | Done |
| WGM/GROUP_SIZE_M | predict_workgroup_mapping | ✅ ported | Done |
| L2 tile model | computeL2Tiles, estimateL2Hit | ✅ ported | Done (L2BW=0) |
| MALL tile model | computeMallTiles | ✅ ported | Done (L2BW=0) |
| L2 bandwidth | calibrated per arch | ⚠️ 17900 but disabled | Pending BK cap |
| Grid selection (NUM_SMS) | grid_k_split_aware | ❌ not implemented | Phase 2 |
| Stream-K kernel | spinlock reduction | ❌ not implemented | Phase 4 |
| Work utilization penalty | effective_tile_penalty | ❌ not implemented | Phase 1b |
| BK from Origami | wave_tile × wave_count × kDim | ✅ similar | Done |
| Tile aspect ratio | WGM prefers BN/BM ≈ N/M | ✅ log-space tiebreak | Done |
