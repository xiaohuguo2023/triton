# Proposal: Triton-AMD Analytical Performance Model

**Author:** AMD Triton team
**Status:** Draft
**Date:** 2026-04-12

---

## 1. Background and Motivation

Triton's AMD backend (`third_party/amd/`) compiles user-annotated tile parameters
(`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_stages`, `num_warps`, …) into optimised
AMDGCN kernels through a chain of MLIR passes.  Several of those passes make
decisions that directly determine kernel performance — which MFMA instruction
size to use, how to distribute warps across the output tile, how many LDS
double-buffers to allocate — yet every one of those decisions today is driven by
**hardcoded thresholds or simple greedy rules** that do not model the underlying
hardware.

The consequence is twofold:

1. **Suboptimal code generation** for non-square or memory-bound GEMMs where the
   right answer depends on the interplay of occupancy, wave quantisation, and
   memory bandwidth — quantities the current heuristics ignore.

2. **Slow autotuning** because the Python-level `triton.autotune` must
   exhaustively benchmark every candidate configuration to find the best one,
   even when many configurations are predictably bad (LDS overflow, VGPR
   spilling, poor wave utilisation).

AMD's **Origami** library (`rocm-libraries/shared/origami`) demonstrates that
analytical performance modelling can replace empirical search for GEMM kernel
selection.  However, Origami is designed specifically for the **TensileLite /
hipBLASLT** execution model and cannot be used directly inside Triton.

This proposal describes a **Triton-specific analytical performance model**,
inspired by Origami's architecture but built from scratch to match Triton's
execution semantics.

---

## 2. Analysis of Origami

### 2.1 What Origami Does

Origami (`rocm-libraries/shared/origami`) is a C++ library that analytically
selects GEMM kernel configurations for AMD GPUs without benchmarking.  Its
public API (`origami.hpp`) exposes:

- `select_config()` / `rank_configs()` — choose or rank kernel configs by
  predicted latency.
- `select_workgroup_mapping()` — optimise warp-to-CU assignment for L2 reuse
  across XCDs.
- `select_staggerU()` — reduce L2 contention between CUs on the same XCD.
- `compute_perf_gflops()` — predicted TFLOPS for a given config.

The analytical model computes:

```
total_latency = max(compute_latency, memory_latency) × num_waves
```

across the full memory hierarchy (L2, MALL/Infinity Cache, DRAM), using a
per-architecture, per-instruction cycle-accurate throughput table
(`INSTRUCTION_MAP` in `hardware.hpp`) and a trainable heuristics database
(`heuristics_database_t`) with per-arch/per-dtype tuned weights.

For cases where the fast analytical path is insufficient, Origami also ships
**Formocast** — a cycle-accurate simulator for TensileLite GEMM kernels that
models L1/L2/L3 cache hit rates, LDS bank conflicts, and memory queue stalls.

### 2.2 What Is Reusable

| Origami component | Applicability to Triton |
|---|---|
| Hardware architecture constants (CU count, clock, memory BW) | Directly reusable — hardware truth is backend-agnostic |
| Per-instruction MFMA/WMMA throughput table | Directly reusable |
| Analytical framework: compute vs. memory latency, wave quantisation | Reusable as design pattern |
| `select_workgroup_mapping()` XCD-aware L2 reuse math | Reusable concept; needs re-derivation for Triton's warp layout |
| Heuristics database (`heuristics_database_t`) | **Not reusable** — tuned for TensileLite kernel structure |
| Formocast simulator | **Not reusable** — models TensileLite-specific LDS/global queue internals |
| `tensile_params_t`, GSU/LSU split overhead | **Not reusable** — TensileLite-specific concepts |
| Python `OrigamiMatmulSelector` | Thin Triton shim, but depends on the C++ model above |

### 2.3 Why Not Use Origami Directly

Origami's analytical model embeds assumptions that are specific to the
TensileLite execution model and incompatible with Triton:

| Dimension | Origami / TensileLite | Triton |
|---|---|---|
| Kernel structure | Pre-compiled, fixed prefetch depth | JIT-compiled, `num_stages`-driven SW pipeline |
| Register model | Closed-form from TensileLite internals | Depends on `blockM × blockN × dtype × kWidth` |
| LDS model | Fixed GSU/LSU buffer layout | `numBuffers = f(numStages, asyncCopy)` from LowerLoops.cpp |
| Memory access | Formocast models TensileLite LDS/global queues | Buffer ops, async copies, optional bypassLds |
| Occupancy | Tuned via `heuristics_database_t` | Driven by `waves_per_eu` forwarded blind to LLVM |
| Workgroup mapping | XCD-aware for hipBLASLT CTA grid | Set at compile time via `planWarps()` in AccelerateAMDMatmul.cpp |

Additionally, taking Origami as a library dependency would introduce a
significant external dependency into the Triton build for functionality that
only applies to AMD.

---

## 3. Analysis of the Triton AMD Backend

### 3.1 Scope of the Investigation

The following files were analysed in detail:

- `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` (1780 lines)
- `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp` (54 KB)
- `third_party/amd/lib/TritonAMDGPUTransforms/LowerLoops.cpp` (43 KB)
- `third_party/amd/lib/TritonAMDGPUTransforms/ScheduleLoops.cpp` (22 KB)
- `third_party/amd/lib/TritonAMDGPUTransforms/Utility.cpp` (20 KB)
- `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp`
- All 24 source files in the AMD transforms directory

### 3.2 Current State: No Analytical Model Exists

The investigation found **no compile-time analytical performance or cost model**
anywhere in the Triton AMD backend.  All key decisions are driven by:

| Decision | Current mechanism | File / lines |
|---|---|---|
| MFMA instruction size (mDim × nDim) | `min(M,N)` threshold: ≥32→32×32, ≥16→16×16, else 4×64 | `AccelerateAMDMatmul.cpp:172–196` |
| Warp distribution (`planWarps`) | Greedy doubling across whichever dimension has more unoccupied tiles | `AccelerateAMDMatmul.cpp:94–155` |
| `kPack` (shared memory vectorisation) | Static pass parameter, default 1 | `AccelerateAMDMatmul.cpp` pass options |
| Ping-pong scheduling pattern | Hardcoded tile-size thresholds (262 K / 16 M / 32 M / 64 M) | `BlockPingpong.cpp:1057–1250` |
| Number of LDS buffers | `max(1, numStages − 1)` + 1 for async copy | `LowerLoops.cpp:618` |
| LDS capacity check | **None** — allocation is computed but never validated against device limit | `LowerLoops.cpp` |
| Occupancy | Not computed; `waves_per_eu` forwarded opaquely to LLVM | `compiler.py:431` |
| Register pressure | One hardcoded avoidance rule (mfma16×16 + large tile + kWidth=8) | `BlockPingpong.cpp:1232` |

One genuinely analytical sub-system exists: the **LDS bank-conflict padding**
logic in `Utility.cpp::composePaddedLayoutForAsyncCopyCDNA4` models DS
instruction types, bank widths, and X-way conflict tolerance.  This is a
narrowly-scoped but high-quality piece of work that the new model should build
alongside, not replace.

### 3.3 Implications

- The Python-level autotuner must benchmark every candidate config empirically
  because the compiler has no way to reject bad ones a priori.
- LDS overflows are silent: `LowerLoops.cpp` computes `numBuffers` without
  checking that the resulting allocation fits within `hw.ldsPerCU`.
- Register spilling is discovered empirically; only one known-bad configuration
  is hard-blocked.
- Non-square GEMMs and memory-bound workloads are especially poorly served
  because the current heuristics assume square tiles and compute-bound regimes.

---

## 4. Proposed Solution: `PerfModel`

### 4.1 Design Principles

1. **Triton-specific** — models Triton's SW-pipeline structure, `bypassLds`
   mode, async copy depth, and kWidth vectorisation.  Does not attempt to model
   TensileLite or any other backend.

2. **Two-layer architecture** — a core IR-agnostic library (`PerfModel.h/cpp`)
   and a thin IR-aware factory layer (`PerfModelIR.h/cpp`).  The core layer has
   no MLIR dependencies and is usable from Python bindings, standalone tests, and
   future non-GEMM characterisers.  The IR layer reads encoding attributes and
   module attributes using exactly the same APIs that the existing passes use,
   eliminating manual transcription of IR values into structs.

3. **No new external dependencies** — the core layer depends only on
   `llvm::StringRef` and `llvm::ArrayRef`.  The IR layer depends only on Triton
   MLIR headers that are already in every AMD transform pass's include set.

4. **Layered** — hardware tables → resource accounting → roofline + wave
   quantisation.  Each layer is independently testable.

5. **Grounded in the actual IR** — key parameters (`kWidth`, `numWarps`,
   `mfmaNonKDim`) are read from the IR using the canonical APIs rather than
   being approximated by formulas.  Specifically, `kWidth` is derived from the
   MFMA throughput table (`kBase × kPack`) — the same relationship that
   `AccelerateAMDMatmul.cpp` uses — not from `AxisInfoAnalysis`.

6. **Inspired by Origami** — borrows the architectural patterns (hardware table
   structure, compute-vs-memory framing, ranking API) but re-derives every
   model from Triton's execution semantics.

### 4.2 File Layout

```
third_party/amd/
  include/TritonAMDGPUTransforms/
    PerfModel.h        ← IR-agnostic public API (types + free functions)
    PerfModelIR.h      ← IR-aware factory layer (requires Triton MLIR headers)
  lib/TritonAMDGPUTransforms/
    PerfModel.cpp      ← hardware tables + model implementation
    PerfModelIR.cpp    ← factory function implementations
```

The split is deliberate:

| File | MLIR dependency | Usable from |
|---|---|---|
| `PerfModel.h/cpp` | None (only `llvm::StringRef/ArrayRef`) | Any pass, Python bindings, standalone tests, future non-GEMM characterisers |
| `PerfModelIR.h/cpp` | Triton dialect headers (already in every AMD pass) | Compiler passes only |

### 4.3 Public API

#### `PerfModel.h` — IR-agnostic core

**Types:**

```
Arch             — AMD architecture enum (CDNA1/2/3/4, RDNA3/4, GFX1250)
ElemKind         — coarse element-type category (FP64/32/16, BF16, FP8/6/4, I8)
HardwareInfo     — per-arch constants; HardwareInfo::get("gfx942") → ready instance
MfmaInstrInfo    — (mDim, nDim, kDim, throughputCycles, aKind, cKind)
GemmProblem      — M, N, K, batch, element types and bit-widths
TritonGemmConfig — blockM/N/K, numStages, numWarps, kWidth, kPack, mfmaNonKDim,
                   bypassLds, useAsyncCopy, wavesPerEu
PerfEstimate     — full breakdown: VGPR, LDS, waves, cycles, predicted TFLOPS,
                   validity flags (ldsExceeded, likelySpills, isComputeBound)
```

**`TritonGemmConfig` key fields and their canonical sources:**

| Field | Default | Canonical IR source |
|---|---|---|
| `blockM/N/K` | 128/128/32 | `RankedTensorType::getShape()` on dot result / A operand |
| `numWarps` | 4 | `ttg::lookupNumWarps(dotOp)` — reads `"ttg.num-warps"` module attr |
| `kWidth` | **0** | 0 = derived from MFMA table (`kBase × kPack`); or `DotOperandEncodingAttr::getKWidth()` post-pass |
| `kPack` | 1 | Pass option (1 or 2); baked into `kWidth` post-pass |
| `mfmaNonKDim` | 0 | 0 = auto via `selectMfmaNonKDim()`; or `AMDMfmaEncodingAttr::getInstrShape()[0]` post-pass |
| `useAsyncCopy` | true | Walk parent `scf.for` for `ttg::AsyncCopyGlobalToLocalOp` |
| `bypassLds` | false | Walk parent `scf.for` for absence of `ttg::LocalAllocOp` |

**Free functions:**

| Function | Replaces / augments |
|---|---|
| `estimateVgpr(prob, cfg, hw)` | The one hardcoded avoidance rule in BlockPingpong.cpp |
| `estimateLdsBytes(prob, cfg, hw)` | Formula-based LDS estimate; used pre-allocation |
| `estimateNumBuffers(cfg)` | Mirrors LowerLoops.cpp logic, exposed for callers |
| `estimatePerf(prob, cfg, hw)` | Full roofline + wave-quantisation estimate |
| `isValidConfig(prob, cfg, hw)` | Replaces the implicit "hope it fits" in several passes |
| `rankConfigs(prob, configs, hw)` | Pre-filters the autotuner search space |
| `selectMfmaNonKDim(prob, cfg, hw)` | Replaces the `min(M,N)` threshold in AccelerateAMDMatmul.cpp |

#### `PerfModelIR.h` — IR-aware factory layer

Populates `GemmProblem` and `TritonGemmConfig` directly from the IR, reading the
same attributes that the existing passes read.  Eliminates manual transcription
of IR values into struct fields by callers.

| Function | Description |
|---|---|
| `elemKindFromMlirType(Type)` | Convert MLIR element type to `ElemKind` |
| `gemmProblemFromDotOp(DotOp)` | Populate `GemmProblem` from tensor shapes and element types |
| `tritonConfigFromDotOpPre(DotOp, numStages, kPack)` | Populate config **before** AccelerateAMDMatmul; reads `ttg::lookupNumWarps`, detects async copy and bypassLds via loop walk; `kWidth=0` (derived lazily) |
| `tritonConfigFromDotOpPost(DotOp, numStages)` | Populate config **after** AccelerateAMDMatmul; additionally reads `AMDMfmaEncodingAttr::getInstrShape()` for `mfmaNonKDim` and `DotOperandEncodingAttr::getKWidth()` for the exact `kWidth` the pass chose |
| `ldsFromAllocation(FuncOp, ModuleAllocation, ...)` | Actual LDS bytes from `ModuleAllocation::getSharedMemorySize()`; falls back to `estimateLdsBytes()` if not yet run |
| `hardwareInfoFromModule(ModuleOp)` | Parse `"ttg.target"` attr (`"hip:gfx942"`) → `HardwareInfo` |

### 4.4 Model Architecture

#### Layer 1 — Hardware database

A static `ThroughputEntry` table maps `(Arch, mDim, nDim, aKind, cKind)` to
reciprocal MFMA throughput in cycles per instruction per SIMD.  Per-arch
`HardwareInfo` instances store CU count, SIMD width, VGPR budget, LDS capacity,
memory bandwidth, and clock frequency.

Values are validated against published peak TFLOPS figures.  Example:

```
MI210 (CDNA2): 1024 FLOP/cycle/CU × 1.7 GHz × 104 CUs ≈ 181 TFLOPS FP16  ✓
```

Architectures covered: CDNA1 (gfx908), CDNA2 (gfx90a), CDNA3 (gfx940/941/942),
CDNA4 (gfx950), RDNA3 (gfx11xx), RDNA4 (gfx12xx), GFX1250.

#### Layer 2 — Resource accounting

**kWidth derivation (`deriveKWidth`):**

A key finding from analysing `AccelerateAMDMatmul.cpp` (lines 679-704) is that
`kWidth` is **not** available from `AxisInfoAnalysis` — it is derived from the
chosen MFMA intrinsic:

```
kBase  = MfmaInstrInfo.kDim / mfmaNonKDim
           // mfma_32x32: kBase = kDim/2
           // mfma_16x16: kBase = kDim/4
           // mfma_4x4:   kBase = kDim/16
kWidth = kBase × kPack          (kPack ∈ {1, 2}, pass option)
```

`TritonGemmConfig.kWidth` defaults to 0, which triggers `deriveKWidth()` inside
`estimateVgpr()`.  Callers can override with the exact value from
`DotOperandEncodingAttr::getKWidth()` once `AccelerateAMDMatmul.cpp` has run.

**VGPR model:**

```
kWidth      = deriveKWidth(prob, cfg, hw)   // or cfg.kWidth if set explicitly

vgpr_accum  = ceil(blockM × blockN × cBytes / (waveSize × 4))
vgpr_a_frag = ceil(blockM × kWidth × aBytes / (waveSize × 4))
vgpr_b_frag = ceil(blockN × kWidth × bBytes / (waveSize × 4))
vgpr_misc   = 28  (empirical: pointers, loop vars, predicates)
total_vgpr  = roundUp(sum, vgprAllocGranule)
```

This correctly predicts the well-known CDNA spill boundary: a `128×128` FP16→FP32
tile on wave64 needs `128×128/64 = 256` accumulator VGPRs — the full CDNA1-3
budget — before accounting for anything else.

**LDS model (two sources):**

The model provides two LDS estimates depending on which pass is calling:

```
// Pre-allocation (formula-based estimate):
numBuffers = numStages          (async copy pipeline)
           = max(1, numStages−1) (synchronous pipeline)

ldsA = numBuffers × blockM × (blockK + 8) × aBytes  // 8-elem padding per row
ldsB = numBuffers × blockN × (blockK + 8) × bBytes

// Post-allocation (exact, via PerfModelIR.h):
actualLds = ModuleAllocation::getSharedMemorySize(funcOp)
            // includes composePaddedLayout() arch-specific padding
            // and layout-conversion scratch space
```

`ldsFromAllocation()` in `PerfModelIR.h` automatically selects the exact value
when available and falls back to the formula otherwise.

**Occupancy:**

```
wavesFromVgpr = floor(vgprPerSimd / vgprCount) × numSimdPerCU
wavesFromLds  = floor(ldsPerCU / ldsBytes) × numWarps
wavesPerCU    = min(wavesFromVgpr, wavesFromLds, maxWavesPerCU)
occupancy     = wavesPerCU / maxWavesPerCU
```

#### Layer 3 — Roofline + wave quantisation

```
// Per output tile, on one CU:
numMfma        = (blockM/mDim) × (blockN/nDim) × (blockK/kDim)
computeCycles  = numMfma × throughputCycles / numSimdPerCU

tileBytesAB    = blockM × blockK × aBytes + blockN × blockK × bBytes
memoryCycles   = tileBytesAB / (peakMemBwBytesPerCycle / numCUs)

// Software-pipeline overlap (fraction of memory latency hidden):
depthNeeded    = memoryCycles / computeCycles
pipelineOverlap = min(1, (numStages − 1) / depthNeeded)

effectiveCycles = max(computeCycles, memoryCycles × (1 − pipelineOverlap))

// Wave quantisation:
totalOutputTiles = ceil(M/blockM) × ceil(N/blockN) × batch
numWaves         = ceil(totalOutputTiles / numCUs)
waveEfficiency   = totalOutputTiles / (numWaves × numCUs)

// Predicted throughput:
totalCycles    = effectiveCycles × numWaves / (occupancy × waveEfficiency)
predictedTflops = 2×M×N×K×batch / (totalCycles / clockMHz×1e6) / 1e12
```

### 4.5 MFMA Instruction Size Selection

`selectMfmaNonKDim()` replaces the pure-threshold heuristic in
`AccelerateAMDMatmul.cpp::chooseMfmaInstruction()` (lines 172-196):

**Current logic:**
```cpp
if (minSize >= 32) → 32×32
if (minSize >= 16) → 16×16
if (minSize >= 4)  → 4×64 or 64×4
```

**Proposed logic:**
```
Try 32×32:
  - Intrinsic exists for (arch, aKind, cKind)?
  - estimateVgpr() with 32×32 ≤ hw.vgprPerSimd?
  → Accept if both true.
Try 16×16 (universal fallback, all CDNA/RDNA generations).
Try 4×4 (last resort, very small tiles).
```

This prevents the common failure mode where the heuristic selects 32×32 for a
`128×128` FP32 tile on CDNA3, consuming the entire 256-VGPR budget on
accumulators and forcing register spilling.

---

## 5. Integration Points

### 5.1 `AccelerateAMDMatmul.cpp` — MFMA size selection

Replace the threshold logic in `chooseMfmaInstruction()` (lines 172-196).  The
IR-aware factories populate the structs using the same attribute reads that the
pass already performs:

```cpp
// Before:
int minSize = std::min(shape[0], shape[1]);
enforcedNonKDim = (minSize >= 32) ? 32 : (minSize >= 16) ? 16 : 4;

// After (using PerfModelIR.h factories):
perf::HardwareInfo hw  = perf::hardwareInfoFromModule(dotOp->getParentOfType<ModuleOp>());
perf::GemmProblem  prob = perf::gemmProblemFromDotOp(dotOp);
// numStages and kPack come from pass options, as before:
perf::TritonGemmConfig cfg = perf::tritonConfigFromDotOpPre(dotOp, numStages, kPack);
enforcedNonKDim = perf::selectMfmaNonKDim(prob, cfg, hw);
```

`tritonConfigFromDotOpPre` reads `numWarps` via `ttg::lookupNumWarps(dotOp)`,
detects `useAsyncCopy` by walking the parent `scf.for`, and leaves `kWidth=0`
so that `deriveKWidth()` computes it from the intrinsic table — the same
`kBase × kPack` logic the pass itself uses.

### 5.2 `BlockPingpong.cpp` — Ping-pong scheduling thresholds

Replace the four hardcoded magic numbers (lines 1057-1250) with a model query:

```cpp
// Before:
constexpr int64_t smallTile = 16'777'216; // empirical
if (tileSize < smallTile) { /* one pattern */ } else { /* another */ }

// After:
auto est = perf::estimatePerf(prob, cfg, hw);
// Use est.isComputeBound, est.occupancy, est.waveEfficiency to select
// the scheduling pattern that best hides the identified bottleneck.
```

### 5.3 `LowerLoops.cpp` — LDS capacity guard

`LowerLoops.cpp` runs before `AllocateSharedMemory`, so `ModuleAllocation` is
not yet populated.  Use the formula-based estimate here; the exact allocation
check can be added as a post-`AllocateSharedMemory` verification pass later.

```cpp
// LowerLoops.cpp — after numBuffers is committed:
perf::HardwareInfo     hw   = perf::hardwareInfoFromModule(funcOp->getParentOfType<ModuleOp>());
perf::GemmProblem      prob = perf::gemmProblemFromDotOp(dotOp);
perf::TritonGemmConfig cfg  = perf::tritonConfigFromDotOpPost(dotOp, numStages);
// tritonConfigFromDotOpPost reads kWidth from DotOperandEncodingAttr and
// mfmaNonKDim from AMDMfmaEncodingAttr — both set by AccelerateAMDMatmul.

int ldsBytes = perf::estimateLdsBytes(prob, cfg, hw);
if (ldsBytes > hw.ldsPerCU) {
  dotOp.emitWarning("LDS usage (" + Twine(ldsBytes) +
                    " B) exceeds device capacity (" +
                    Twine(hw.ldsPerCU) + " B); "
                    "consider reducing num_stages or block sizes");
}

// Optional: post-AllocateSharedMemory exact check using PerfModelIR.h:
// int exactLds = perf::ldsFromAllocation(funcOp, moduleAlloc, prob, cfg, hw);
```

### 5.4 Python autotuner pre-filter (future)

Expose `isValidConfig()` and `estimatePerf()` through a thin Python binding so
that `triton.autotune` can skip provably-bad configs before benchmarking:

```python
from triton._C.libtriton import amd_perf_model as pm

valid_configs = [
    cfg for cfg in all_configs
    if pm.is_valid(M, N, K, dtype, cfg, arch)
]
ranked_configs = pm.rank(M, N, K, dtype, valid_configs, arch)
# Run benchmark only on top-K ranked configs.
```

---

## 6. What Is Explicitly Out of Scope

- **StreamK / persistent kernel scheduling** — the model assumes one CTA per
  output tile.  StreamK changes the wave structure significantly and warrants a
  separate model extension.

- **Flash Attention / chained dots** — the two-dot pipeline (`ChainedDotSchedule`)
  has additional constraints (warp layout forced to `{numWarps, 1}`, kWidth
  forced to kBase) that require a dedicated treatment.

- **Formocast-equivalent cycle-accurate simulation** — the roofline model is
  intentionally approximate.  A cycle-accurate Triton kernel simulator is a
  larger research project.

- **FP6 / FP4 microscaling** — the scaled MFMA instruction family is included
  in the hardware table but the resource model for scale-tensor LDS allocation
  is not yet implemented.

---

## 7. Implementation Plan

### 7.1 Line-count Analysis and PR Sizing

The sections below present precise line counts derived from the written files,
then use those counts to size each PR.  The target is **≤ 400 net new lines per
PR** — a threshold that keeps individual reviews focused and reflects the norms
of the Triton and LLVM upstream communities.

#### Existing written files (line counts by section)

**`PerfModel.h` (302 lines total)**

| Section | Lines |
|---|---|
| Hardware description — `Arch`, `HardwareInfo` | 69 |
| MFMA types — `ElemKind`, `MfmaInstrInfo`, `getMfmaInstrInfo` | 41 |
| GEMM problem + `TritonGemmConfig` | 73 |
| `PerfEstimate` struct | 36 |
| Public API function declarations | 53 |
| File boilerplate (includes, guards, namespace) | 30 |

**`PerfModel.cpp` (752 lines total)**

| Section | Lines |
|---|---|
| Hardware DB — `archFromString`, `HardwareInfo::get`, `peakFlops` | 173 |
| MFMA table — `ElemKind` helpers, `kMfmaThroughputTable`, `getMfmaInstrInfo` | 139 |
| Resource accounting — `estimateNumBuffers`, `deriveKWidth`, `estimateVgpr`, `estimateLdsBytes` | 110 |
| Roofline — `estimatePerf` | 183 |
| Ranking — `isValidConfig`, `rankConfigs` | 66 |
| `selectMfmaNonKDim` | 38 |
| File boilerplate | 43 |

**`PerfModelIR.h` (139 lines total)**

| Section | Lines |
|---|---|
| `elemKindFromMlirType` | 8 |
| `gemmProblemFromDotOp` | 18 |
| `tritonConfigFromDotOpPre` | 20 |
| `tritonConfigFromDotOpPost` | 15 |
| `ldsFromAllocation` | 12 |
| `hardwareInfoFromModule` | 12 |
| File boilerplate | 54 |

**`PerfModelIR.cpp` (244 lines total)**

| Section | Lines |
|---|---|
| `elemKindFromMlirType` | 28 |
| `gemmProblemFromDotOp` | 35 |
| Loop-walk helpers (`hasAsyncCopyInLoop`, `isBypassLds`) | 36 |
| `tritonConfigFromDotOpPre` | 38 |
| `tritonConfigFromDotOpPost` | 43 |
| `ldsFromAllocation` | 16 |
| `hardwareInfoFromModule` | 14 |
| File boilerplate | 34 |

#### Existing pass files — lines affected

| Pass file | Section touched | Lines deleted | Lines added | Net |
|---|---|---|---|---|
| `AccelerateAMDMatmul.cpp` | `chooseMfmaInstruction` threshold block (lines 172-196) | −25 | +2 includes +5 call +10 remark = +17 | **−8** |
| `LowerLoops.cpp` | After `numBuffers` commit | 0 | +1 include +15 guard = +16 | **+16** |
| `BlockPingpong.cpp` | Tile-size threshold block (lines 1057-1250) | −211 | +1 include +30 model call = +31 | **−180** |
| `python/triton_amd.cc` | New binding functions | 0 | +60 | **+60** |
| `CMakeLists.txt` | Source list | 0 | +2 | **+2** |

#### Estimated test files

| Test file | Est. lines |
|---|---|
| `perf-model-hardware.mlir` — hardware lookup unit tests | 60 |
| `perf-model-resource.mlir` — VGPR/LDS boundary tests | 80 |
| `perf-model-roofline.mlir` — throughput and ranking tests | 80 |
| `perf-model-ir.mlir` — IR factory round-trip tests | 100 |
| `accelerate-amd-matmul-perf-model.mlir` — MFMA selection regression | 80 |
| `lower-loops-lds-guard.mlir` — LDS overflow guard tests | 60 |
| `block-pingpong-perf-model.mlir` — scheduling pattern tests | 80 |
| `test_perf_model.py` — Python binding unit tests | 60 |

---

### 7.2 PR Sizing Assessment

The original 6-PR plan has **PR 1 at ~1,700 lines** — too large for upstream.
The root cause is shipping all four new files in one PR.

Splitting the new files by functional layer gives eight PRs, each at or under
~400 net new lines.  PRs that touch only existing files (PRs 5–7) are small by
construction.

| PR | Content | New lines | Modified lines (net) | Test lines | **Total** |
|---|---|---|---|---|---|
| **PR 1** | Hardware tables | 312 | +2 CMake | 60 | **~375** |
| **PR 2** | Resource accounting + `selectMfmaNonKDim` | 272 | 0 | 80 | **~355** |
| **PR 3** | IR pre-pass factory + `AccelerateAMDMatmul` | 223 | −8 | 80+80 | **~375** |
| **PR 4** | Roofline + ranking | 355 | 0 | 80 | **~435** |
| **PR 5** | IR post-pass factory + `LowerLoops` | 101 | +16 | 60 | **~180** |
| **PR 6** | `BlockPingpong` model-driven scheduling | 0 | −180 | 80 | **~−100** *(net deletion)* |
| **PR 7** | Python binding | 0 | +60 | 60 | **~120** |
| **PR 8** | Calibration | ~30 | ~10 | 50 | **~90** |

> **Note on PR 6:** Replacing 211 lines of threshold code with 31 lines of
> model calls is a net deletion of ~180 lines — a compelling case for reviewers
> that the change is a simplification, not an addition.

---

### 7.3 Revised PR Dependency Graph

```
PR 1 — Hardware tables  (HardwareInfo, ElemKind, MFMA throughput table)
  │
  └── PR 2 — Resource accounting  (VGPR, LDS, selectMfmaNonKDim)
        │
        └── PR 3 — IR pre-pass factory + AccelerateAMDMatmul  [perf]
              │
              └── PR 4 — Roofline + ranking  (estimatePerf, rankConfigs)
                    │
                    ├── PR 5 — IR post-pass factory + LowerLoops  [correctness]
                    └── PR 6 — BlockPingpong model-driven         [perf]

PR 7 — Python binding  (depends on PR 2 only)      [autotuning speed]
PR 8 — Calibration    (depends on PR 1–7 merged)   [accuracy]
```

PRs 5, 6, and 7 can be worked on in parallel after their shared prerequisites
land.  None of them touch the same existing files.

---

### 7.4 PR 1 — Hardware Tables  *(~375 lines, pure addition)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModel.h` (new, partial) — sections: `Arch`, `HardwareInfo`, `ElemKind`, `MfmaInstrInfo` | 110 |
| `PerfModel.cpp` (new, partial) — sections: hardware DB, MFMA table, ElemKind helpers | 312 |
| `CMakeLists.txt` | +2 |
| `test/TritonGPU/amd/perf-model-hardware.mlir` (new) | 60 |

No existing files modified.  No model logic — purely data tables and type
definitions.  Reviewers can verify the MFMA throughput numbers against AMD ISA
documentation independently.

**Demonstrated benefit:** Tests confirm `HardwareInfo::get("gfx942")` returns
`numCUs=228`, `ldsPerCU=65536`, `vgprPerSimd=256` and
`getMfmaInstrInfo(CDNA3, 32, 32, FP16, FP32)` returns `throughputCycles=64`.

---

### 7.5 PR 2 — Resource Accounting  *(~355 lines, pure addition)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModel.h` additions — `GemmProblem`, `TritonGemmConfig`, `PerfEstimate` (partial), `estimateVgpr`, `estimateLdsBytes`, `estimateNumBuffers`, `selectMfmaNonKDim` | 162 |
| `PerfModel.cpp` additions — `estimateNumBuffers`, `deriveKWidth`, `estimateVgpr`, `estimateLdsBytes`, `selectMfmaNonKDim` | 196 |
| `test/TritonGPU/amd/perf-model-resource.mlir` (new) | 80 |

No existing files modified.

**Demonstrated benefit:** Tests prove the VGPR spill boundary:
`estimateVgpr({blockM=128, blockN=128, cBits=32}, gfx942) ≥ 256` →
`likelySpills=true`.  `selectMfmaNonKDim` correctly returns 16 (not 32) for
128×128 FP32 on CDNA3.

---

### 7.6 PR 3 — IR Pre-Pass Factory + `AccelerateAMDMatmul`  *(~375 lines)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModelIR.h` (new, partial) — `elemKindFromMlirType`, `gemmProblemFromDotOp`, `tritonConfigFromDotOpPre`, `hardwareInfoFromModule` | 83 |
| `PerfModelIR.cpp` (new, partial) — same four functions + loop-walk helpers | 141 |
| `CMakeLists.txt` | +2 (PerfModelIR.cpp source entry + dep) |
| `AccelerateAMDMatmul.cpp` | −25 threshold, +17 model call + remark = **−8 net** |
| `test/TritonGPU/amd/perf-model-ir-pre.mlir` (new) | 80 |
| `test/TritonGPU/amd/accelerate-amd-matmul-perf-model.mlir` (new) | 80 |

**Guard type:** `emitRemark` (soft).

```cpp
// AccelerateAMDMatmul.cpp — replaces lines 172-196:
#include "TritonAMDGPUTransforms/PerfModelIR.h"
auto hw   = perf::hardwareInfoFromModule(dotOp->getParentOfType<ModuleOp>());
auto prob = perf::gemmProblemFromDotOp(dotOp);
auto cfg  = perf::tritonConfigFromDotOpPre(dotOp, numStages, kPack);
enforcedNonKDim = perf::selectMfmaNonKDim(prob, cfg, hw);

if (int vgpr = perf::estimateVgpr(prob, cfg, hw); vgpr > hw.vgprPerSimd)
  mlir::emitRemark(dotOp.getLoc())
      << "MFMA " << enforcedNonKDim << "x" << enforcedNonKDim
      << " requires ~" << vgpr << " VGPRs (limit " << hw.vgprPerSimd << ")";
```

**Demonstrated benefit:** `--verify-diagnostics` test confirms 128×128 FP32
selects 16×16 MFMA on gfx942 and emits the spill remark.  Benchmark shows
throughput improvement for large FP32 tiles that previously spilled.

---

### 7.7 PR 4 — Roofline + Ranking  *(~435 lines, pure addition)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModel.h` additions — `PerfEstimate` (complete), `estimatePerf`, `isValidConfig`, `rankConfigs` | 105 |
| `PerfModel.cpp` additions — `estimatePerf`, `isValidConfig`, `rankConfigs` | 282 |
| `test/TritonGPU/amd/perf-model-roofline.mlir` (new) | 80 |

No existing files modified.  This is the largest PR but still pure addition.
`estimatePerf` at 183 lines is the single biggest function; if needed it can
be split into a further sub-PR by shipping `isValidConfig` + `rankConfigs` (66
lines) separately, bringing each sub-PR under 300 lines.

**Demonstrated benefit:** Tests verify `predictedTflops` is within 20% of the
MI210 published peak for a representative 128×128×K FP16 GEMM.
`rankConfigs` places the LDS-overflowing config last in the sorted output.

---

### 7.8 PR 5 — IR Post-Pass Factory + `LowerLoops`  *(~180 lines)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModelIR.h` additions — `tritonConfigFromDotOpPost`, `ldsFromAllocation` | 35 |
| `PerfModelIR.cpp` additions — same two functions | 59 |
| `LowerLoops.cpp` | +1 include, +15 guard = **+16 net** |
| `test/TritonGPU/amd/lower-loops-lds-guard.mlir` (new) | 60 |

**Guard type:** `emitError` + `signalPassFailure()` (hard).

```cpp
auto cfg = perf::tritonConfigFromDotOpPost(dotOp, numStages);
int  lds = perf::estimateLdsBytes(prob, cfg, hw);
if (lds > hw.ldsPerCU) {
  dotOp.emitError() << "LDS usage " << lds << " B exceeds limit "
                    << hw.ldsPerCU << " B";
  return signalPassFailure();
}
```

**Demonstrated benefit:** `expected-error` lit test confirms `num_stages=6` on
128×128 FP16 (CDNA3) now fails at compile time rather than producing a silently
mis-executing kernel.

---

### 7.9 PR 6 — `BlockPingpong` Model-Driven Scheduling  *(net −100 lines)*

**Files changed:**

| File | Lines |
|---|---|
| `BlockPingpong.cpp` | −211 threshold block, +1 include +30 model call = **−180 net** |
| `test/TritonGPU/amd/block-pingpong-perf-model.mlir` (new) | 80 |

No new files added.  This PR is a net code deletion, replacing 211 lines of
magic-number branches with 30 lines of model queries.

**Guard type:** `emitRemark` + early return (soft).

**Demonstrated benefit:** FileCheck test confirms the correct scheduling
pattern (`rocdl.s.setprio 1` vs `rocdl.sched.barrier`) is chosen for
memory-bound vs compute-bound shapes.  Benchmark shows improvement where the
old thresholds misclassified the workload.

---

### 7.10 PR 7 — Python Autotuner Pre-filter  *(~120 lines)*

**Files changed:**

| File | Lines |
|---|---|
| `python/triton_amd.cc` | +60 (two new binding functions) |
| `third_party/amd/python/test/test_perf_model.py` (new) | 60 |

Exposes only `PerfModel.h/cpp` (IR-agnostic).  `PerfModelIR` stays
compiler-side and is never bound to Python.

**Demonstrated benefit:** Python tests confirm known-invalid configs
(LDS overflow, VGPR spill) return `is_valid=False`; autotuner wall-clock time
reduces measurably by skipping benchmark runs for those configs.

---

### 7.11 PR 8 — Calibration  *(~90 lines)*

**Files changed:**

| File | Lines |
|---|---|
| `PerfModel.cpp` | ~10 lines changed (tune `vgprMisc` constant and pipeline overlap formula) |
| `third_party/amd/python/tools/calibrate_perf_model.py` (new) | ~80 |

**Tuning targets:**

| Constant | Tuning method |
|---|---|
| `vgprMisc = 28` | Compare `estimateVgpr()` against `; NumVgprs:` from AMDGCN assembly across 20+ configs |
| Pipeline overlap formula | Compare predicted vs. measured on tall/wide GEMMs on gfx90a, gfx942, gfx1100 |

**Accuracy target:** `predictedTflops` within ±20% of measured for ≥80% of
tested configurations across square, tall, wide, batch, and non-power-of-two
shapes.

---

## 8. Expected Impact

| Metric | Current | With PerfModel |
|---|---|---|
| MFMA size selection correctness for non-square tiles | Threshold-based, fails for large FP32 tiles | Model-driven, VGPR-aware |
| `kWidth` accuracy in VGPR model | Hardcoded 8 (pass parameter, misused as model input) | Derived from MFMA intrinsic table (`kBase × kPack`), matching the pass's own logic |
| LDS estimate accuracy | Formula with fixed 8-element padding | Pre-pass: same formula; post-`AllocateSharedMemory`: exact arch-specific value from `ModuleAllocation` |
| Struct population in callers | Manual IR value transcription, error-prone | Factory functions read canonical IR sources (`ttg::lookupNumWarps`, encoding attrs, `"ttg.target"`) |
| Silent LDS overflows | Possible (no guard) | Caught at compile time via `tritonConfigFromDotOpPost` + `estimateLdsBytes` |
| Autotuner configs evaluated before pruning | All (O(N)) | Only valid + top-ranked (O(K), K ≪ N) |
| Autotuner / Python binding dependency | None | None (`PerfModel.h/cpp` is IR-agnostic; only compiler passes use `PerfModelIR`) |
| Origami dependency | N/A | None |

The most immediately measurable improvement is the elimination of silent LDS
overflows and the VGPR-spill failure mode for large FP32 tiles on CDNA1-3
hardware.  The IR-aware factory layer additionally removes the maintenance
burden of callers having to manually mirror attribute-read logic that already
exists in `AccelerateAMDMatmul.cpp`.

---

## 9. References

- Origami source: `rocm-libraries/shared/origami/`
- tritonBLAS paper: [arXiv:2512.04226](https://arxiv.org/abs/2512.04226)
- `AccelerateAMDMatmul.cpp`: `third_party/amd/lib/TritonAMDGPUTransforms/`
- `PerfModel.h/cpp`: `third_party/amd/include|lib/TritonAMDGPUTransforms/` (IR-agnostic core)
- `PerfModelIR.h/cpp`: `third_party/amd/include|lib/TritonAMDGPUTransforms/` (IR-aware factory layer)
- AMD ISA documentation: [CDNA3 ISA Reference](https://gpuopen.com/amd-cdna3-white-paper/)
- Roofline model: Williams et al., "Roofline: An Insightful Visual Performance
  Model", CACM 2009
- `AxisInfo.h`, `Allocation.h`, `Utility.h`: `include/triton/Analysis/` — MLIR
  analyses queried by `PerfModelIR`

---

## 10. Long-Term Vision: Generalising to Non-GEMM Kernels

### 10.1 Motivation

The analytical model described in Sections 3–8 targets GEMM exclusively.  However,
the Transformer workloads that motivate Triton's existence contain a substantial
fraction of non-GEMM compute: RMSNorm, LayerNorm, Softmax, RoPE, element-wise
activations, and attention score reductions.  These kernels are almost entirely
memory-bandwidth-bound on AMD hardware, meaning a simple roofline estimate would
already capture most of the optimisation signal — if the model knew how to
characterise them from the IR.

A parallel investigation into Origami's source (`gemm.cpp`, `hardware.hpp`,
`heuristics.hpp`) and Triton's MLIR analysis infrastructure (`AxisInfo.h`,
`Allocation.h`, `Utility.h`, `RangeAnalysis.cpp`) reveals that the path to a
general kernel performance model is both technically feasible and architecturally
natural.

### 10.2 Key Finding: Origami's Model Has a Generic Core

Reading `gemm.cpp` in full, Origami's latency engine splits into two halves with
very different generality:

| Half | What it computes | Kernel-specific? |
|---|---|---|
| **Memory hierarchy engine** | `bytes → L2 / MALL / DRAM latency` via bandwidth-occupancy model | No — pure hardware physics |
| **Workload characterisation** | Bytes per tile, A/B cache-reuse hit rates, MFMA instruction count | Yes — hardcoded for GEMM A/B/C tiles |

The memory hierarchy engine is **entirely reusable**.  Its four inputs are:

```
bytes_per_block   — how many bytes does one CTA read/write?
cache_hit_rates   — what fraction is served from L2 vs. MALL vs. DRAM?
num_blocks        — how many CTAs are in the grid?
hardware_t        — the GPU itself
```

For a GEMM, Origami computes those four inputs from M/N/K tile shapes and the
A/B reuse pattern.  For any other kernel, different kernel-specific logic
produces the same four inputs, and the rest of the model runs unchanged.

The following components from Origami are directly reusable without modification:

- **`hardware_t`** — all fields except `parallel_mi_cu` and `INSTRUCTION_MAP`
- **`compute_mem_bw_from_occupancy()`** — the quadratic BW = f(active\_CUs) model,
  derived from microbenchmarks and not workload-dependent
- **Occupancy decay**: `pow(occupancy_decay_base, waves_per_cu)` — generic GPU
  latency-hiding model
- **Memory hierarchy formula**:
  `L_mem = max(bytes_L2 / BW_L2,  miss_bytes / BW_MALL,  miss_miss_bytes / BW_DRAM)`
- **Wave quantisation**: `num_waves = ceil(num_blocks / N_CU)`,
  `efficiency = num_blocks / (num_waves × N_CU)` — identical for any kernel

### 10.3 Key Finding: Triton's IR Provides the Characterisation Inputs

A thorough audit of Triton's MLIR analysis infrastructure found that all four
inputs the memory hierarchy engine needs can be derived **directly from the IR**
for any kernel, using analyses that already exist:

| Required input | Triton IR source (already available) |
|---|---|
| `bytes_per_block` | Walk `tt.load` / `tt.store` ops; multiply `RankedTensorType::getShape()` × `elemBits/8` |
| Vectorisation / coalescing | `AxisInfoAnalysis::getContiguity(dim)` — already used by the pipeliner |
| `smem_bytes_per_block` | `ModuleAllocation::getSharedMemorySize()` — already computed for every kernel |
| Reduction scratch | `ReduceOpHelper::getScratchSizeInBytes()` — already exists in `Utility.h` |
| Op mix (flops vs. loads) | Walk `arith.*` / `math.*` / `tt.dot` / `tt.reduce` ops |
| Loop trip count | `RangeAnalysis::getTotalLoopTripCount()` — AMD-specific, already used |
| Grid dimensions | `GetProgramIdOp` range from `RangeAnalysis` |

The gap is precisely what a general cost model would fill: there is currently no
op-level cycle model for non-GEMM operations and no bandwidth model for non-GEMM
memory patterns.  The infrastructure to *query* the needed information exists in
full; what is missing is the *model* that consumes it.

### 10.4 Proposed Architecture: `KernelPerfModel`

The generalisation adds one abstraction between the hardware engine and the
pass-level callers: a **`KernelProfile`** struct that decouples the characterisation
of a kernel from the physics of predicting its latency.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hardware Engine  (reused / extended from        │
│                     Origami + existing PerfModel HardwareInfo)      │
│                                                                     │
│   memory_hierarchy_latency(profile, hw)                             │
│   compute_mem_bw_from_occupancy(active_cus, hw)   ← from Origami   │
│   occupancy_from_smem(smem_bytes, hw)                               │
│   wave_quantisation(num_blocks, hw)                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ consumes
                   ┌────────▼────────┐
                   │  KernelProfile  │
                   │                 │
                   │ bytesRead       │
                   │ bytesWritten    │
                   │ flopsPerBlock   │
                   │ smemBytes       │
                   │ l2HitRate       │
                   │ mallHitRate     │
                   │ numBlocks       │
                   │ hasReduction    │
                   └────────▲────────┘
                            │ produced by
┌───────────────────────────┴─────────────────────────────────────────┐
│                    Kernel Characterisers                             │
│             (one per class; walk Triton MLIR IR)                    │
│                                                                     │
│   GemmCharacteriser         — A/B/C tile bytes, MFMA flops, grid   │
│   ElementwiseCharacteriser  — load/store bytes, VALU flops, grid   │
│   ReductionCharacteriser    — input bytes, smem comms, grid        │
│   FusedKernelCharacteriser  — combines above per op region         │
└─────────────────────────────────────────────────────────────────────┘
```

The `KernelProfile` struct:

```cpp
struct KernelProfile {
  double bytesReadPerBlock;    ///< Global memory reads per CTA
  double bytesWrittenPerBlock; ///< Global memory writes per CTA
  double flopsPerBlock;        ///< Floating-point ops per CTA
  int    smemBytesPerBlock;    ///< Shared memory bytes per CTA
  double l2HitRate;            ///< Estimated L2 hit fraction   [0..1]
  double mallHitRate;          ///< Estimated MALL hit fraction [0..1]
  int64_t numBlocks;           ///< Total grid size
  bool    hasReduction;        ///< Cross-warp reduction present?
};
```

The hardware engine's entry point becomes:

```cpp
PerfEstimate estimatePerfFromProfile(const KernelProfile &profile,
                                     const HardwareInfo  &hw);
```

This is the same roofline + wave-quantisation computation already implemented in
`PerfModel.cpp`, generalised to accept pre-computed profile inputs instead of
GEMM-specific tile shapes.

### 10.5 How Each Non-GEMM Kernel Class Maps

#### Elementwise kernels (RoPE, GELU, element-wise scale)

These are the simplest case and the most bandwidth-bound.

```
bytesRead   = Σ shape(load) × elemBytes  over all tt.load ops
bytesWrite  = Σ shape(store) × elemBytes over all tt.store ops
flops       = count arith.mulf / arith.addf / math.exp / …  ops
l2HitRate   ≈ 0  (streaming — working set >> L2 for typical batch/seq)
numBlocks   = product of program_id grid dimensions (from RangeAnalysis)
smemBytes   = 0  (no cross-warp communication)
```

The model immediately degenerates to the pure memory-bandwidth limit:
`latency ≈ bytesTotal / peakBandwidth`, with wave-quantisation applied.
This is exactly the regime where even a rough model is useful, because the
current backend has no way to detect that a kernel is bandwidth-bound and
therefore `num_stages > 2` provides no benefit.

#### Reduction kernels (RMSNorm variance, Softmax row-max/sum, LayerNorm mean)

```
bytesRead   = input tensor bytes  (one full pass over the reduction dimension)
bytesWrite  = output tensor bytes  (much smaller — one scalar per row)
flops       = hidden_dim × ops_per_element  (adds, multiplies)
smemBytes   = ReduceOpHelper::getScratchSizeInBytes()  ← already computed
l2HitRate   ≈ 1.0  if  hidden_dim × elemBytes < l2SizeBytes / numCUs
              ≈ 0.0  otherwise   (working set does not fit in L2 per CU)
numBlocks   = batch × seq_len   (one CTA per row)
hasReduction = true
```

The L2 hit rate estimate is the most impactful term here.  For typical LLM
hidden dimensions (4096–16384 elements of BF16 = 8–32 KB), the working set
fits comfortably within the per-CU L2 share on CDNA3 (256 MB / 228 CUs ≈
1.1 MB/CU), meaning reductions are effectively L2-bandwidth-bound rather than
DRAM-bandwidth-bound — a meaningful distinction for predicting performance.

#### Fused kernels (full RMSNorm, full Softmax, FlashAttention score rescaling)

A fused kernel combines multiple phases.  The characteriser walks the `scf.for`
body and accumulates contributions from each op:

```cpp
// Pseudocode for FusedKernelCharacteriser::characterise(scf::ForOp loop):
KernelProfile profile = {};
for (Operation &op : loop.getBody()->without_terminator()) {
  if (auto load = dyn_cast<tt::LoadOp>(&op))
    profile.bytesReadPerBlock += shapeBytes(load.getResult());
  else if (auto store = dyn_cast<tt::StoreOp>(&op))
    profile.bytesWrittenPerBlock += shapeBytes(store.getValue());
  else if (auto reduce = dyn_cast<tt::ReduceOp>(&op)) {
    profile.smemBytesPerBlock += ReduceOpHelper(reduce).getScratchSizeInBytes();
    profile.hasReduction = true;
  } else if (isa<arith::ArithDialect, math::MathDialect>(op.getDialect()))
    profile.flopsPerBlock += estimateFlops(&op);
}
profile.numBlocks = inferGridSize(loop, rangeAnalysis);
profile.l2HitRate = estimateL2HitRate(profile.bytesReadPerBlock, hw);
```

This walk is entirely mechanical and uses only existing IR traversal APIs.

### 10.6 Integration with Triton's MLIR Pass Infrastructure

The characterisers are best implemented as lightweight **analysis passes** that
run ahead of the scheduling and lowering passes.  They require:

- `AxisInfoAnalysis` — already run by `ScheduleLoops` and `CoalesceAsyncCopy`
- `ModuleAllocation` — already run by `AllocateSharedMemory`
- `ReduceOpHelper` — already used by reduction lowering
- `RangeAnalysis` (AMD) — already run for buffer ops optimisation

No new MLIR analyses are needed.  The characteriser is a pass that queries
existing analysis results and populates a `KernelProfile` attached to the
function as an MLIR attribute or passed directly to subsequent passes.

#### Immediate use: `num_stages` guidance for non-GEMM loops

The most actionable near-term use is detecting when pipelining a non-GEMM loop
is futile.  Currently the compiler pipelines any loop with `num_stages > 1`,
even purely compute-bound loops where the added register pressure hurts
occupancy without hiding any memory latency.

```cpp
// In ScheduleLoops.cpp, before pipelining a non-dot loop:
KernelProfile profile = characterise(loop, axisInfo, allocation);
PerfEstimate est = estimatePerfFromProfile(profile, hw);
if (est.isComputeBound) {
  // Pipelining hides memory latency — not the bottleneck here.
  // Warn or skip pipelining to preserve occupancy.
  loop.emitRemark("loop is compute-bound; num_stages > 1 unlikely to help");
}
```

### 10.7 Kernel Expansion Roadmap

The natural implementation sequence, from simplest to most complex:

| Phase | Kernel class | New characteriser | Primary benefit |
|---|---|---|---|
| **A** | Elementwise (RoPE, GELU, scale) | `ElementwiseCharacteriser` | Detect BW-bound, skip redundant pipelining |
| **B** | Single-axis reduction (RMSNorm var, Softmax max/sum) | `ReductionCharacteriser` | L2 hit rate model, occupancy from smem |
| **C** | Fused elementwise + reduction (full RMSNorm, full Softmax) | `FusedKernelCharacteriser` | Predict optimal `num_stages`, block size |
| **D** | Scan (cumsum, prefix ops) | `ScanCharacteriser` | Scratch-size-aware occupancy |
| **E** | Fused attention (FlashAttention score rescaling) | extends `GemmCharacteriser` | Model the dot + softmax fusion cost |

Phases A and B can proceed in parallel with the GEMM integration work (Sections
7.2–7.5) since they share only the hardware engine, not the GEMM-specific code.

### 10.8 Relationship to Origami Long-Term

This architecture positions the Triton performance model as a **superset** of
what Origami provides for the TensileLite case, rather than a competing
implementation:

- Origami remains the authoritative model for hipBLASLT / TensileLite kernels,
  where its Formocast simulator and TensileLite-specific heuristics are
  indispensable.
- The Triton model reuses Origami's hardware constants and memory hierarchy
  physics (the generic core), while substituting Triton-specific characterisers
  for the workload-specific inputs.
- Over time, the hardware engine could be factored into a shared AMD GPU
  physics library consumed by both, eliminating the current duplication of
  hardware constant tables between the two codebases.
