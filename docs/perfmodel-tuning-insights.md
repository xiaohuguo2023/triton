# PerfModel Tuning Insights (gfx950 / MI355X, Gluon GEMM)

Captured during the regression-fix work for the tutorial shape sweep. Each
insight is grounded in empirical rocprofv3 measurements and HIP-event
bench results, not intuition.

> **See also — [`perf-model-saturation-physics.md`](perf-model-saturation-physics.md)**
> (2026-07): a beginner-friendly, worked-example explainer of the current model.
> It re-derives tile selection (block-K, num_warps, num_stages, occupancy) from
> Little's-law saturation + clean-wave MFMA efficiency, and **supersedes Insight 3
> below** (the linear per-K-iter overhead + `BK<64 ×1.30` penalty) and the
> `stallAmp`/occupancy heuristics. Overall fp16 win rate 83% → 97%. Read that doc
> first for the *why*; this file is the terse per-fix history.

---

## Insight 0 — Measurement methodology: the same kernel reads 2× apart

Before trusting ANY number, know how it was measured. The same
`4096×4096×1536` `BM=256²` config reads ~2× apart across methods:

| Method | Reports | Why |
|---|---:|---|
| TA `tuning_cache.json` | 1006 TF | inner tuning loop, launch-amortized — NOT reproducible |
| TA `gemm_bench.py --benchmark` (same cfg) | 563 TF | TA's own harness; rotating-tensor + icache-flush per iter |
| rocprofv3 launch loop (isolated GPU) | 787 TF | per-dispatch kernel time, clean GPU — **ground truth** |
| do_bench on contended GPU | 500 TF | co-resident vLLM (−40%) + do_bench underestimate (−30%) |

**Three confounds that stack into a phantom "2× kernel gap":**
1. **GPU contamination** — a co-resident vLLM server costs 40–70%. ALWAYS
   `rocm-smi --showuse --showpids` and confirm the GPU is idle before benching.
2. **do_bench underestimate** — undercounts GEMM TFLOPS up to ~30% at these
   shapes; use rocprofv3 `--kernel-trace`.
3. **TA tuning_cache inflation** — cached TFLOPS read ~25–80% above what TA's
   own `--benchmark` reproduces. Cache = ranking signal, NOT absolute wall-clock.

**Rule:** PerfModel validation uses rocprofv3 kernel-trace on a verified-idle
GPU. TA `tuning_cache` is used only for relative config ranking.

---

## Modeling principle: hardware-dependent, not compiler-dependent

When calibrating PerfModel terms against TensorAtlas exhaustive tuning,
TA gives us **the current Triton compiler's best achievable performance**,
not **the theoretical hardware ceiling**. These two can drift apart:

  * **Fundamental** (hardware-rigid, persists across compiler versions):
    HBM/L2/LDS bandwidth, MFMA instruction throughput and shape, VGPR/LDS
    capacity per CU, wave occupancy limits, per-CTA dispatch overhead.
    These are constants of the silicon; no compiler change will move them.

  * **Compiler-residue** (Triton-version-dependent, expected to improve
    over time): K-loop unroll quality, waitcnt placement density, register
    allocation tightness, address-arithmetic instruction count, mfma
    pingpong scheduling, async-copy issue interleaving.

**The rule**: PerfModel should encode the fundamental terms. Compiler-
residue gaps should be tracked separately (in the validation harness) so
they're visible but not baked into the cost model — when Triton improves,
the model should still rank correctly without re-calibration.

**Why this matters**: if we add a penalty that captures "today's PM-pick
runs 5% slower than today's TA-pick due to compiler-side overhead," we
implicitly assume that overhead is permanent. When the compiler improves,
our model becomes too pessimistic on the formerly-loser config and
mis-ranks it the other direction.

### Case study — 16×24576×1536 (LARGE_N skinny)

Decomposition of the 14% PM-vs-TA wall-clock gap (TA reference here is the
rocprofv3 ground truth, not the inflated tuning_cache — see Insight 0):

| component                                                            | type                | est. magnitude | model? |
|---|---|---:|---|
| Sub-mfma-tile lane waste (rigid 16×16 mfma, per-warp tile `[4,32]`)  | **fundamental**     | ~5-7%          | YES (Fix 6a) |
| Extra LDS bw consumption from 3× more `ds_read` per useful output    | **fundamental** (LDS = 128 B/cyc/CU on gfx950, fixed) | ~3-5% | YES (Fix 8) |
| `s_waitcnt` placement, K-loop unroll quality, register allocation    | **compiler-residue** | ~3-5%         | NO          |
| Address-arithmetic count (PM has 17× `v_lshlrev` vs TA's 5)          | mostly compiler      | ~1-2%         | NO          |

So roughly **10% fundamental + 5% compiler-residue**. Fix 8 adds the
LDS-cycles term to capture the fundamental piece; we explicitly DO NOT
add an `alpha × (1/mfmaUtil − 1)` post-max overhead term that would
capture the compiler-residue piece, because that 5% is what Triton
should fix in its scheduler, not what PerfModel should bake in.

### Validation-harness side: surface the compiler-residue gap

`scripts/perf_baseline/pm_vs_measured.py` should report three numbers
per shape:

  * **fundamental_floor** — analytical roofline from HBM/LDS/compute peaks
  * **ta_best_tflops**    — what Triton compiler achieves today (TA exhaustive)
  * **predicted_tflops**  — PerfModel's prediction

The gap `(fundamental_floor − ta_best)` IS the compiler residue. When
this gap shrinks over Triton releases, that's a positive sign — and a
trigger to verify our fundamental-only model still ranks correctly
against the now-tighter TA reference.

---

## Insight 1 — `stallAmp` must use LDS-aware wave count, not VGPR-only

**Problem:** PerfModel picked (128,128,128) over (128,128,64) at 3072³,
losing 28% to empirical winner.

**Cause:** `stallAmp` was indexed by `est.wavesPerSimd`, which is computed
purely from VGPR pressure. Both BK=64 and BK=128 of (128,128,*) have
VGPR=108 → both got `wavesPerSimd=4` → both got `stallAmp=1.0`. Model
treated them identically.

**Reality:** LDS for (128,128,128) at numStages=2 is 144 KB → only 1 CTA
fits per CU → 4 resident waves → **1 wave/SIMD**.
LDS for (128,128,64) is 64 KB → 2 CTAs/CU → 8 resident waves → **2 waves/SIMD**.

**Fix:** use the LDS-AND-VGPR-aware count:
```cpp
const double effectiveWavesPerSimd = wavesPerCU / numSimdPerCU;
```
where `wavesPerCU = min(wavesFromVgpr, wavesFromLds, max)`.

**Effect:** (128,128,128) now gets `stallAmp=1.5×`, (128,128,64) gets `1.0×`
(after threshold change in Insight 2). Picks flip correctly.

---

## Insight 2 — `stallAmp` threshold should be 2 waves/SIMD, not 3

**Problem:** With threshold=3, configs with `wavesPerSimd ∈ {1,2,3}` all
got penalties — including (128,128,64) (2 waves/SIMD) which is the
empirical winner. Meanwhile (128,128,32) with 4 waves/SIMD got no penalty
and won.

**Reality:** Empirically (gfx950 rocprofv3), 4 resident waves → 2 resident
waves shows no measurable throughput change. The MFMA pipeline saturates
by 2 waves issuing in alternation. Going from 2 → 1 wave sharply increases
wallclock because the single wave can't hide its own dependencies.

**Fix:**
```cpp
const double stallAmp =
    (effectiveWavesPerSimd >= 2.0)
        ? 1.0
        : 1.0 + (2.0 - effectiveWavesPerSimd) * 0.5;
// Result: 4 waves → 1.0×, 2 waves → 1.0×, 1 wave → 1.5×
```

**Effect:** BK=32 (4 waves) and BK=64 (2 waves) both score 1.0× on
stallAmp; per-K-iter overhead (Insight 3) differentiates them.
Only BK=128 (1 wave/SIMD) gets the cliff penalty.

---

## Insight 3 — Linear per-K-iter overhead alone can't capture BK choice

> **⚠️ SUPERSEDED (2026-07).** The `BK<64 ×1.30` penalty and linear per-K-iter
> overhead below are replaced by the Little's-law DRAM-saturation model — see
> Insight 4 in [`perf-model-saturation-physics.md`](perf-model-saturation-physics.md).
> The finding here ("a single monotonic term can't capture BK") was *correct* and
> is exactly why the real fix is memory-side saturation, not a per-iter penalty.

**Problem:** With perKIter=500, model picks BK=128 (fewer iters → less
penalty). With perKIter=200, picks BK=32 (high occupancy + low
per-iter penalty). With perKIter=350, model still picks asymmetric tiles
like (256,128,32).

**Reality:** Empirically BK=64 is the sweet spot for most tiles ≥ 128.
Linear `perKIter × numKIter` cannot simultaneously satisfy "BK=64 beats
BK=32" AND "BK=64 beats BK=128", because:
- BK=64 → BK=32 doubles K-iters (penalty too small to counteract BK=32's
  occupancy advantage)
- BK=64 → BK=128 halves K-iters (would need huge per-K-iter to penalize
  BK=128 enough)

**Fix:** Two-part:
1. Use moderate `perKIter = 350` (enough to nudge between adjacent BKs)
2. Add **explicit multiplicative penalty for BK < 64**:
   ```cpp
   if (cfg.blockK > 0 && cfg.blockK < 64) totalCycles *= 1.30;
   ```

The 1.30× captures un-modeled costs of small-BK that arise even at high
occupancy: tighter LLIR scheduling, more LDS bank conflict pressure per
FLOP, more frequent waitcnts that disrupt scheduling.

**Effect:** (128,128,32) drops from 421→201 TF prediction; (128,128,64)
stays at ~300 TF. Ranking flips correctly.

---

## Insight 4 — `vgprScale` was double-counting at mid shapes

**Problem:** Initially added `vgprScale = max(1.0, vgpr/100)` to penalize
(256,256,64)'s 316 VGPRs at 3840³+. But this OVER-penalized 256 at 3072³
(model said 256 was 1.91× slower than 128, empirical said 1.44×).

**Reality:** At mid shapes (3072), `(256,256,64)`'s waveEff is already
0.5625 → that's the discriminator. VGPR scaling on top double-counts the
"this config is bad here" signal.

**Fix:** Removed `vgprScale` entirely. The LDS-aware stallAmp (Insight 1)
catches the occupancy cost; waveEff catches the under-subscription cost.
Together they get the 256 vs 128 ratio within 5% of empirical at every
shape from 3072 to 4096.

---

## Insight 5 — Calibrate against EMPIRICAL ratios, not magnitudes

The model's predicted TFLOPS are systematically ~3× lower than empirical
(model 280 vs empirical 822 at (128,128,64) 3072³). This is because the
absolute cycle counts and clock-rate conversion are uncalibrated.

**What matters for ranking:** the *ratio* between configs. Calibration
should target:

```
model_cycles[A] / model_cycles[B]   ≈   empirical_wallclock[A] / empirical_wallclock[B]
```

Specifically, the empirical wallclock ratios (rocprof GRBM_GUI_ACTIVE) at
3072³:
- 64×64×128  / 128×128×64 = 1.67
- 256×256×64 / 128×128×64 = 1.44

After tuning, model ratios:
- 64×64×128  / 128×128×64 = ~1.5 (close)
- 256×256×64 / 128×128×64 = ~1.40 (very close)

---

## Insight 6 — LLIR scheduling must be enabled per kernel variant

**Problem:** When PerfModel picked an asymmetric tile like (128,64,128)
that routes to `v_persistent_v1`, the kernel ran 30% slower than expected
because v1 didn't auto-wrap launch in `TRITON_ENABLE_LLIR_SCHED=1`.

**Cause:** `v_any_tile.matmul` and `v_persistent_any_tile.matmul` wrap
launches in `_llir_enabled()` context manager; `v_persistent_v1.matmul`
did not (was left "for caller to enable").

**Fix:** Added `with _llir_enabled():` around the launch in v1's matmul
function. Now LLIR is enabled regardless of which variant the router
selects.

**Lesson:** When a single dispatcher routes to multiple kernel variants
based on tile size, ALL variants must apply the same compile-time options
(LLIR scheduling, env vars) to maintain consistent performance baseline.

---

## Insight 7 — Asymmetric tiles need extra scrutiny in the model

**Problem:** PerfModel was picking asymmetric tiles like (128,64,128) and
(256,128,32) at various shapes, but empirically these often underperform
the corresponding square tile.

**Cause:** The model's `waveEfficiency` computation favors asymmetric
tiles (more total tiles → better fit in 256 CUs). Combined with lower
per-tile compute, asymmetric tiles look "cheaper" in the model.

**Reality:** Asymmetric tiles often have:
- Different LDS layout per A vs B → bank conflict variance
- Routing penalties (some go to v1 instead of any_tile)
- Suboptimal MFMA decomposition

**Mitigation:** The BK<64 penalty (Insight 3) incidentally caught
(256,128,32) and (128,64,32) cases. For BK=64 asymmetric tiles like
(128,64,64), no explicit penalty exists — but they're rarely chosen
because square tiles usually win on raw compute density.

---

## Calibration Constants (gfx950)

> **⚠️ SUPERSEDED (2026-07).** The constants in this block (`stallAmp`,
> `perKIterOverheadCycles=350`, `BK<64 ×1.30`, `perTileOverheadCycles`) belong to
> the earlier model era and are NOT in the current code. The shipped gfx950
> constants are in the "Current calibration constants" section of
> [`perf-model-saturation-physics.md`](perf-model-saturation-physics.md)
> (`kPeakMfmaEff=0.40`, `cleanWaveRel` aM=76/aN=156, `hbmLatencyCycles=2000`,
> `kMfmaLatencyWaves=1.2`, occupancy = Little's-law HBM saturation). Keep the block
> below only as historical record of how the model looked pre-2026-07.

After all the above:

```cpp
// stallAmp: cliff at 1 wave/SIMD
const double effectiveWavesPerSimd = wavesPerCU / numSimdPerCU;
const double stallAmp = (effectiveWavesPerSimd >= 2.0)
    ? 1.0 : 1.0 + (2.0 - effectiveWavesPerSimd) * 0.5;

// per-tile overhead (no vgpr scaling)
constexpr double perTileOverheadCycles  = 50000.0;
constexpr double perKIterOverheadCycles = 350.0;
const double overheadPerTile = perTileOverheadCycles + perKIterOverheadCycles * numKIter;

// BK<64 penalty (multiplicative, applied to totalCycles)
if (cfg.blockK > 0 && cfg.blockK < 64) totalCycles *= 1.30;
```

### Fix 8–11 constants (2026-06-08/09)

```cpp
// Fix 10: HBM3e peak (official 8 TB/s, was 3000 = 7.2 TB/s estimate)
hw.peakMemBwBytesPerCycle = 3333.0;   // 8 TB/s / 2.4 GHz; 7.18 TB/s sustained measured = 90%

// Fix 8: LDS read bandwidth (silicon-fixed) — adds ldsCycles to the roofline
hw.ldsBwBytesPerCycleCU   = 128.0;    // gfx950, per AMD GFX-9 Prog. Guide ch.15 (MI-350)
double ldsCycles = ldsBytesPerCTA / hw.ldsBwBytesPerCycleCU;

// Fix 6a + Fix 11: mfmaUtil with a 1/16 floor; inflation applied ONLY in max()
mfmaUtil = std::max(0.0625, mfmaUtilM * mfmaUtilN);   // 0.0625 = 1/16
double inflatedComputeCycles = est.computeCycles / mfmaUtil;   // baseline kept for overlap
double maxCycles = std::max({inflatedComputeCycles, est.memoryCycles, est.ldsCycles});

// Fix 9: cdna4 candidate stages
numStagesVec = {2, 3};   // was {2}; nS=3 LDS-overflow dropped by isValidConfig
```

**Why the `0.0625` floor on `mfmaUtil`?** It is `1/16` — the cap on the
compute-inflation penalty. `mfmaUtil = mfmaUtilM × mfmaUtilN`, each factor
`min(1, perWarpDim / mfmaDim)`. A degenerate per-warp geometry (e.g. a `[1,1]`
sub-tile against a 16×16 mfma) would give raw `1/256`, inflating compute 256×
and letting one pathological config dominate the ranking. The 1/16 floor caps
the penalty at "16× wasted compute" — empirically safe because realistic skinny
tiles (BM=16 against a 16×16 mfma) already give `mfmaUtilM = 1.0` and land at
0.25, well above the floor. It is a guard constant, not a derived value.

**Compute-roof calibration (FIXED 2026-06-10, was ~2–4× too low).** The CDNA4
compute peak was undercounted: the fallback `peakMfmaFlopsPerCycleCU` gave ~629
TF (4× low) and the main-path MFMA table gave ~1.26 PF (2× low) vs the published
MI355X dense fp16 peak of ~2.5 PF. Root cause: the table assumed CDNA4 had the
"same latency as CDNA3," ignoring the ~2× faster matrix engine. **Fix:** halved
the CDNA4 fp16/bf16 table cycles (16×16×32 32→16, 32×32×16 64→32) → 4096
FLOP/cyc/CU = 2.5 PF; fallback `throughputCycles` 64→16; fp8 32/64→8/16 = 5 PF
(2× fp16). **Validated by a before/after pick sweep (40 fp16 shapes): only 3
picks changed, all improvements** (BM128×256 → BM256×256 on large compute-bound
shapes): 32768×2112×7168 +16.5% (now matches TA exhaustive winner 1133 TF),
8192×5120×5120 +3%, 8192×5120×2880 +5%; 37/40 unchanged. fp8 change is
hardware-certain (2× fp16) but NOT sweep-validated for the fp8 kernel path;
fp6/fp4 cycles still need their own peak calibration (follow-on).

## Empirical validation (PM/Audit ratio, gluon kernel sweep)

| M    | Before (vgprScale + perKIter=500) | After (this doc) |
|------|---|---|
| 1536 | 95% | 97% |
| 2048 | 92% | 103% |
| 2560 | 132% | 133% |
| 3072 | 98% | 115% |
| 3584 | 81% | **100%** |
| 3840 | 80% | **95%** |
| 4096 | 79% | **93%** |

Ratios > 100% mean PM beats the audit table; ratios near 100% mean PM
matches audit. All previous regressions resolved.

---

## Open issues

1. **TFLOPS magnitude is uncalibrated.** Model predicts ~280 TF, empirical
   is ~822 TF for (128,128,64) at 3072³. Ratios match but absolute values
   don't. Would require modeling clock rate / cycles-per-second conversion.

2. **3840/4096 PM still 5-7% below audit.** Model picks (256,256,64)
   correctly but the next-best (128,128,64) is within 5% empirically.
   Tighter calibration could swing it either way.

3. **Asymmetric tiles still possible at non-standard BK.** If user
   requests an explicit (128,64,64) tile, no penalty applies. Could add
   a BM:BN ratio penalty if it becomes a problem.

4. **Cross-arch generality.** All constants here are calibrated against
   gfx950 only. CDNA3 (gfx942) has different VGPR file size and LDS
   budget; constants would need re-tuning.
