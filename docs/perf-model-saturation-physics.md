# PerfModel — Saturation Physics, Explained (gfx950 / MI355X, 2026-07)

**Audience: your future self, or anyone new to the perf model.** This document
explains — from the ground up, with the actual shapes and numbers we hit — how
we rewrote the GEMM tile-selection cost model so that it *derives* the right
`(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` from a few physical laws
instead of a pile of fitted constants.

If you only remember one thing: **almost every "which config is faster?" question
comes down to one of two saturations —**
1. **Is the MFMA (matrix) pipeline kept busy?** (compute side)
2. **Is HBM (main memory) kept busy?** (memory side)

and both are governed by the same idea — **you need enough work *in flight* to
hide latency** (Little's law). Everything below is a special case of that.

---

## Part 0 — The 5-minute hardware primer (so the rest makes sense)

You can skip this if you already know CDNA4, but read it once.

### The chip
- An **MI355X** GPU has **256 Compute Units (CUs)**. A CU is the basic engine.
- Each CU has **4 SIMDs** (think: 4 lanes that each run one wavefront at a time).
- A **wavefront** ("wave") = **64 threads** that execute in lockstep. A SIMD can
  hold *several* waves resident at once and switch between them instantly to hide
  latency. On gfx950 up to **8 waves per SIMD** fit (limited by registers/LDS).
- **VGPRs** = per-thread registers (a fixed file per SIMD). More registers per
  wave → fewer waves fit → lower *occupancy*.
- **LDS** = fast on-chip scratch memory (~64 KB/CU) used to stage tiles.
- **HBM** = the big, slow, off-chip main memory. High bandwidth (~8 TB/s) but
  high latency (~1000–2000 cycles for a round trip).
- **MFMA** = the matrix-multiply instruction (the "tensor core"). It has a
  *throughput* (how often you can issue one) and a *latency* (how long until the
  result is ready). Back-to-back **dependent** MFMAs stall on that latency unless
  other independent work fills the gap.

### The GEMM and how a "config" maps to hardware
We compute `C[M,N] = A[M,K] × B[K,N]`. The output is cut into **tiles** of size
`BLOCK_M × BLOCK_N` (e.g. `128×256`). **One tile = one CTA** (a thread block, aka
workgroup). A CTA is placed on one CU and computes its tile by looping over K in
chunks of `BLOCK_K`:

```
for k in 0, BLOCK_K, 2*BLOCK_K, ... K:      # numKIter = K / BLOCK_K iterations
    load A[:, k:k+BLOCK_K]  and  B[k:k+BLOCK_K, :]   # from HBM → LDS → registers
    C_tile += A_frag @ B_frag                        # MFMA instructions
```

A **config** is the set of knobs the compiler/tuner picks:
- `BLOCK_M, BLOCK_N` — the output tile shape.
- `BLOCK_K` — how much of K to process per loop iteration.
- `num_warps` — how many waves per CTA (they split the `BM×BN` tile among them).
- `num_stages` — software-pipelining depth: how many loop iterations' loads are
  prefetched ahead of the compute that consumes them. `num_stages=3` ("ns3")
  keeps ~2 iterations of loads in flight; ns2 keeps ~1. More stages = more LDS.

**The perf model's job:** given a shape `(M,N,K)`, score every candidate config
and predict which is fastest — *without running it*. That prediction is a
**roofline**: `time ≈ max(compute_time, memory_time)` with some overlap.

### Why this is hard: "the roofline ties configs"
The naive roofline is *invariant* to several knobs, so it predicts a **tie** and
some arbitrary tie-break picks the winner — often the wrong one:
- **BLOCK_K:** compute ∝ BK and the number of iterations ∝ 1/BK, so they cancel.
  The roofline can't tell BK32 from BK128. **(fixed in Insight 4 below)**
- **num_warps:** the model capped "SIMDs used" at 4, so W4 and W8 looked
  identical. **(Insight 3)**
- **num_stages** for compute-bound tiles: compute doesn't depend on ns, so ns2
  and ns3 tied. **(Insight 3)**
- **occupancy vs tile width:** the old penalty counted CTAs, ignoring how many
  bytes each CTA moves. **(Insight 2)**

The whole 2026-07 effort was: **replace those ties + tie-breaks with real
physics.** Result on our 471-shape fp16 test suite: **win rate vs Triton's own
autotuner went 83% → 95%, geomean speedup ~1.44×.**

### How we measure (so numbers are trustworthy)
For each shape we run PerfModel's top pick AND Triton autotune's pick and report
the ratio `PM_TF / auto_TF` (>1 means PerfModel picked a faster config), on a
**verified-idle GPU** (`rocm-smi --showuse`, refuse to start if any GPU is busy).

**Measurement methodology (updated 2026-07, TensorAtlas-style — this matters a
lot for small kernels):**
- **Device-time, via `torch.profiler`**, not HIP-event wall-clock. The profiler
  reads the actual GPU kernel duration (`self_device_time_total`, from
  roctracer). Wall-clock includes ~2–4 µs of **CPU launch overhead per call**,
  which for a 3–5 µs kernel *dominates* the measurement — it both inflates ratios
  and is unfair if the two configs launch via different wrappers.
- **Interleaved** — each round measures a short burst of PM, then autotune, then
  rocBLAS, so all configs see the same thermal/clock state (fair A/B), instead of
  measuring one fully then the next.
- **IQR outlier filter** — drop upper (slow throttle/jitter) spikes via the Tukey
  fence, average the rest.

> **⚠️ Hard lesson (2026-07).** An earlier version of this doc claimed the tiny
> SMALL/`*_SKINNY` categories "all win 1.5–2.1× and the sweep losses were noise."
> **That was wrong** — an artifact of HIP-event wall-clock timing. When we
> switched to profiler device-time + interleaved + IQR, two of those categories
> turned out to be **real losses**: `LARGE_M_SKINNY` (0.91) and `LARGE_N_SKINNY`
> (0.93) — tiny-K (K=32/64) skinny shapes where PM's *kernel* is genuinely ~8%
> slower than autotune's. The old "clean" event debug had compared PM's raw
> kernel launch against autotune's *wrapper* (unfair CPU overhead) and measured
> wall-clock (launch overhead), inflating PM to a fake 1.5×. **Lesson: for
> microsecond kernels, only device-time is trustworthy; wall-clock lies.** These
> 6 skinny shapes were then a real regression — **now fixed** (Insight 5): the
> better measurement revealed a genuine model bug in the block-K term, which we
> could only see *because* we finally measured correctly.

---

## Part 1 — Little's law, the one idea behind everything

**Little's law** (from queueing theory, proved by John Little in 1961):

```
   L  =  λ  ×  W
(items    (through-  (time each
 in       put rate)   item spends
 flight)              in the system)
```

It's absurdly general — true for any stable system regardless of the details.

**Highway analogy.** A highway sustains `λ` cars/hour. Each car is on it for `W`
hours. So the number of cars on the road at any instant is `L = λ×W`. If you want
to *sustain* high throughput on a long (high-`W`) highway, you need *many* cars on
it at once. Too few cars in flight → you can't hit the road's capacity.

**Apply it to GPU memory.** HBM is the highway. To *sustain* peak bandwidth (`λ =
peak BW`) when the round-trip latency is `W = hbmLatency ≈ 2000 cycles`, you need

```
bytes_in_flight  =  peakBW × hbmLatency          (to saturate)
achievedBW       =  min(peakBW, bytes_in_flight / hbmLatency)
```

If your kernel only keeps, say, half that many bytes outstanding, you get **half
the bandwidth** — even though the hardware is "idle" waiting. This single formula
explains occupancy (Insight 2), block-K (Insight 4), and num_stages' memory
effect. The MFMA-pipeline version (Insight 3) is the *same idea* applied to the
matrix unit instead of memory.

**Where "bytes in flight" comes from in a GEMM:**
```
bytes_in_flight = (resident CTAs on the CU)          # how many tiles at once
                × (pipeline depth = num_stages - 1)  # loads prefetched ahead
                × (DRAM bytes loaded per K-iteration) # = (BM+BN)*BK*2 for fp16
```
Every knob that was "tied" shows up here: CTAs (occupancy/num_warps), depth
(num_stages), and `(BM+BN)*BK` (tile width and block-K). That's why one law
untangles all of them.

---

## Part 2 — The four fixes, each with its story

Ordered by which "tie" they resolve. Each has: **the symptom** (a real shape that
lost), **the detective work**, **the physics**, **the fix**, **the result**.

Notation: a config is written `BM×BN×BK Wn Sm` = block-M × block-N × block-K,
`n` warps, `m` stages. "cb" = compute-bound (else memory-bound).

---

### Insight 1 — Clean-wave MFMA efficiency: big tiles are more efficient, and it's asymmetric

**Symptom.** On MEDIUM shapes like `4096×5120×5120`, the model picked a `128×128`
tile, but the measured-fastest was `128×256`. Separately, on LARGE_N shapes like
`4×24576×1536`, the model top-picked a *tiny square* `32×32` that measured ~2.5×
SLOWER than the real winner.

**Detective work.** The compute part of the roofline used a "de-rate":
```
mfmaEfficiency = 0.40 × min(1, min(BM,BN)/128)
```
i.e. efficiency grows with the *smaller* tile dimension and then **saturates at
128**. Two problems: (a) it rates `128×128`, `128×256`, `256×256` all equal (all
hit the cap), and (b) it's far too generous to tiny tiles — `32×32` gets `0.40 ×
32/128 = 0.10`, but its *real* efficiency is ~0.07.

To find the truth we measured a **clean single wave** (no wave-quantization
noise) at `M=N=16384` for many tile shapes, reading the realized fraction of peak
MFMA throughput. It fit an **asymmetric, saturating** curve (SSE 0.0025):

```
frac_of_peak  ≈  2.14 × (BM/(BM+76)) × (BN/(BN+156))
```

**The physics.** Bigger per-warp output tiles keep the MFMA accumulator pipeline
fed with back-to-back independent instructions and reuse each loaded operand more
times → closer to peak. The curve **saturates** (a 512-wide tile isn't much
better than 256) and is **asymmetric** — the `BN` (the B/weight operand) matters
more than `BM`, because the B-fragment drives the inner MFMA cadence. Concretely,
per this formula relative to a `128×128` baseline (=1.0):
`32×32 → 0.18`, `128×256 → 1.37`, `256×256 → ~1.6`.

**The fix.** We defined `cleanWaveRel(BM,BN)` = that curve normalized to
`128×128`, and applied it to `mfmaEfficiency`:
- **Large tiles** (`cleanWaveRel > 1`) get a *credit* — but only when it's
  physically real:
  - **fill gate:** only if the tile still covers ≥ `numCUs` output tiles (i.e.
    fills the machine). *Why:* a huge tile that leaves most CUs idle is not
    actually faster; an earlier version without this gate regressed 11 of 13
    categories by over-picking big tiles on small shapes.
  - **padding guard:** require `N ≥ BN` and `M ≥ BM`, so we don't credit a tile
    wider than the problem (a `BN=256` tile on an `N=128` problem wastes half).
  - clamped to `[1, 1.61]` (≤ the measured `256×256` realized efficiency).
- **Small tiles** (`cleanWaveRel < 1`, i.e. `min(BM,BN) < 128`) get a *penalty*
  (via `min()`, penalty-only), and this one is **ungated**. *Why it's safe
  ungated:* the compute de-rate only matters when a tile is compute-bound; a
  memory-bound small tile (correct on skinny/tiny-K shapes) is unaffected. Only
  the *over-predicted compute-bound* tiny tiles move.

**Result.** MEDIUM `4096×5120×5120` correctly flips `128×128 → 128×256`. The
tiny-square over-prediction is cured: for `4×24576×1536`, `32×32` was predicted
231 TF but measured 93; the penalty drops it below the real winner. MEDIUM win
rate **75% → 96%**.

---

### Insight 2 — The occupancy penalty is HBM *saturation*, not CTA count

**Symptom.** LARGE_N shapes (tiny M, huge N — e.g. `4×24576×1536`) and the
tiny-M members of LARGE_NK (e.g. `128×53248×16384`) kept losing. The model's top
pick was a *machine-filling* narrow/tiny tile (like `32×32` or `BM64×BN64`), but
the measured winner was a **wide-BN** tile (`16×128`, or `BM128×BN256`) — even
though the wide tile has **low occupancy** (few CTAs resident per CU).

**Detective work.** The upstream model penalized low occupancy directly:
```
occupancyPenalty = 1 / vgprOccupancy      # 2× if occ=0.5, up to 4×
```
The reasoning: "few resident waves → can't hide memory latency → slow." That's
half-right, but it measures parallelism by **counting CTAs** and completely
ignores **how many bytes each CTA has in flight.** A wide-BN tile loads a huge
chunk per iteration, so even one CTA can keep HBM saturated.

We verified this with an instrumentation probe (`PM_DEBUG_SAT`) that printed the
actual bytes-in-flight. Example — `4×24576×1536`, tile `16×128`:
- `1/occupancy` said: penalty **2.0×** (occupancy 0.5) → looks slow.
- Little's law said: bytes in flight ≈ **66 KB**, needed to saturate ≈ **13 KB**
  → **fully saturated → true penalty 1.0×.**

The wide tile was being wrongly slowed 2× in the model, so the model picked the
tiny square instead — which measured *slower* in reality.

**The physics.** This is Part 1's formula, verbatim:
```
outstanding = ctasPerCU × pipelineDepth × dramBytesPerKIter
achievedBW  = min(peakBW, outstanding / hbmLatency)
penalty     = peakBW / achievedBW        # ≥ 1, capped at 4×
```
A wide/deep tile saturates HBM with few CTAs (no penalty); a genuinely starved
tile (few CTAs *and* small per-CTA loads) still gets penalized — for the right
reason now.

**The fix.** Replaced the dense `1/occupancy` penalty with the Little's-law
saturation penalty above. **Important:** the MX/a8w4 path *keeps* the old
`1/occupancy` form — for sparse-decode a8w4, narrow BN genuinely wins and that
penalty is load-bearing; all our changes are `!hasMxScales`-gated so a8w4 is
byte-for-byte unchanged.

**Result.** LARGE_N **71% → 100%** win. LARGE_NK **+0.10 geomean**. Notably, this
also fixed the M=128 LARGE_NK family (`BM128×BN256` on deep K): those tiles were
mis-labeled "memory-bound" by the old model; once the memory penalty is removed
they correctly become compute-bound and win. A compute-side credit alone could
not have fixed them — the memory misclassification had to go first.

---

### Insight 3 — MFMA latency hiding ranks num_warps AND num_stages (one term)

**Symptom.** Two separate ties:
- `256×256` compute-bound tiles: **W8 measured +23% over W4**, but the model tied
  them (e.g. `8192×2112×7168`: W8=794 TF, W4=644 TF, model said equal).
- Compute-bound deep-K tiles: **ns3 measured +23% over ns2** (same shape: ns3=794
  vs ns2=617), but the model tied them (compute doesn't depend on num_stages).

**Detective work.** For num_warps: the model computed compute time as
`... / min(num_warps, 4)` because a CU has 4 SIMDs — so W4 and W8 both "use 4
SIMDs" and got identical compute. But that misses *latency hiding*: W8 puts **2
waves on each SIMD**, W4 puts **1**. We dumped the estimate and confirmed W4→1
wave/SIMD, W8→2 waves/SIMD, model prediction identical.

**The physics (the MFMA version of Little's law).** An MFMA has a *latency* — a
dependent chain of MFMAs (accumulating into the same tile) stalls waiting for
each result unless *independent* MFMA groups fill the gap. Those independent
groups come from **two** sources:
1. **resident waves per SIMD** — different waves' MFMAs interleave.
2. **software-pipeline depth** — the `num_stages-1` prefetched k-iterations each
   have an independent MFMA group ready.

So the hiding parallelism is the **product**:
```
hiding = wavesPerSimd × pipelineDepth
inflatedComputeCycles *= 1 + kMfmaLatencyWaves / hiding      # k ≈ 1.2
```
- `num_warps` enters via `wavesPerSimd = min(maxWaves, vgprPerSimd / vgprPerWarp)`:
  more warps split the accumulator → fewer registers per warp → more waves fit.
  (This *also* captures register spills — a spilling config drops to 1 wave and
  is penalized automatically.)
- `num_stages` enters via `pipelineDepth`.

We calibrated `k ≈ 1.2` from the measured wave curve: doubling waves gives +16–23%
(1→2), +5% (2→4), ~0% (4→6) — a saturating return, which `1 + k/hiding` matches.

**The fix.** Added the `mfmaLatencyStall` above to the compute roofline (not the
baseline used for overlap). Now the model picks W8 for `256×256` and ns3 for
compute-bound deep-K, both for the right reason.

**Result.** `8192×2112×7168` correctly top-picks `BK32×ns3×W8` (the −23% gap
closed). LARGE_N **+0.16 geomean** on top of Insight 2.

**A wrong turn worth remembering.** We *first* tried a "pipeline fill cost"
(charge the `ns-1` prologue iterations) to also capture the shallow-K case where
ns2 wins. It penalized ns3 everywhere and regressed the deep-K categories that
legitimately want ns3 (LARGE_NK/MK/K all dropped). Reverted. The depth-aware
stall alone is the clean win; the shallow-K ns2 preference is left as a known
residual (see below).

---

### Insight 4 — Block-K is Little's-law DRAM saturation (NOT a per-iteration penalty)

**Symptom.** LARGE_MK shapes (large M, *narrow* N=128–256, large K) kept picking
`BLOCK_K=32` where the measured winner used `BK64` — same tile, wrong block-K,
losing ~5–8%. Examples: `16384×256×7168` (0.92× autotune), `32768×128×5120`
(0.95×). The model *tied* all block-K values.

**Detective work — and a false start.** Our first instinct was "small BK = more
loop iterations = more per-iteration overhead," so we added a linear
`overhead × numKIter` term. It fixed deep-K but **inverted shallow-K**: on
LARGE_MN shapes (`4096×24576×1536`, K=1536), the measured winner is *small* BK,
and the linear term wrongly forced large BK — dropping that category from
1.115 → 0.928. **A single monotonic term cannot express a non-monotonic
preference** (deep-K wants big BK, shallow-K wants small BK). Reverted.

So we measured properly: clean BK-isolated bandwidth curves (fix everything,
vary only BK). Key finding — **the block-K effect is per-*iteration* bytes, not
per-matrix "burst granularity."** How we know: **B-heavy** shapes (where the A
matrix's traffic is ≈ 0) show the *same* BK dependence as A-heavy shapes. If it
were about A's contiguous memory bursts, B-heavy shapes wouldn't care about BK —
but they do. So it's tied to the *total* bytes loaded per iteration,
`(BM+BN)×BK×2`, which scales with BK for both operands.

And that is **exactly Little's law again**: small BK → few bytes per iteration →
few bytes in flight → HBM under-saturates → achieved BW < peak. Confirmed by the
strong **num_stages × BK coupling** we measured: for `16384×256×7168`, BK32 hits
only 0.65 of peak BW at ns2 but 0.83 at ns3 — a deeper pipeline compensates for
smaller transfers, precisely as `outstanding = depth × BK×bytes` predicts.

**Why the model tied BK.** The roofline's `memoryCycles` used **peak** bandwidth
(which is BK-invariant — total traffic is the same for any BK). The achieved-BW
Little's law existed in the code but only fed the *overlap* term, not the main
`memoryCycles`. So BK never moved the roofline.

**The fix (the "unification").** Make `memoryCycles` itself use *achieved* BW:
```
dramBwEff    = min(peakBW, pipelineDepth × dramBytesPerKIter / hbmLatency)
memoryCycles = dramTraffic / dramBwEff   (+ L2 traffic / L2 BW)
```
and we **raised `hbmLatency` from 1000 → 2000 cycles**, back-solved from the
measured saturation curve (`16384×256×7168` ns2 = 0.65 / 0.97 / 1.0 of peak at
BK32/64/128 pins the latency at ~2000). The old 1000 saturated too early, which
is *why* every BK looked equal. This single constant is now shared by both the
saturation term and the latency-hiding overlap term.

The non-monotonic BK preference now **emerges for free**: deep K needs big BK to
put enough bytes in flight (large-BK wins); shallow K already saturates and
prefers small BK for other reasons (occupancy) — no special-casing.

**Two dead ends this replaced:** the old `BK<64 → ×1.30` penalty, and an interim
fudge `memoryCycles *= 1 + kGran/BK` (calibration showed its implied constant
varied from 4.8 to 29 — i.e. it wasn't a constant, so the form was wrong).

**Result.** LARGE_MK **83% → 100%** win; LARGE_NK **94% → 97%**; and it *removed*
a fitted constant instead of adding one. The whole memory model is now internally
consistent: block-K, num_stages, occupancy, and num_warps all flow from
`outstanding = CTAs × depth × bytesPerIter`.

---

### Insight 5 — Tiny-K (BK > K): charge the masked block

**Symptom.** Once we could measure microsecond kernels correctly (profiler +
interleaved + IQR — see the ⚠️ box above), the tiny-K skinny categories showed up
as **real losses**: `LARGE_M_SKINNY` 0.91, `LARGE_N_SKINNY` 0.93 (K=32/64, 0% win).
A profiler config sweep showed PM picked **BK256/BK512** where the measured winner
is **BK64/BK32**.

**Two coupled bugs, both when `BK > K`** (the block is bigger than the whole
reduction, e.g. K=64 with BK=256 → `numKIter = K/BK = 0.25`):

1. *Saturation over-credit* (in Insight 4's Little's-law term). For BK>K the loop
   runs ONE masked iteration streaming only K (not BK) elements, but the term used
   `depth × (BM+BN)·BK·2` outstanding bytes — 4× too many for K=64/BK=256 — so it
   declared BK256 bandwidth-saturated and fast. **Fix:** cap the effective depth at
   the iterations that actually exist — `depth = min(numStages-1, numKIter)` — so
   outstanding never exceeds the total traffic.

2. *Compute under-charge.* `numKIter = K/BK` is *exact*, so BK256 on K=64 was
   charged 0.25 iterations = only BK64-worth of MFMAs. But the kernel runs one
   FULL masked BK256 block — the MFMAs execute over the padded BK (K/BK useful),
   ~4× the work. **Fix:** floor `numKIter` at 1 for `computeCycles`, so BK>K pays
   for the masked block it actually runs.

Both guards fire **only when `BK ≥ K`** (tiny-K tiles); every `K ≥ 256` shape is
byte-identical. Together they make the model correctly prefer BK64/BK32 for K=32/64.

**Result.** `LARGE_M_SKINNY` 0.91 → 1.07, `LARGE_N_SKINNY` 0.93 → 1.06 (both now
win), `SMALL` → 1.19 (100%), and even `LARGE_NK` +0.004 (its K=128 members now
prefer BK128 over a wasteful BK256). Overall win 93% → **95%**. Residual: the
skinny 67% win is a warps/ns tie-break *within* the now-correct BK — a smaller,
separate axis.

---

## Part 3 — The calibration constants (2026-07)

These are the only tuned numbers, and each is grounded in a measurement:

```cpp
// (Insight 1) clean-wave MFMA efficiency, fit at M=N=16384
constexpr double kPeakMfmaEff = 0.40;              // achievable/theoretical MFMA ceiling
cleanWaveRel(BM,BN) = ((BM/(BM+76)) * (BN/(BN+156)))
                     / ((128/(128+76)) * (128/(128+156)));   // aM=76, aN=156, normalized to 128x128
//   large tile: mfmaEfficiency *= clamp(cleanWaveRel, 1.0, 1.61)          [fill + padding gated]
//   small tile: mfmaEfficiency  = min(mfmaEfficiency, 0.40*min(cleanWaveRel,1))   [minTileDim<128]

// (Insights 2 & 4) HBM round-trip latency — ONE constant, both BW-saturation AND latency-hiding
constexpr double hbmLatencyCycles = 2000.0;        // was 1000; back-solved from the BK bandwidth curve

// (Insight 4) achieved DRAM BW → feeds the roofline memoryCycles
dramBwEff = min(peakBwPerCU, (num_stages-1) * dramBytesPerKIter / hbmLatencyCycles);

// (Insight 2) occupancy penalty = HBM saturation deficit (dense); MX keeps 1/vgprOcc
outstanding      = ctasPerCU * pipelineDepth * dramBytesPerKIter;
occupancyPenalty = min(4.0, peakBwPerCU / min(peakBwPerCU, outstanding/hbmLatencyCycles));

// (Insight 3) MFMA latency-hiding stall — captures num_warps AND num_stages
constexpr double kMfmaLatencyWaves = 1.2;
inflatedComputeCycles *= 1 + kMfmaLatencyWaves / (wavesPerSimd * pipelineDepth);
```

**Everything above is dense (bf16/fp16) only, gated on `!hasMxScales`. The
a8w4/MX path is byte-identical** — its residency/decode physics is different and
still uses the older `1/occupancy` treatment (deliberately, it's validated).

---

## Part 4 — Results (fp16, 471-shape tutorial suite, PerfModel vs Triton autotune)

`PM/auto` = geomean device-time speedup of PerfModel's pick over autotune's pick
(>1 = better). **Measured with the profiler + interleaved + IQR methodology
above** — the honest numbers (the earlier event-timing table over-stated the
tiny-shape rows).

| category | n | PM/auto | win% | which physics carried it |
|---|--:|--:|--:|---|
| LARGE_NK | 261 | 1.60 | 92% | Insights 2 & 4 |
| MEDIUM | 53 | 1.12 | 98% | Insight 1 |
| VERY_LARGE | 49 | 1.17 | 98% | (already good; unchanged) |
| LARGE_K | 28 | 1.96 | 100% | Insight 4 |
| LARGE | 18 | 1.17 | 100% | Insight 1 |
| LARGE_MK | 18 | 1.45 | 94% | Insight 4 |
| LARGE_N | 14 | 1.15 | 100% | Insights 2 & 3 |
| LARGE_MN | 13 | 1.12 | 100% | (stable) |
| SMALL | 6 | 1.13 | 83% | (mostly ties/wins) |
| LARGE_K_SKINNY | 3 | 2.07 | 100% | Insight 4 |
| LARGE_M_SKINNY | 3 | 1.07 | 67% | fixed (Insight 5) |
| LARGE_N_SKINNY | 3 | 1.06 | 67% | fixed (Insight 5) |
| LARGE_M | 2 | 1.41 | 100% | |
| **overall (n-weighted)** | **471** | **~1.44** | **95%** | |

(Before this work, overall win rate was ~83% with MEDIUM/VERY_LARGE tie-break
*regressions*. The 97% figure quoted in earlier drafts was event-timing-inflated;
95% is the device-time truth, after the tiny-K fix in Insight 5.)

For reference, before this work overall win rate was **83%**, and MEDIUM /
VERY_LARGE had tie-break *regressions*.

---

## Part 5 — What did NOT work (so we don't retry it)

Recorded because each looked reasonable and cost real time:

1. **Ungated large-tile efficiency boost.** Crediting big tiles everywhere (no
   fill gate) over-picked them on small/skinny/under-filled shapes → −11/13
   categories. → Fixed with the fill + padding gate (Insight 1).
2. **Additive per-k-iteration overhead** (`effTile += ov·numKIter`) for block-K.
   Favored large BK on deep-K but *inverted* shallow-K (LARGE_MN 1.115→0.928). A
   single monotonic term can't fit a non-monotonic curve. → Replaced by
   memory-side saturation (Insight 4).
3. **Pipeline fill/prologue cost** to force ns2 on shallow-K. Penalized ns3 on
   the deep-K tiles that legitimately want it (LARGE_NK/MK/K down). → Dropped;
   kept only the depth-aware stall (Insight 3).
4. **`kDramGranularity = 1 + kGran/BK`** (interim block-K fudge). Right direction,
   wrong form — its "constant" varied 4.8–29 across shapes. → Replaced by the
   Little's-law achieved-BW model (Insight 4).

**The discipline that made this work:** *every* core-model change was gated on a
full 13-category A/B sweep, and any change that was net-negative on the reliable
(n≥13) categories was **reverted, not shipped** — even when it fixed the target
category.

5. **HIP-event wall-clock re-measurement of tiny shapes** (to "confirm" the n≤6
   categories). We did this repeatedly and it lied — event wall-clock includes
   launch overhead and, when PM and autotune launch via different wrappers, is
   *unfair*. It told us the skinny categories won 1.5×; profiler device-time
   showed two of them are ~8% **losses**. Lesson: re-measuring clean is not
   enough — you must measure the *right thing* (device time), interleaved, IQR.

---

## Part 6 — Honest remaining residuals

Not everything is perfect; these are known:

0. **~~LARGE_M_SKINNY / LARGE_N_SKINNY real losses~~ — FIXED (Insight 5).** These
   tiny-K (K=32/64) shapes were the losses the better measurement uncovered; the
   root cause was the BK>K block-K bug, now fixed (0.91→1.07, 0.93→1.06). Residual:
   they win at only 67% (n=3) due to a warps/ns tie-break *within* the correct BK.

1. **Shallow-K num_stages** (e.g. `4096×24576×1536`, K=1536, measures ns2 > ns3):
   the depth-aware stall (Insight 3) still prefers ns3, ~6% off the optimum — but
   it *still beats autotune*, so LARGE_MN stays 100% win and it costs no category.
   The fix for it (fill cost) regressed deep-K, so we left it.
2. **`16384×256×7168`** wants `BN256` (a whole-tile choice, not just block-K) —
   a different axis than Insight 4's block-K fix; still slightly off.
3. **VERY_LARGE / MEDIUM lost ~1 shape each** when we raised `hbmLatency` to 2000.
   That higher latency is *more correct* (it's what the BK curve measures) and it
   slightly tightens those categories' overlap. Accepted as the price of deleting
   the `kDramGranularity` fudge — net overall win rate still went up.

---

## Appendix — Debug tools used (all in `scripts/perf_baseline/`)

- `pm_config_sweep.py <M N K BM BN>` — measure every `(BK, ns, warps)` variant of
  a tile vs the model's prediction. This is how we found the W8-vs-W4 and
  ns2-vs-ns3 ties.
- `pm_bk_sweep.py` / `pm_bk_bw.py` — clean block-K isolation curves (fix
  warps/ns, vary only BK). This is how we found block-K is Little's-law, not
  granularity.
- `pm_debug_nk.py` / `pm_debug_mk.py` / `pm_debug_large_n.py` /
  `pm_debug_skinny.py` — per-shape PM-pick vs oracle vs autotune, with configs.
- `pm_losers_large_nk.py` — fast loser-finder across a whole category.
- `sweep_ab_categories.py` — the 13-category A/B gate (with the GPU-idle guard)
  used to accept/reject every change. Writes `docs/perf-baselines/ab_by_category-*.csv`.
- `PM_DEBUG_SAT=1` env — a (temporary) stderr print of bytes-in-flight vs the
  saturation threshold; how we proved Insight 2 numerically before changing
  ranking.

The terse per-fix changelog (with commit hashes) lives in
`perfmodel-tuning-insights.md`; this document is the "why it works" companion.
