# AMD CDNA MFMA / Wavefront Primer (for PerfModel)

Background concepts behind PerfModel's cost functions, for gfx950 / MI355X
(CDNA4). Captures how the MFMA matrix instruction, the wavefront/lane/SIMD
hierarchy, and `kBase`/`kWidth`/`kPack` work, and how each feeds the model.

---

## 1. MFMA instruction tile: mDim · nDim · kDim

An MFMA (Matrix Fused Multiply-Add) is the matrix-core primitive executed by
one wavefront in (a few) instructions. One MFMA computes:

```
D[mDim × nDim]  +=  A[mDim × kDim]  ×  B[kDim × nDim]
```

- **mDim** = output rows produced (M side)
- **nDim** = output cols produced (N side)
- **kDim** = contraction depth done in that instruction (K accumulated)

The instruction name encodes them: `V_MFMA_F32_16x16x32_F16` → mDim=16,
nDim=16, kDim=32 → 16·16·32 = 8192 MACs per issue.

**Instruction tile vs block tile** — don't confuse with the per-workgroup
macro-tile `BM/BN/BK` (the config knob):

| | what it is | example (gfx950 fp16) |
|---|---|---|
| mDim/nDim/kDim | one hardware MFMA instruction tile (fixed by silicon) | 16×16×32 |
| BM/BN/BK | the workgroup macro-tile (the config we pick) | 256×256×64 |

A workgroup builds its BM×BN×BK tile by issuing many MFMAs:
`#MFMA per K-step = (BM/mDim)·(BN/nDim)`, `K-steps = BK/kDim`.

---

## 2. Wavefront, lane, SIMD, CU

### Wavefront = 64 lanes (wave64)
On CDNA the **wavefront** is the SIMD execution unit, **64 lanes wide**. The 64
lanes execute the *same* instruction in lockstep, driven by **one program
counter** per wavefront. `hw.waveSize = 64`.

### "Lane" vs "thread"
Same 64 slots, two abstraction levels:
- **"thread"** = the SIMT *programming* abstraction (write scalar-looking code,
  pretend 64 independent threads).
- **"lane"** = the SIMD *hardware* truth: one element slot of a 64-wide vector
  unit. Lanes share one PC → no independent branching (divergence handled by an
  **EXEC mask**, masked-off lanes idle). A VGPR is a 64-wide *vector* register;
  "lane i" = element i of it.

### Per-lane register width
**One VGPR = 32 bits = 4 bytes, per lane** (`bytesPerVgpr = 4`). Across the
wave a single VGPR spans 64 × 4 = **256 bytes**. One VGPR per lane holds 2 fp16
/ 4 fp8 / 1 fp32.

### CU = 4 SIMDs
```
CU ├─ SIMD0 ─┐  4 independent vector units, each runs its own wavefronts
   ├─ SIMD1 ─┤  + shared per CU: LDS (64KB on cdna3; 160KB budget gfx950), L1/TCP, scalar unit
   ├─ SIMD2 ─┤
   └─ SIMD3 ─┘
```
- `numSimdPerCU = 4`.
- A wavefront is assigned to **one SIMD for its entire lifetime** (never migrates).
  An 8-warp CTA spreads 2 waves/SIMD.
- The 4 SIMDs run **independently in parallel** — up to 4 waves issue per CU per
  cycle (one per SIMD).
- Each SIMD holds many **resident** waves for latency hiding
  (`maxWavesPerSimd = 10` on gfx950) → up to 4×10 = 40 waves resident per CU.
  When the running wave stalls on memory, the SIMD switches to another resident
  wave.
- **Physical width nuance:** each SIMD is physically **16 lanes wide** (SIMD16);
  it executes a wave64 by pipelining over **4 cycles** (16×4 = 64). "wave64" is
  the logical width. (RDNA = SIMD32/wave32; CDNA = 4×SIMD16/wave64.)

**Per-SIMD vs shared:** VGPR file, wave scheduler, resident-wave slots, and a
matrix (MFMA) unit are *per-SIMD*; LDS, L1/TCP, scalar unit are *shared per CU*.
This asymmetry is why VGPR pressure limits waves **per SIMD** while LDS limits
CTAs **per CU**.

---

## 3. kBase: elements per lane per MFMA issue

To issue one MFMA, each of the 64 lanes must hold its slice of the A operand
(mDim×kDim elements) in registers. Spread evenly:

```
kBase = (mDim × kDim) / waveSize = (mDim × kDim) / 64 = kDim / (64/mDim)
```

So the **divisor is `64/mDim`**:

| MFMA | mDim | divisor = 64/mDim | kBase |
|---|---|---|---|
| 32×32 | 32 | 2 | kDim/2 |
| 16×16 | 16 | 4 | kDim/4 |
| 4×4 | 4 | 16 | kDim/16 |

That is the `(mDim>=32)?2:(mDim>=16)?4:16` line in `deriveKWidth`.

**Worked (gfx950 fp16 16×16×32):** kBase = 16·32/64 = 512/64 = **8** (= kDim/4).

Intuition: smaller M (16 vs 32) means fewer rows but the same 64 lanes to fill,
so each lane is responsible for *more* K — hence 16×16 packs kDim/4 per lane
while 32×32 packs kDim/2.

---

## 4. kWidth = kBase × kPack

**kWidth** = K-elements each lane holds per A/B operand fragment. It sets the
A/B register-fragment size and the `ds_read`/`global_load` vectorization width.

```
kWidth = kBase × kPack          (kPack ∈ {1, 2}, an AccelerateAMDMatmul option)
```

- **kPack=1**: layout holds exactly one MFMA's worth of K per lane (kWidth=kBase).
- **kPack=2**: holds **two** consecutive MFMA issues' worth (kWidth = 2·kBase) —
  one fragment fetch stages operands for two back-to-back MFMAs along K.

### kPack=2 effects
1. **Register footprint doubles** — 2× the live A/B fragment VGPRs → higher
   pressure, can lower occupancy or spill. (This is the `kWidth` term in
   `estimateVgpr`.)
2. **Load width — dtype dependent:**
   - **fp8** (kBase=8 → 8 B = `ds_read_b64`): kPack=2 → 16 B = `ds_read_b128` —
     genuinely widens the single load (fills to the 128-bit max, halves #reads).
   - **fp16** (kBase=8 → 16 B = `ds_read_b128` already at the 128-bit max):
     kPack=2 doesn't widen one read — it **groups two b128 reads** to feed two
     MFMAs. Benefit is scheduling, not load width.
3. **Latency hiding** — more independent MFMA work queued per load → fewer
   load↔compute sync boundaries.

### Tradeoff / when it helps
kPack=2 trades register pressure for memory efficiency + scheduling slack. Wins
on sub-16-bit dtypes (fp8/fp4, real b64→b128 win) and compute-bound tiles with
VGPR headroom; loses when registers are tight (spilling). TA tuned **kPack=1**
for the gfx950 fp16 shapes (fp16 already at b128 → little load-width benefit,
costs registers) — the kPack=2 win on this HW is mostly an fp8/fp4 story.

### deriveKWidth (PerfModel.cpp)
```
if (cfg.kWidth > 0) return cfg.kWidth;     // in-compiler: read DotOperandEncodingAttr.getKWidth()
mDim   = cfg.mfmaNonKDim or selectMfmaNonKDim()
kDim   = MFMA table lookup
divisor = (mDim>=32)?2:(mDim>=16)?4:16
kBase  = kDim / divisor
return kBase * kPack                        // offline: derive analytically
```
This mirrors AccelerateAMDMatmul's own `kBase × kPack` derivation — offline it
*predicts* the kWidth the compiler would pick; in-compiler it *reads back* the
actual one (the `cfg.kWidth>0` short-circuit). Same number, two sources.

---

## 5. How these feed the model

- **kBase / mDim·nDim·kDim** → `selectMfmaNonKDim` (instruction pick) and compute-cycle count.
- **kWidth** → `estimateVgpr` (A/B fragment regs = `blockM·kWidth·aBytes / (waveSize·4)`) and the ds_read width in the LDS roofline (Fix 8).
- **waveSize=64, bytesPerVgpr=4** → VGPR counts = `total_bytes / (64·4=256)`.
- **numSimdPerCU=4** → ×4 in `peakMfmaFlopsPerCycleCU` (4 matrix units/CU) and ÷4 in `effectiveWavesPerSimd = wavesPerCU/4` (occupancy; stallAmp cliff below 2 waves/SIMD).
- **maxWavesPerSimd=10** → residency cap per SIMD.

See [[perf-model-skill]] for the calibration constants and
`docs/perfmodel-tuning-insights.md` for the regression-driven corrections.

---

## 6. Warp gridding: planWarps → wpcM / wpcN → mfmaUtil

`num_warps` is the warps in **one CTA** — but the model needs to know how those
warps are **laid out over the BM×BN output tile**. `planWarps` (a port of
Triton's AccelerateAMDMatmul warpsPerCTA logic, Fix 6a) returns:

- **wpcM** = warps spanning the M (rows / BM) direction
- **wpcN** = warps spanning the N (cols / BN) direction
- invariant: **wpcM × wpcN = num_warps**

Each warp then owns a per-warp sub-tile `perWarpM = BM/wpcM`, `perWarpN = BN/wpcN`,
which is compared to the MFMA shape to get the lane-utilization:
```
mfmaUtilM = min(1, perWarpM / mDim)
mfmaUtilN = min(1, perWarpN / nDim)
mfmaUtil  = max(1/16, mfmaUtilM × mfmaUtilN)
```
**Why it matters:** if a warp's sub-tile is *smaller* than the MFMA shape, the
MFMA still runs full 16×16 but wastes output lanes → `mfmaUtil < 1` → the compute
term is inflated by `1/mfmaUtil`. This is the sub-MFMA-tile penalty.

**Worked (gfx950 fp16, 16×16×32):**
- 256×256, 8 warps → wpcM=2, wpcN=4 → perWarp 128×64 → both ≫16 → mfmaUtil=1.0 (no waste).
- skinny split with perWarpM=4 vs mDim=16 → mfmaUtilM=4/16=0.25 → 4× wasted compute.

The algorithm greedily doubles warps in whichever dim has more room
(`leftM=(BM/(mDim·2))/r0` vs `leftN=(BN/nDim)/r1`), keeping each warp's sub-tile
as close to ≥-mfma-shape as possible → maximizing mfmaUtil. Compiler-mirroring:
if Triton changes warp distribution, this port must track it.

### Worked example: why BM=16, BN=32, 4 warps → per-warp [4, 32]

The CTA owns a **16×32 output tile** and has **4 warps**. The warps divide it as
a 2-D grid (`wpcM × wpcN = 4`); each warp's slice = `[BM/wpcM, BN/wpcN]`. Options:

```
wpcM=1, wpcN=4 (side by side)         wpcM=4, wpcN=1 (stacked)  ← planWarps picks this
 ┌────┬────┬────┬────┐  rows 0–15      ┌─────────────────┐ rows 0–3   warp0
 │ w0 │ w1 │ w2 │ w3 │                 ├─────────────────┤ rows 4–7   warp1
 └────┴────┴────┴────┘                 ├─────────────────┤ rows 8–11  warp2
 each: 16 rows × 8 cols = [16, 8]      ├─────────────────┤ rows 12–15 warp3
                                       └─────────────────┘
                                       each: 4 rows × 32 cols = [4, 32]
```

**Why no clean split exists:** the MFMA wants each warp ≥ 16×16, but the tile is
only 16×32 = room for **2** full MFMA tiles, yet there are **4** warps. Every
split forces some warp below 16: `[16,8]` (N=8), `[4,32]` (M=4), `[8,16]` (M=8).
A clean 4-way split needs 16×64 (→1×4) or 32×32 (→2×2).

**Why planWarps lands on [4,32]:** it first tries to put all 4 warps across N,
but that needs 4×16 = 64 columns and N is only 32 → the guard
`if (r1·nDim > blockN)` (64 > 32) fires and **swaps the split to M** →
`wpcM=4, wpcN=1` → `[4, 32]`. Plain English: *"can't give 4 warps 16 columns
each (only 32 available) → stack them in M instead,"* leaving 4 rows/warp.

[4,32] vs the 16×16 MFMA → `mfmaUtilM = 4/16 = 0.25`, `mfmaUtilN = 1` →
`mfmaUtil = 0.25` → 4× wasted MFMAs → 4× ds_read traffic → LDS-bound (the
16×24576×1536 case study). **BN=64** makes the tile 16×64 = room for a clean
1×4 → per-warp `[16,16]` → mfmaUtil=1, no waste → back to HBM-bound.

**Root cause = M=16:** with only 16 rows, once there are more warps than the tile
can absorb cleanly in N, the extra warps subdivide the already-minimal M below the
MFMA height. **This is not the model choosing** — planWarps mirrors Triton's
`warpsPerCTA`; [4,32] (util 0.25) is what Triton actually emits even though [16,8]
(util 0.5) exists, and the model reproduces it to cost the real kernel.

---

## 7. num_warps vs resident waves (don't confuse them)

Three different "wave" counts:

| term | what it is | typical |
|---|---|---|
| **num_warps** | wavefronts in **one CTA** (config knob) | 4 or 8 |
| **waves/SIMD** | **resident** wavefronts on one SIMD (across multiple CTAs) | up to 10 |
| **waves/CU** | resident across the CU (4 SIMDs) | up to 40 |

A CTA's warps spread across the 4 SIMDs: one `num_warps=8` CTA gives only
`8/4 = 2` waves/SIMD. To reach the **saturation point of 4 waves/SIMD (= 16
waves/CU)** you need **multiple resident CTAs**, not a bigger CTA:
- num_warps=8 → 2 CTAs;  num_warps=4 → 4 CTAs.

How many CTAs fit (`ctasPerCU`) is set by **VGPR + LDS resources**, not by
num_warps. So **more occupancy comes from more CTAs, not more warps/CTA** — which
is why the model only sweeps num_warps ∈ {4, 8} and reaches saturation via CTA
residency. `occPenalty` caps the reward at `satOcc = 4/maxWavesPerSimd = 0.4`
(PDF-confirmed: SQ holds 40 wavefronts = 4 groups of 10).

---

## 8. How each formula section decides ranking, by shape regime

Ranking key: `predicted_TFLOPS = 2·M·N·K / (effTile × numWaves × occPenalty / clockHz)`.
- **effTile is a roofline `max(compute, memory, lds)`** — the *largest* term is the
  bottleneck, and it *changes with shape*. `overlap` resolves the compute↔memory tie.
- **numWaves, occPenalty** are outer multipliers.
- **Resources (VGPR/LDS)** never appear directly — they gate feasibility and feed occupancy.

Role of each section:

| section | role in ranking | when it's the discriminator |
|---|---|---|
| **Resources** | hard feasibility filter (lds>cap → dropped) + feeds occupancy | large tiles (spill / LDS-overflow guard) |
| **Occupancy** | `occPenalty` multiplier (memory-bound only, cap 0.4) | memory-bound tiles that differ on resident waves |
| **Compute** | `compute` term in max() | large square / compute-bound (4096³, 8192³) |
| **Memory** | `memory` term in max() (+ L2 hit rate) | skinny-M / low-AI (decode, MoE, 16×N×K) |
| **Wave quant** | `numWaves` step multiplier | medium prefill near the ⌈tiles/numCUs⌉ boundary |
| **LDS + overlap** | `lds` term in max(); overlap shrinks residual | extreme M-skinny (ds_read inflation, 16×24576×1536) |

**Per-regime summary:**

| regime | bottleneck (max) | what decides the pick |
|---|---|---|
| Large square / compute-bound | **compute** | biggest useful tile, mfmaUtil=1, no spill |
| Memory-bound / skinny-M | **memory** | fewest bytes/output, best L2 hit, enough waves |
| Extreme M-skinny + sub-tile waste | **lds** | BN≥64 to restore ds_read_b128 |
| Medium prefill near wave boundary | compute≈memory | fill whole waves (avoid +1-wave cliff) |
| Tiny | memory + under-fill | numWaves≈1, launch-dominated |

**One-sentence intuition:** the roofline `max()` auto-selects the bottleneck per
shape; `overlap` breaks the compute-vs-memory tie; `numWaves`/`occPenalty`
modulate; `Resources` filter the infeasible. No single term decides — the
*dominant* term switches as the shape moves between regimes, which is exactly what
a fixed heuristic or an asm-level model can't do. (This is also why each
correction in `perfmodel-tuning-insights.md` repaired one regime's discriminator
without breaking the others.)
