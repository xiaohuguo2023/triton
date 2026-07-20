# PerfModel grouped-GEMM ranking — concrete change plan

Turns `grouped_gemm_ranking.md` + `grouped_gemm_interface_sketch.md` into a ready-to-run
`PerfModel.cpp` change plan, and adds the **K-aware term** that the grid_m fix alone does
not cover. Line numbers verified against the current `gluon_perf_model` tree.

---

## 0. Two DISTINCT bugs (do not conflate)

The TP4/TP8 regression has two independent root causes. grid_m fixes the first, **not** the second.

| Bug | Symptom | Distinguished by | Fixed by |
|---|---|---|---|
| **B1 — padding-inflated grid** | scalar M (padded) over-inflates tile count → over-picks wide BN (BN256) for sparse decode / bm64-conc128 | real tile count `grid_m` vs scalar M | **Change Set A** (grid_m/pad_frac) |
| **B2 — K-driven BN flip** | down-proj `n2880`: **k720→BN128 (TP4)** but **k360→BN64 (TP8)**; same N, same routing, same grid_m | only K differs | **Change Set B** (K-aware latency term) |

Why A cannot fix B2: for the down-proj, N=2880 and the routing histogram (→ `grid_m`) are
**identical** at TP4 and TP8; only K changes. Every occupancy/tile-count term
(`total_blocks = grid_m·ceil(N/BN)`) is therefore identical for the two → the model must
pick the *same* BN for both, and is wrong for one. A K-dependent term is required.
(A *does* help the gate-up GEMM, where N differs: TP4 n1440 vs TP8 n720.)

---

## Change Set A — grid_m / pad_frac interface  (primary; from the sketch, verified)

**Inputs (per `rank_configs` call, precomputed once — `grid_m` is BN-independent):**
```
grid_m   = Σ_e ceil(M_e / block_m)     # real M-tile count, from routing_data.hist
useful   = Σ_e M_e
pad_frac = 1 - useful / (grid_m * block_m)
```

**A1. Struct** — `include/TritonAMDGPUTransforms/PerfModel.h:184` (`struct GemmProblem`), add:
```cpp
  int64_t gridMTiles = 0;   // Σ_e ceil(M_e/block_m); 0 ⇒ derive from M (dense fallback)
  double  padFrac    = 0.0; // wasted-compute fraction
```

**A2. Helper + route the 3 grid_m sites** (`lib/TritonAMDGPUTransforms/PerfModel.cpp`):
```cpp
static inline int64_t gemmGridM(const GemmProblem &p, int64_t blockM) {
  return p.gridMTiles > 0 ? p.gridMTiles : ceildiv(p.M, blockM);
}
```
- `:1052`  `const int64_t gridM = gemmGridM(prob, cfg.blockM);`  (selectGroupSizeM)
- `:1264`  `est.totalOutputTiles = gemmGridM(prob, cfg.blockM) * ceildiv(prob.N, cfg.blockN) * prob.batchSize;`
- `:1385`  `const int64_t gridM = gemmGridM(prob, cfg.blockM);`  (L2 hit-rate)

Everything downstream (`:1267` numWaves, `:1274` partial-wave CUs, `:1400` activeCUs,
`:1698` mem-cycle tile scaling, `:1814` low-tile occupancy) flows from `totalOutputTiles`
and is corrected automatically — no extra edits.

**A3. `padFrac` efficiency term** (optional second step): in the compute-bound path scale
compute cycles by `1/(1-padFrac)` (or add a wasted-tile term) so padding-heavy sparse
decode is not credited as useful FLOPs.

**A4. Bindings** — `python/PerfModelBindings.cpp:67-92`, add defaulted args
`grid_m_tiles=0, pad_frac=0.0` + `def_readwrite`. Defaulted ⇒ all existing callers (bf16,
dense) unchanged.

**A5. Selector** — `perfmodel_a8w4_select.py`:
```python
def _grid_m_from_hist(hist, block_m):
    tiles  = sum((int(m)+block_m-1)//block_m for m in hist if m)
    useful = sum(int(m) for m in hist)
    padded = tiles*block_m
    return tiles, (1.0 - useful/padded if padded else 0.0)
# in pick_a8w4 when hist present:
gm, pf = _grid_m_from_hist(hist, block_m)
prob = _pm.GemmProblem(rank_m, n, k, _F8,_F4,_F32, 8,4,32, grid_m_tiles=gm, pad_frac=pf)
```
hist absent → pass 0 → dense fallback (today's behavior).

**A-interim (no rebuild, direction check):** in the selector pass `M_eff = grid_m*block_m`
as the scalar M instead of `next_pow2(m)`, so `ceildiv(M_eff, blockM) = grid_m`. Strictly
better than `next_pow2(padded)`; use to confirm direction before the `.so` change.

---

## Change Set B — K-aware latency-hiding term  (fixes B2; our analysis)

**Problem:** the BN decision is a tension between (i) `numWaves` — fewer tiles for bigger BN,
a pure *cost* → favors wide BN; and (ii) `occupancyPenalty = 1/max(occ,0.25)` — VGPR occ,
K-independent. Neither depends on K, and per-tile arithmetic intensity
`AI = 2·BM·BN/(BM+BN)` is itself K-independent (K cancels). So the model cannot flip BN as
K changes. Physically, at **short K** the K-loop is tiny → per-tile compute is small vs the
fixed DRAM latency → you need **more concurrent tiles (smaller BN)** to hide latency
(Little's law). The model never rewards extra tiles for latency-hiding — `numWaves` is only
ever a cost.

**Fix (sketch):** in the memory-bound branch (near `occupancyPenalty`, ~`:1765–1815`),
make the occupancy reward depend on how latency-bound the tile is:
```
perTileSteadyCompute = est.computeCycles / numWaves      // ~ compute work per tile
latHidingDeficit     = clamp(memLatencyCycles / max(perTileSteadyCompute,1) , 1, occCap)
// when perTileSteadyCompute << memLatency (short K), deficit >> 1 → block-supply matters:
occForPenalty        = min(occForPenalty, achievedOcc)   // (the uncommitted occForPenalty cap)
effectiveOcc         = occForPenalty * latHidingDeficit  // more resident tiles help ONLY when latency-bound
occupancyPenalty     = est.isComputeBound ? 1.0 : 1/max(effectiveOcc, minOcc)
```
Net effect: at K=360 (short) the deficit is high → smaller BN (more N-blocks → more
resident tiles) is rewarded → BN64. At K=720 the deficit is ~1 → reverts to today's
behavior → BN128. Only MX-scaled (`hasMxScales`) + memory-bound; bf16/dense untouched.

`memLatencyCycles` and `perTileSteadyCompute` are already computable from existing
`est.*` fields; this is a re-weighting of the penalty, not new machinery.

---

## Order of operations

1. **Gap-1 ground truth first** — `run_ab.sh` enforce-eager matched A/B (conc16 + conc128),
   per-config `avg_us` across `(block_m, N, K)`. Without it we cannot calibrate/validate.
   *(Requires the machine — defer until free.)*
2. **Change Set A** (grid_m). Rebuild `perf_model.so` (`build_standalone_perf_model.sh`,
   inside the container). Expect: fixes the padding-inflation BN256 over-picks and the
   gate-up N-driven cases. Re-run whole-surface (`run.sh`).
3. **Check the down-proj residual** — if `n2880_k360` still ranks BN128 (or `k720` ranks
   BN64), that's B2 → implement **Change Set B**, recalibrate the deficit against Gap-1.
4. **Deferred refinement** — per-expert load-balance / tail-wave from the full histogram.

## Validation gates (all must hold)
- Top-1 BN matches Gap-1 measured winner across `(block_m, N, K)` — specifically
  **`n2880_k720→BN128` AND `n2880_k360→BN64`** (the B2 acid test).
- Whole-surface geomean ≥ current cap (**0.988**, `RANDOM_RANGE_RATIO=0.8`; or the
  official-harness **0.994** at ratio 1.0), with TP4/TP8 restored and no new regressions.
- **bf16 byte-identical** (gridMTiles=0, non-MX ⇒ both change sets are no-ops there).

## Risk notes
- Change B re-weights a load-bearing penalty; guard strictly on `hasMxScales && !isComputeBound`
  so it cannot touch the bf16 tie-break path the occupancy term currently protects.
- The uncommitted `occForPenalty` cap in the working tree is a *prerequisite piece* of B
  (the `min(occForPenalty, achievedOcc)` line) — fold it in rather than discarding it.
- All measurement gated behind machine availability; the code/interface edits (A1–A5, B)
  are machine-free and can be staged now.
