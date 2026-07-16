# Making the perf model rank grouped MoE GEMM correctly

*Supersedes the earlier "density-aware `rank_m`" plan. Same data source (the routing
histogram), but used to feed real grouped-GEMM features instead of a scalar heuristic.*

This documents why the current perf model mis-ranks grouped MoE GEMM, and the concrete
change: feed it **grid_m (tile count) + padding**, not a scalar M.

---

## 1. The one-paragraph version

The perf model ranks configs from a **scalar M** (total routed rows). For a grouped
MoE GEMM that scalar hides everything that actually decides performance — how the work
is split across experts, the real tile count, padding, and how tiles pack onto the CUs.
So two very different grouped layouts with the same scalar M get ranked identically,
and the model over-prefers wide tiles (BN256) when the real grid says otherwise. The
fix is to compute the real grouped features from the routing histogram
(`grid_m = Σ_e ceil(M_e/block_m)`, padding fraction) and rank on **total blocks vs CUs
(occupancy)** — which handles sparse decode and dense prefill with one mechanism, no
sparse/dense heuristic.

---

## 2. Background (unchanged): grouped MoE GEMM and padding

- gpt-oss-120b is Mixture-of-Experts: 128 experts, each token routed to top-4.
- The experts' matmuls run together as a **grouped GEMM**: each expert e contributes
  `M_e` rows; they're stacked into one launch, tiled `block_m` rows at a time.
- **`block_m` is set by routing** (`max(16, min(next_pow2(tokens_per_expert), 128))`),
  not by the config picker. The picker chooses `block_n` (BN), `block_k`, `num_stages`,
  `num_warps`.
- **Padding**: each expert's rows are padded up to a multiple of `block_m`. In decode,
  ~1 token/expert still occupies a full `block_m`-row block → mostly padding.

Two regimes:
- **Decode (sparse)**: few tokens, ~1/expert, many tiny expert-blocks.
- **Prefill (dense)**: many tokens/expert, padding negligible.

---

## 3. Why scalar-M ranking is wrong for grouped GEMM

The model is asked `rank(M, N, K, block_m)`. It internally derives a grid from `M`
(`n_blocks(M, block_m)`) and estimates occupancy/cost. But `M` alone can't distinguish:

| | scalar M | real grid_m = Σ_e ceil(M_e/block_m) | optimal BN |
|---|---|---|---|
| bm16 decode | ~1024 (94% padding) | ~64 (active experts) | narrow (BN64/128) |
| bm64 conc128 | ~4440 (real, dense) | ~134 | narrow (BN128) — measured |
| bm64 conc16 | small | small | ~wash |

Feeding scalar M, the model widens BN **monotonically with M** (BN64→128→256). But the
real optimum tracks the **grid / occupancy**, not M. Measured contradictions the scalar
model can't explain:
- bm16 decode: scalar M looks large (padding) → model picks BN256; real work is sparse
  → BN64/128 is ~3× faster.
- bm64 conc128: model sees inflated M (`next_pow2` of padded) → grid_m≈253 → BN256; real
  grid_m≈134 → BN128 is faster.

**This is why the earlier "density ⇒ sparse→narrow, dense→wide" heuristic failed:**
dense (bm64/bm128) can want *narrow* BN too. Density (a ratio) is the wrong summary;
the **tile count + occupancy** is the right feature.

---

## 4. The fix: feed real grouped features, rank on occupancy

From the routing histogram (`routing_data.hist`, per-expert token counts `M_e`), compute:

```
grid_m   = Σ_e ceil(M_e / block_m)          # exact M-tile count (NOT n_blocks bound, NOT scalar M)
padded   = grid_m * block_m
useful   = Σ_e M_e
pad_frac = 1 - useful / padded              # wasted-compute fraction
```

Then rank each candidate BN by **how the real grid packs onto the CUs**:

```
total_blocks(BN) = grid_m * ceil(N / BN) * split_k
occupancy/wave-quantization from total_blocks vs 256 CUs   (the model already has this machinery)
+ pad_frac efficiency term
```

This handles both regimes with **one mechanism, no heuristic**:
- **bm16 decode**: `grid_m` small → BN256 gives too few blocks → low occupancy → model
  disprefers it → picks narrow BN. ✓
- **bm64 conc128**: real `grid_m≈134` (not the inflated 253) → correct occupancy → BN128. ✓
- **bm64 conc16**: small grid → both BN close → ~wash (matches measured). ✓

The density numerator **was** `grid_m` all along — we just stop dividing it out and
stop collapsing to a scalar.

---

## 5. The interface change (this is the real work)

A single scalar M cannot encode "small real work" + "specific tile count" + "padding"
simultaneously — that's the fundamental limitation. So this is **not** a selector-only
tweak; the perf model interface must accept the grouped features.

- **Interim (selector-only, better-than-today)**: pass `M_eff = grid_m * block_m` (the
  true padded total from the histogram) instead of `next_pow2(m)`. Less wrong, but
  still a scalar — a stopgap to sanity-check the direction, not the fix.
- **Principled (`.so` change)**: extend `GemmProblem` / `rank_configs` to take
  `grid_m` (tile count) and `pad_frac`, and compute occupancy from the real grid.
  See `grouped_gemm_interface_sketch.md`.

---

## 6. How we make sure it actually helps (calibration loop)

The model "helps" only if its ranking **matches measured reality across regimes**:

1. **Ground truth (Gap 1)** — matched-routing, per-config `avg_us` across
   `(block_m, N, K, density/conc)`, measured via `--enforce-eager` AB alternation
   (eager so graph-replayed decode kernels get profiler labels). Target set.
2. **Grouped-aware ranking** — feed `grid_m`/`pad_frac`, compute occupancy, calibrate
   to (1).
3. **Validate** — model top-1 BN must match the measured winner across the grid
   (small/large M, low/high density, each block_m) AND whole-surface geomean must hold.
   Any mispick is a calibration bug to fix, not a heuristic to add.

---

## 7. Order of operations

1. **Gap 1 first** — enforce-eager matched A/B (conc16 + conc128) → ground-truth BN
   preferences. (Without this we can't calibrate or validate.)
2. **Interim selector bridge** — `M_eff = grid_m * block_m`; confirm real-grid
   direction helps before touching the `.so`.
3. **Principled interface change** — feed `grid_m`/`pad_frac`; calibrate to (1);
   validate top-1 BN + geomean.
4. **Refinement (deferred)** — per-expert load-balance / tail-wave from the full
   histogram, if step 3's grid_m+occupancy leaves residual mispicks.

---

## 8. TL;DR

- The perf model ranks on **scalar M**, which hides the grouped structure → it
  over-prefers wide BN. Density (a ratio) was the wrong summary and its sparse/dense→BN
  rule is contradicted by data.
- Use the **same histogram** to compute the **real tile count `grid_m`** and
  `pad_frac`, and rank on **total-blocks-vs-CUs occupancy**. One mechanism, both
  regimes, no heuristic.
- Needs an **interface change** to feed those features (scalar can't carry them).
  Selector `M_eff = grid_m*block_m` is only an interim stopgap.
- Validate against **enforce-eager matched-routing ground truth** (Gap 1).

*Related: `density_aware_rank_m.md` (the superseded scalar-heuristic version — kept for
history), `grouped_gemm_interface_sketch.md`, `../FINDINGS.md`,
`../trace_tools/README.md`.*
