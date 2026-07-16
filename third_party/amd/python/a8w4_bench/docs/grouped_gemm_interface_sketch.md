# Interface sketch: feeding grouped-GEMM tile count to the perf model

Concrete change to make `rank_configs` grouped-aware. Grounded in the actual source:
`third_party/amd/.../PerfModel.cpp`, `include/.../PerfModel.h`, `python/PerfModelBindings.cpp`.

## The bug, in one line

`PerfModel.cpp:1263`
```cpp
est.totalOutputTiles =
    ceildiv(prob.M, cfg.blockM) * ceildiv(prob.N, cfg.blockN) * prob.batchSize;
est.numWaves = ceildiv(est.totalOutputTiles, hw.numCUs);   // 1267
```
`ceildiv(prob.M, blockM)` is the **dense** tile count — it assumes all M rows are one
contiguous group. For grouped MoE the real M-tile count is
`gridM = Σ_e ceil(M_e / block_m)` (per-expert padding creates extra partial tiles).
`numWaves` (hence occupancy, tail-wave efficiency, and the whole BN ranking) is
therefore computed on the wrong grid. Same wrong derivation at `:1052`
(`selectGroupSizeM`) and `:1385` (L2 hit-rate `gridM`).

## Why a precomputed scalar works here

`gridM` depends on `block_m` and the routing histogram — **not** on `block_n`. In
`pick_a8w4`, `block_m` is fixed by routing and constant across all ranked candidates
(only BN/BK/ns/nw vary). So `gridM` is a **single precomputed value** for the whole
`rank_configs` call — it can live in `GemmProblem`, no per-candidate recompute.
(General non-MoE case where blockM varies would need recompute; a8w4 MoE doesn't.)

## Change 1 — struct (`PerfModel.h`, after line ~188)

```cpp
struct GemmProblem {
  int64_t M = 0, N = 0, K = 0;
  int64_t batchSize = 1;
  // Grouped-GEMM: real M-tile count = Σ_e ceil(M_e/block_m). 0 = derive from M
  // (dense fallback, current behavior). Set by MoE callers from the routing hist.
  int64_t gridMTiles = 0;
  double  padFrac = 0.0;   // wasted-compute fraction: 1 - Σ M_e / (gridMTiles*block_m)
  ...
};
```

## Change 2 — use it wherever gridM is derived (`PerfModel.cpp`)

Add a helper and route the 3 sites through it:
```cpp
static inline int64_t gemmGridM(const GemmProblem &p, int64_t blockM) {
  return p.gridMTiles > 0 ? p.gridMTiles : ceildiv(p.M, blockM);
}
```
- `:1052`  `const int64_t gridM = gemmGridM(prob, cfg.blockM);`
- `:1263`  `est.totalOutputTiles = gemmGridM(prob, cfg.blockM) * ceildiv(prob.N, cfg.blockN) * prob.batchSize;`
- `:1385`  `const int64_t gridM = gemmGridM(prob, cfg.blockM);`

That alone fixes numWaves/occupancy/tail — the dominant BN driver. `padFrac` is an
optional second step: scale the compute-bound cycles by `1/(1-padFrac)` (or add a
wasted-tile term) so padding-heavy sparse decode isn't credited as useful FLOPs.

## Change 3 — bindings (`PerfModelBindings.cpp:67-92`)

```cpp
py::class_<GemmProblem>(m, "GemmProblem")
    .def(py::init([](int64_t M, int64_t N, int64_t K, ...,
                     int64_t gridMTiles, double padFrac){
       GemmProblem p; ...; p.gridMTiles = gridMTiles; p.padFrac = padFrac; return p; }),
         py::arg("M"), ..., py::arg("grid_m_tiles")=0, py::arg("pad_frac")=0.0)
    .def_readwrite("grid_m_tiles", &GemmProblem::gridMTiles)
    .def_readwrite("pad_frac",     &GemmProblem::padFrac);
```
Defaulted args ⇒ every existing caller is unchanged (dense fallback).

## Change 4 — selector (`perfmodel_a8w4_select.py`)

Compute the real grid from the routing hist and pass it:
```python
def _grid_m_from_hist(hist, block_m):
    # hist[e] = tokens routed to expert e (already captured for AITER_A8W4_HIST_LOG)
    import math
    tiles = sum((int(m)+block_m-1)//block_m for m in hist if m)
    useful = sum(int(m) for m in hist)
    padded = tiles*block_m
    return tiles, (1.0 - useful/padded) if padded else 0.0

# in pick_a8w4, when hist available:
gm, pf = _grid_m_from_hist(hist, block_m)
prob = _pm.GemmProblem(rank_m, n, k, _F8,_F4,_F32, 8,4,32, grid_m_tiles=gm, pad_frac=pf)
```
`rank_m` (the scalar M) still carries the FLOP magnitude; `grid_m_tiles` now carries the
real tile count for occupancy. When `hist` is absent, pass 0 → dense fallback (today).

## Interim (no `.so` rebuild) — for a quick direction check

Selector-only: pass `M_eff = gridMTiles * block_m` as the scalar M instead of
`next_pow2(m)`. Makes `ceildiv(M_eff, blockM) = gridMTiles`, so the *dense* derivation
lands on the real tile count. Approximate (still couples FLOPs and grid through one
scalar, and the `n_blocks` bound vs exact differs), but strictly better than
`next_pow2(padded)` and needs no rebuild — use it to confirm the direction before the
proper interface change.

## Validation

Rebuild standalone (`build_standalone_perf_model.sh`), then check top-1 BN matches the
Gap-1 enforce-eager ground truth across (block_m, N,K, density), and re-run the
whole-surface sweep: geomean ≥ current cap (0.988) with tp8/conc128 restored and no new
regressions. bf16 path must stay byte-identical (gridMTiles=0 there ⇒ unchanged).
```
