# pick_gemm_config() CPU overhead (2026-06-10)

Measured with `time.perf_counter` on gfx950 (CPU-side analytical call — NOT a
GPU op, so rocprofv3 is the wrong tool). Warm = steady-state median over 200 calls.

## Headline
- Global cold-start (first-ever call): **0.46 ms**
- Warm median across 17 shapes: **0.58 ms**  (min 0.09, max 9.7)

## The cost driver is OUTPUT-TILE COUNT, not candidate count

Candidate count is ~flat (152–220) for every shape. What scales is `rank_configs`,
and it tracks `grid_tiles = ceil(M/BM)*ceil(N/BN)`, NOT #candidates:

| shape          | grid_tiles | #cand | gen_ms | rank_ms | total_ms |
|----------------|-----------:|------:|-------:|--------:|---------:|
| moe-gating-M32 |          1 |   164 |   0.04 |    0.06 |     0.09 |
| square-1k      |         16 |   152 |   0.04 |    0.94 |     0.98 |
| square-2k      |         64 |   152 |   0.04 |    3.59 |     3.57 |
| output-proj-M4k|        192 |   152 |   0.04 |    6.95 |     7.05 |
| input-proj-M4k |        320 |   152 |   0.04 |    8.23 |     8.19 |
| input-proj-M8k |        640 |   152 |   0.04 |    9.56 |     9.61 |

**Why:** inside `rank_configs`, each candidate's cost estimate runs the formocast
L2 hit-rate simulation (`formocastL2HitRate`, ported from Origami) — a per-XCD
workgroup-launch walk that is **O(num_output_tiles) per candidate**. So:
`rank_ms ≈ #candidates × num_output_tiles × per-tile-sim-cost`.

Skinny/decode shapes have tiny grids (1–20 tiles) → cheap → sub-ms. Large
prefill shapes have huge grids (320–640 tiles) → ~10 ms. It is NOT
"memory-bound vs compute-bound"; it is small-grid vs large-grid.

## Optimization opportunity
The L2 WG-sim result depends only on (BM, BN, grid, numXCDs), not the full
config — many candidates share the same (BM,BN). Memoizing the sim per
(BM,BN,M,N) would collapse the large-shape cost (the ~150 candidates reuse a
handful of distinct tile geometries).

Even un-optimized, 9.7 ms worst case is 1-2 orders of magnitude below autotune
(which GPU-benchmarks N configs × warmup+timing each, seconds per shape).
