# gpt-oss-120b W4A8 MI355X — on-the-fly PerfModel a8w4 vs tuned-JSON dispatch: measured attribution

**Date:** 2026-07-13
**Container/build:** `xguo-nightly-latest` = `vllm/vllm-openai-rocm:nightly`, vllm `0.23.1rc1.dev786+g34b560b72`, triton `3.6.0` (stock)
**perf_model:** standalone `perf_model.so` rev `218917094+dirty` (branch `gluon_perf_model`)
**Recipe:** `benchmarks/single_node/fixed_seq_len/gptoss_fp4_mi355x.sh`, `RANDOM_RANGE_RATIO=0.8`, cold cache per cell (IX-CI parity)
**Matrix:** TP{1,2,4,8} × CONC{4,8,16,32,64,128,256} × shape{8192/1024, 1024/1024}

## TL;DR — the nuanced conclusion (do NOT collapse this)

1. **On-the-fly PerfModel (canonical_m) REGRESSES vs tuned-JSON on the realistic recipe.**
   Geomean PM/JSON = **0.951** over 56 cells (32/56 below parity). This *supersedes* the earlier
   "PM matches/beats JSON e2e" result, which was measured at `RANDOM_RANGE_RATIO=0.0`
   (fixed-length → near-uniform routing) — the classic microbench-not-representative trap.
2. **exact-m is useful but NOT safe as a blanket policy.** Over the 32 losing cells it lifts
   geomean 0.909 → 0.946 (recovers ~40% of the gap), fully fixes the TP4/8K cells, but
   **regresses several B-shape cells** (e.g. tp8/conc16/1K 0.915 → 0.821). "Just use exact-m" would
   trade C-gains for B-losses.
3. **canonical_m is too lossy, but it is NOT the only problem.** Candidate-generation and
   ranking gaps dominate the hard residual that exact-m cannot touch.

## Result 1 — PM-canonical vs JSON (full surface, 56 paired cells)

geomean PM/JSON by TP:  TP1 **1.007** · TP2 0.997 · TP4 **0.867** · TP8 0.941
by shape: 8192/1024 = 0.966 · 1024/1024 = 0.937 · by CONC: flat 0.93–0.98
All damage is swizzle-off **TP4/TP8**; TP1/2 (swizzle-on) are parity-or-better.

## Result 2 — exact-m measured pass (32 losing cells)

geomean over losing cells: canon/json **0.909** → exact/json **0.946**.
Verdict histogram: C-confirmed 8, C-mostly 4, mixed 10, B/A-confirmed 10.
- TP4/8K → **C-confirmed** (exact fully recovers: 0.879→1.007, 0.881→1.016).
- TP4/1K → mixed (partial).
- TP8 → mostly B/A (no help; exact *regresses* some).

## Root-cause attribution (per unique A8W4 shape; estimate + measured backed)

| class | #shapes | mechanism | fix |
|---|---|---|---|
| **A grid/no-entry** | 8 | 6 TP4/8-sharded shapes (`n1024_k3072`, `n3072_k512`) have **no tuned JSON entry** (freq 850k+); 2 `BK=128` winners PM can't emit | grid expansion / tuned-table fallback |
| **B ranking** | 6 | `bm128` `nw=8` where tuner uses `nw=4`; `estimate_perf` over-predicts nw=8 (1050 vs 788 TFLOPS). Present even at TP1 swizzle-on | PerfModel.cpp num_warps calibration |
| **C cache/canonical** | 5 | `bm16/32/64` where model *itself* ranks JSON higher at exact-m (e.g. `bm64 n1536_k3072` est JSON 762 vs PM 411) but canonical picked worse | selector bucketing (no model change) |
| match | 9 | PM == JSON | — |

Candidate grid PM cannot reach 41% (20/48) of tuned winners: nonkdim=32 (15), BK<256 (12), wpe≠0 (12), ns=1 (3).

## Direction — fix order (agreed, risk-ordered)

1. **[done] Findings/memory writeup.**
2. **C selector fix** — bounded `m_bucket` / *selective* exact-m for C-class shapes ONLY, guarded so it
   never applies where exact-m regresses (B-shapes). Lowest risk, no model change, recovers TP4/8K.
   Track: cells recovered / unchanged / B-regressions-avoided / unique-config count / JIT count.
   **Do NOT implement "exact-m everywhere."**
3. **B PerfModel num_warps calibration** for `bm128` nw=8 over-ranking. Higher impact, riskier (global model change).
4. **A grid expansion / tuned-table fallback** last — highest risk (creates new choices across many shapes);
   only pull earlier if one missing-grid shape dominates the remaining e2e gap.

## Artifacts (in this dir)
- `per_cell.csv` (144 rows: json/pmcanon/pmexact × 56/56/32 cells) — throughput, TTFT/TPOT/ITL/e2el, meta.
- `per_a8w4.csv` (416 rows) — per cell×shape: PM cfg, JSON cfg, in-grid + excluding field, estimate_perf(PM)&(JSON), freq.
- `run_meta.txt`, `parse_db.py`, `aggregate.py`, `run.sh` (112-cell), `run_exact.sh` (32-cell).
- JSON baseline (same build, partial): `../full_nightly_34b560/`. Tuned table: aiter `configs/moe/gfx950-A8W4.json` (48 entries).

## UPDATE 2026-07-13 — (1) C selector fix MEASURED (guarded m-bucket)

Implemented in `perfmodel_a8w4_select.py`: `block_m<=64` ranks at `next_pow2(real_m)` (bounded ~2 buckets);
`block_m==128` keeps canonical (defer to B). Root cause: canonical_m=block_m*32 is 2-23x below real routed-rows M
-> mis-ranks tile size. Measured on the 31 losing cells (arm `pmbucket`, `run_bucket.sh`):
- 22 recovered / 5 unchanged / 4 REGRESSED; geomean over losers 0.911 -> 0.955.
- Whole-surface geomean PM/JSON **0.951 -> 0.976** (below-parity 32->23/56). TP4 **0.867 -> 0.948** (fixed); TP8 0.941->0.955; TP1 1.007->1.010; TP2 0.997->0.994.
- 4 regressions (all nw=8-triggered / B-bug): tp8/conc16/1K 0.900, tp8/conc16/8K 0.941, tp8/conc8/8K 0.969, tp2/conc256/1K 0.942.
Conclusion: C fix is a net measured win (no model change), halves the gap. Residual + the 4 regressions are the B (nw=8) and A (grid) causes -> proceed to (2) num_warps calibration, which should fix the regressions AND lift recovered cells.

## UPDATE 2026-07-13 — (2) B fix attempt: nw-cap occupancy cap is HARMFUL (reverted)

Attempted B fix: cap the memory-bound occupancy penalty by achieved-supply (occForPenalty =
min(vgprOccupancy, max(est.occupancy, 0.25))), hasMxScales-gated. On checkpoint branch
`a8w4-nw-occupancy-cap` (NOT committed, NOT deployed).
- Analytically correct: ties bm128 nw4/nw8 (525.3=525.3), bf16/dense byte-identical.
- Verified (per-config OCCDBG): the line-1802 memory-bound correction DOES fire for BN256
  (penBefore 1.0 -> penAfter 4.0); BN256 pays the same penalty as BN128. The wide-BN over-credit
  is NOT the penalty — it is numWaves halved (6 vs 12) while per-tile effTile barely grows
  (memoryCycles scales ~6% for 2x blockN). Measured microbench (a8w4_dataset.json, uniform M=4096)
  says BN256 is ~20% FASTER than BN128 for bm128 shapes -> model direction not obviously wrong.
- e2e (bucket + nw-cap, arm pmbcnw): CATASTROPHIC on TP1/TP2 — geomean bcnw/json 0.731 vs canon
  0.972; tp1/conc32 1.015(bucket)->0.600(bcnw). Killed at 5/32. The occupancy penalty is load-bearing
  for the broader MX-scaled ranking; capping it globally breaks many shapes.
CONCLUSION: occupancy is the WRONG lever for the B fix. Reverted deployed .so to canonical
(218917094+dirty). Keep the C bucket selector (net win 0.951->0.976). B/wide-BN residual: attack via
M/routing-dependent wide-BN memoryCycles scaling (memoryCycles must grow with blockN), carefully
validated e2e — NOT via occupancy.

## UPDATE 2026-07-14 — DECISIVE: isolated a8w4 GEMM bench is NON-AUTHORITATIVE

Profiled PM-vs-JSON on TP8 no-json cell (isl8192/conc64), cold cache both arms, kernel-level.
Per-call avg with IDENTICAL call counts (n=115200 both) + hipBLAS PM-independent control:
  _moe_gemm_a8w4: json 51.92us/call -> pm 61.54us/call = PM 1.185x SLOWER
  hipBLAS (ref, consistent shape): PM/JSON 0.993 -> NO global/thermal slowdown confound
  (Attn 1.32x, RMSNorm 1.26x are NOT clean refs: per-call cost varies with captured prefill/decode mix.)
Contrast with isolated microbench (ALL forms): PM 25-27% FASTER (uniform M, real M+swizzle, real-skew replay).
=> Same config (128/256/2/8), same shape/M/skew: 25% faster ALONE, 18% slower IN SERVING.
DECISION RULE (kernel slower under PM) => isolated a8w4 GEMM bench is NON-AUTHORITATIVE for ranking.
Serving-context effect (cache/memory contention with co-resident attention/norm/comms/weights) that no
isolated GEMM reproduces. Ruled out earlier: graph-capture (eager per-call), JIT (equal counts),
routing-skew (real-histogram replay still PM-faster).
IMPLICATIONS:
  1. no-json e2e loss is real at GEMM level (not whole-cell artifact); JSON fallback 256/256/2/4 genuinely
     beats PM 128/256/2/8 in serving.
  2. PerfModel a8w4 ranking cannot be validated/calibrated by ANY isolated GEMM bench; only e2e is truth.
  3. Defensible deliverable = C bucket selector (0.951->0.976), measured purely e2e.

## UPDATE 2026-07-14 (2) — ORDER-SWAP: e2e kernel profile is UNUSABLE (variance-dominated)
Order-swap [json,pm,pm,json], cold each, 90s cooldowns, hipBLAS control:
  MoE _moe_gemm_a8w4 per-call PM/JSON:  json-first=0.855 (PM faster)  pm-first=1.191 (PM slower)  -> FLIPS with order
  hipBLAS ctrl: 0.973 / 0.977 (stable -> NO thermal/global confound)
  Root: JSON MoE per-call swings 76.25us(n=75456) <-> 52.77us(n=114048) with run position: the 200-iter
  profiler window captured different prefill-vs-decode M-mix each run; per-call avg is a shifting mix, not
  config-clean. hipBLAS stable (consistent shape) confirms it's M-mix noise, not thermal.
DECISION RULE (flips with order) => e2e MoE kernel profile UNUSABLE. Conclude nothing from it.
FINAL: three measurement paths exhausted -- isolated bench (non-representative), whole-cell e2e (aggregate,
not per-config), e2e-kernel profile (order-variance). NO trustworthy per-config a8w4 ground truth exists in
serving. a8w4 RANKING is measurement-blocked, not model-blocked. Ship C bucket (0.976); do not chase the
ranking/no-json residual further with benches or per-kernel profiling.

## UPDATE 2026-07-14 (3) — RESOLVED via per-config labels: PM no-json GEMM is FASTER
Added env-gated record_function label per a8w4 launch (shape+config); gpu_user_annotation carries the
kernel's GPU dur. Verified: 1 kernel per annotation, annotation.dur == kernel.dur (158us sampled).
Labeled order-swap [json,pm,pm,json], per (N,K,BM=128) no-json shapes, BOTH orders consistent:
  (1024,3072,128): JSON BN256/nw4 214us vs PM BN128/nw8 153us -> PM/JSON 0.717 / 0.713
  (3072,512,128):  JSON BN256/nw4 193us vs PM BN128/nw8 141us -> PM/JSON 0.730 / 0.727
=> PM's no-json bm128 PREFILL config is ~28% FASTER at kernel level in e2e (isolated bench was RIGHT here).
The earlier "PM slower" reads were M-mix confounds: kernel-NAME aggregate (1.16x) mixed prefill-triton
(PM-affected, 587) with DECODE-GLUON (_moe_gemm_a8w4_decode_gluon via get_kernel_config_gluon, PM does NOT
touch -> identical both arms). Per-config annotation removes the mix.
DECISION RULE (PM per-call faster, whole-cell slower) => no-json bucket is NOT the culprit; cell slowdown is
mixed-shape/decode-path/global. CORRECTS the 2026-07-14(1) "isolated bench non-authoritative" conclusion:
for the prefill no-json GEMM the isolated bench matches e2e; the confound was aggregation, not the bench.
Net: PM DOES win the no-json prefill GEMM (the intended value); residual cell loss is elsewhere.

## UPDATE 2026-07-14 (4) — SLOWDOWN LOCATED: DECODE a8w4 GEMM (C-bucket BN256 on sparse decode)
Whole-cell breakdown of decode no-json cell (isl1024/conc16 TP8), per DECODE-stage per-call avg (json-first):
  Other/_moe_gemm_a8w4:  json 8.20us -> pm 18.97us = 2.31x SLOWER  <== the ONLY slowdown
  ALL other kernels flat 0.99-1.01x: CustomAR comms, _reduce_grouped(0.993), _downcast(0.994),
  RMSNorm(0.996), _topk, Attn2D(1.000), rocBLAS, MoE Reduce, etc.
  => earlier "adjacent kernels +12-29%" was M-MIX confound; per-stage they are flat.
Config traces: PM decode picks BN256 for bm64 (40-65us) via C-bucket next_pow2(total_m)~4096 -> wide tile;
JSON uses narrower. Deployed bucket: bm32->BN256, bm64->BN256/nw8 vs canonical bm32/64->BN128.
ROOT: C-bucket over-buckets small block_m decode -> BN256; real decode routing is SPARSE (few tokens/expert)
so wide BN256 wastes ~half the tile -> 2.3x slower a8w4 GEMM. This IS the C-bucket's 4 TP8-decode regressions.
RECONCILED: prefill bm128 PM 0.72x FASTER; decode bm32/64 PM 2.3x SLOWER -> net ~0.94 no-json aggregate.
FIX: restrict bucketing to prefill/large-M; keep canonical BN128 for small-block_m decode (sparse -> narrow BN).
Should recover the 4 decode regressions while keeping prefill/TP4 wins.

## 2026-06-26 — Forced-config A/B resolves it: PM configs are per-shape FASTER

Method: single-run alternating-config A/B (AITER_A8W4_AB in perfmodel_a8w4_select.py).
For a target shape, pick_a8w4 alternates JSON-config (A) and PM-bucket-config (B)
call-by-call in ONE serve => both measured under IDENTICAL routing/window/thermal.
The 587 record_function label encodes the returned config, so parse_ab.py
disaggregates A vs B. Cell = tp8/conc16/isl1024 (worst-hurt decode cell). 2 runs.

RESULT (reproducible across both runs, 144 calls each config):
  bm64 n1024_k3072: PM BN256/ns2/nw8 = 65.2us  vs JSON BN128/ns1/nw4 = 74.9us -> PM 13% FASTER (0.871/0.874)
  bm64 n3072_k512 : PM BN256/ns2/nw8 = 40.9us  vs JSON BN128/ns1/nw4 = 44.2us -> PM  7% FASTER (0.926/0.928)
  bm128           : PM BN128/ns2/nw8 = 151us   vs JSON BN256 = 199us          -> PM 24% FASTER (0.76)

CONCLUSION (user's decision rule: PM wins shape-by-shape but cell loses => NOT config ranking):
  * PM's C-bucket per-shape KERNEL configs are FASTER, not slower. Config ranking is GOOD.
  * The earlier "PM decode 2.3x slower" was a ROUTING-MIX ARTIFACT of two separate arms
    with different random prompts (JSON arm produced ZERO bm64 calls; PM arm had them).
  * Selection CPU overhead: pick_a8w4 2.38us vs JSON 1.42us = +0.95us/call
    (~68us/decode-step over ~72 a8w4 calls; ~1.3% of the ~5.2ms step, async-hidden). Too small.
  * PRIME SUSPECT for residual cell-level deltas: run-to-run routing variance between
    the separate json-arm and pm-arm (no bench --seed available to pin routing).

NEXT (confirm variance): json-vs-json control (same arm, 2x different random prompts);
  if json/json spread ~= the json/pm "regression", the cell delta is routing variance, not PM.

## 2026-06-26 — CPU+GPU critical-path decode decomposition (decompose.py)

Tool: decompose.py buckets GPU kernels + CPU cuda_runtime into each decode step's
own CPU-annotation interval (execute_context_..._generation_16(16)); single GPU
stream verified => gpu_busy=sum(kernel dur), gpu_idle=wall-busy = host-dispatch bubble.
Categories: moe_gemm / comm / attention / dense_gemm / moe_aux / other. Per-step.

DECODE cell tp8/conc16/isl1024, per-step (json=pos1, pm=pos2, pm=pos3):
  metric          json    pm2     pm3
  step_wall(us)   1424    1463    1455
  gpu_busy        1406    1446    1448
  gpu_idle          13     8.8     0.7   <- decode is ~99% GPU-BOUND
  moe_calls         30      25      25
  moe_us/call     8.19   18.75   18.97
  comm_us/call    8.28    8.24    8.82   <- identical => comm unaffected

A-E classification of the MoE decode delta:
  A kernel slower : per-shape forced A/B (matched routing) = PM FASTER on bm64/bm128.
  B idle gaps     : NO -- gpu_idle < 13us of 1420us; fully host-hidden.
  C CPU dispatch  : NO -- cpu_pick hidden (0 in decode window), GPU-bound.
  D lost overlap  : N/A -- single GPU stream, comm serialized, comm/call identical.
  E work/routing  : the whole json-vs-pm delta lives here (reproducible pm2==pm3).

MoE-duration histogram (moe_hist.py) exposed the real mechanism:
  json: 50% of moe calls <6us + 50% 10-20us  (avg 8.2us)
  pm  : 0% <6us, 45% 10-20us, 50% 20-30us    (avg 18.8us)  [pm2==pm3]
PM picklog ground truth: decode is bm16 (57%) + bm32 (34%) dominated; bm64 only 8%.
  => My earlier forced A/B tested bm64 (minority). PM's C-bucket escalates the
     DOMINANT bm16/bm32 to BN256 where JSON stays narrow:
       bm16 n2880 k360: JSON BN64  vs PM BN256   (36864 calls, dominant)
       bm32 n1024 k3072: JSON BN128 vs PM BN256  (36288 calls)
  This turns fast <6us narrow-BN decode calls into 20-30us wide-BN calls.
NEXT: forced A/B (run_ab2) on the DOMINANT bm16/bm32 decode shapes to measure
  JSON-config vs PM-BN256 at matched routing -- the decisive test I initially missed.

## 2026-06-26 — ROOT CAUSE of decode regression: padded-m escalates small block_m to BN256

Forced A/B on the DOMINANT decode block_m (bm16 57% + bm32 34%), matched routing
(ab_decode2, 50/50 alternation verified in picklog):
  moe_us_per_call: pure-JSON 8.2 | AB(half/half) 12.4 | pure-PM 18.8
  => interpolated JSON-narrow ~6us vs PM-BN256 ~18.8us => PM ~3x SLOWER on bm16/bm32.
  moe_hist: forcing half back to JSON-narrow makes the <6us fast cluster REAPPEAR (25%).

WHY PM escalates (pure-PM picklog, natural picks):
  bm16 n2880 k360 : m>=512  -> BN256 (36864 calls)   [m<512 -> BN64/128, fine]
  bm32 n1024 k3072: m>=1024 -> BN256 (36288, ALL)
  m = PADDED routed rows. Decode: ~64 real tokens over ~64 experts, each padded to
  block_m => m~1024 mostly PADDING. next_pow2(m) sees "large M" => perf model picks
  wide BN256, but useful work is sparse => wasted compute on padding.
  JSON's block_m heuristic caps small block_m to narrow BN => avoids it.

CLASSIFICATION (final): the decode delta is component A (kernel slower) but ONLY for
bm16/bm32 via the SELECTOR's padded-m ranking; NOT B/C/D (decode GPU-bound, no bubbles,
single stream, selection hidden). bm64/bm128: PM faster (earlier A/B). So C-bucket is
right for bm64/bm128, wrong for bm16/bm32.

FIX OPTIONS (data-grounded):
  a. Cap block_n<=128 for block_m<=32 in the C-bucket grid (simple; matches JSON decode).
  b. Density-aware rank_m: use routing hist (blocks/active_expert); sparse (decode) ->
     rank at REAL m -> narrow BN; dense (prefill) -> allow BN256 (root-cause; needs hist).
  c. Revert block_m<=16 to canonical (fixes bm16 only; bm32 canonical=1024 still BN256).
Recommend (a) now + validate whole-surface geomean; (b) as principled version if hist reliable.

## 2026-06-26 — FIX implemented: cap block_n<=128 for block_m<=32 (option a)

perfmodel_a8w4_select.py `_pick`: `bn_cap = 128 if block_m<=32 else 512`, applied to
the candidate grid (AITER_A8W4_BNCAP=0 disables for A/B). block_m>=64 unchanged.

Validation done:
  1. Picklog sanity (offline + IN-SERVING): bm16 -> BN128/BN64, bm32 -> BN128, ZERO
     BN256; bm64 still {64,128,256}, bm128 {128} unchanged. Confirmed in live sweep picklog.
  2. Matched-cell decode (tp8/conc16/1024), fixed vs unfixed(BNCAP=0), same script:
       moe_us_per_call 18.99 -> 13.06 (-31%); moe_hist 20-30us BN256 cluster 50% -> 0%.
       total tput 2676 -> 2699 (+0.9% e2e, routing-masked: fixed run drew heavier
       routing 27 vs 23 moe_calls; matched-routing math ~= -89us/step moe ~ +6%).
  3. Whole-surface: RUNNING run_capped.sh (56 cells, pmcapped arm, ~4-5h) vs existing
     JSON baseline. parse_db.py + aggregate.py updated to fold in `pmcapped` (new
     section: whole-surface geomean, per-TP/CONC, the 5 prev-regressed cells cap-vs-
     bucket, worst-10). Expect geomean >= 0.976 (bucket) with the 5 regressions reduced.

Analysis after sweep: `python3 parse_db.py && python3 aggregate.py` in the DB dir.

## 2026-06-26 — CAP fix whole-surface sweep DONE (56 cells, 0 failures)

parse_db.py + aggregate.py, pmcapped arm vs existing json baseline.
WHOLE SURFACE: canon/json 0.9514 -> cap/json 0.9883 (+3.7pts, best of 3 arms).
All 5 target regressions recovered:
  tp8/conc16/1024  bkt 0.823 -> cap 0.977 (+18.8%)
  tp8/conc16/8192  0.882 -> 0.984
  tp2/conc256/1024 0.911 -> 0.980
  tp8/conc8/8192   0.951 -> 0.970
  tp2/conc64/1024  0.957 -> 0.967
Cap helps vs canon almost everywhere (+1..15%).

BUT the blunt cap introduced a serious NEW regression:
  tp8/conc128/1024: canon 1.056 -> cap 0.827 (cap/canon 0.784, -21.6%)
  (+ mild tp1/conc4/8192 0.997->0.960)
Cause: conc128 => M=512 => ~4 tok/expert => routing block_m=16, but routing is
DENSE so bm16 wants BN256; the cap can't tell dense-conc128 from sparse-conc16
(both block_m=16) and forces BN128 => craters a cell canon had ABOVE parity.
=> This is the empirical case FOR density-aware rank_m: narrow for sparse, keep
   BN256 for dense. Density fixes the cap's one regression AND lifts the surface.

Still below parity 22/56, TWO causes:
  (1) cap bluntness (tp8/conc128) -> density fixes.
  (2) pre-existing TP4/TP8 decode weakness (canon already <1, e.g. tp4/conc16 0.846;
      cap improved to 0.94 but not parity) -> separate no-json/grid_n/split-K issue,
      NOT the BN256 bug. Ties to the n_blocks short-circuit + BN>=64/split_k=1 fusion cap.

DECISION: do NOT ship the blunt cap as-is (tp8/conc128 -21.6%). Next: density-aware
rank_m (#2) -- keep cap wins, drop its regression, push toward parity.

## 2026-06-26 — CORRECTION: tp8/conc128/1024 regression root cause (picklog-confirmed)

Earlier I mis-attributed this to "dense bm16 wants BN256, cap forces BN128." WRONG.
Picklog diff (pmcanon vs pmcapped, same cell):
  CANON  bm64 -> BN128 (259200 calls)  => cell 1.056
  CAPPED bm64 -> BN256 (251712 calls)  => cell 0.827
The BN cap only touches block_m<=32, so it did NOT cause this. The flip is the
C-BUCKET's next_pow2(m): at conc128 bm64's REAL m is >=4096 (4440 for n1024_k3072,
8308 for n3072_k512 -- genuinely DENSE, not padding). canonical_m ranks bm64 at fixed
64*32=2048 -> model picks BN128; next_pow2(m)=8192 -> model picks BN256. Empirically
bm64 BN128 > BN256 here (1.056 vs 0.827).

=> This is a PERF-MODEL RANKING inaccuracy (B bucket), NOT density/padding. And it's
INVERTED vs the conc16 forced A/B (which proved bm64 wants BN256 at small m): for these
bm64 no-json shapes reality is "BN256 at small M -> BN128 at large M", but the model does
the opposite (BN128 small M -> BN256 large M). The wide-BN memoryCycles-vs-M scaling
calibration target. Density fix does NOT address this (bm64 genuinely dense).
Fix options: (a) perf-model wide-BN@large-M calibration (principled, whole B class);
(b) restrict C-bucket next_pow2 to block_m<=32, keep bm64 canonical (needs conc16 A/B
to confirm no low-conc bm64 regression).

## 2026-06-26 — bm64 forced A/B (grouped-MoE scalar-M mismatch investigation)

Framing (corrected): this is GROUPED MoE, not dense GEMM. PM ranks on scalar (padded)
M; grouped perf also depends on per-expert M_i dist, active-expert count, padding,
tile count/expert, tail-wave balance. Hypothesis: at large scalar M, PM prefers BN256
but real grouped layout prefers BN128 => "grouped-MoE scalar-M ranking mismatch".

CONC16 matched A/B (AITER_A8W4_AB, canon BN128/ns2/nw4 vs capped BN256/ns2/nw8, 144 calls each):
  bm64 n1024_k3072: BN128 61.5us vs BN256 65.4us -> BN128 +6.4%
  bm64 n3072_k512 : BN128 42.8us vs BN256 41.1us -> BN256 +3.9%
  => small-M: MIXED / close (~+-6%), no clear winner. (Earlier "BN256 wins conc16" was
     confounded by ns1/nw4 vs ns2/nw8; with matched ns2 it's a wash.)

CONC128: bm64 labels don't attach (decode graph-replayed, not eager) -> can't use
per-config annotation. Using whole-cell force A/B instead: capped-PM everywhere, bm64
FORCED BN128 vs FORCED BN256 (only bm64 differs). [running]
Prior sweep signal: tp8/conc128/1024 canon(bm64 BN128) 1.056 vs capped(bm64 BN256) 0.827,
bm64 the dominant differing shape => strong pre-indication BN128 >> BN256 at large M.

## 2026-06-26 — Gap 1 (enforce-eager matched A/B) results + measurement limits

Method: AITER_A8W4_AB alternation + --enforce-eager (so graph-replay decode kernels
launch eagerly and get 594 record_function labels) + small profiler window (40 iters).
gfx950 has NO gluon path (use_gluon = gfx1250 only, line 382) -> everything is triton 594.

RESULTS:
- conc16 bm64 (labeled, matched, first ab_decode run): BN128 vs BN256 = WASH
  (n1024_k3072 61.5 vs 65.4; n3072_k512 42.8 vs 41.1; ~+-6%).
- conc128 bm64: COULD NOT isolate. bm64 for n1024_k3072/n3072_k512 (65088 picks each in
  picklog) occurs in a transient batch-fill regime (tokens/expert in (32,64]) OUTSIDE the
  profiler window (captured at bench start = prefill bm128 + early decode bm16). 0 bm64
  labels despite enforce-eager. Whole-cell force runs unreliable (variance + infra hangs).
- conc128 bm16 (labeled, matched, CLEAN -- accidental but decisive):
    n720_k2880 : BN64 11.2us vs BN128 14.4us -> BN64 1.28x faster
    n2880_k360 : BN64  7.2us vs BN128 23.8us -> BN64 3.31x faster
  PM (and the cap, which forces bm16->BN128) pick BN128; BN64 is 1.3-3.3x FASTER.
  => real mis-ranking, same grouped-occupancy cause (BN64 = 2x grid_n = more blocks).
  The cap partially helped n2880_k360 (BN256->BN128) but the true optimum is BN64.

MEASUREMENT LESSON: per-shape bm64 decode labels are hard to capture (rare + mistimed
vs profiler window). Reliable validation = WHOLE-SURFACE SWEEP (ran fine for 56 cells),
not per-shape labels. The bm16 conc128 data already validates the grouped-aware direction.

DOCS: docs/grouped_gemm_ranking.md (grouped-feature framing, supersedes density heuristic),
docs/grouped_gemm_interface_sketch.md (concrete GemmProblem.gridMTiles + PerfModel.cpp:1263
gemmGridM() change + selector hist computation).

## 2026-06-26 — Selector policy refinement (pmpol): evidence-based ALLOWED_BN_BY_BM

Adopted the "keep C-bucket selector + candidate constraints" approach (not PM redesign;
the scalar grouped-feature bridge dry-run was insufficient/inconsistent, and the real
grouped-aware PM fix needs a risky cost-model change).

perfmodel_a8w4_select.py: `_ALLOWED_BN_BY_BM = {16:(64,), 32:(64,128)}`; _pick filters
the candidate grid to allowed BN per block_m; PM ranks within it. bm64/bm128 = full grid.
Cache fix: `bncap` passed into _pick (part of lru_cache key) so AITER_A8W4_BNCAP toggles
correctly within one process.

Rationale:
  bm16 -> BN64 only: measured (Gap-1 enforce-eager, matched routing, conc128) BN64 is
    1.3-3.3x faster than BN128 (n2880_k360 3.31x); conc16 PM already picks BN64. NOTE:
    the OLD cap allowed (64,128) but PM picks BN128 within it -> misses the win; (64,)
    captures it. Evidence is decode/conc128-heavy; reconfirm if bm16 appears denser.
  bm32 -> BN<=128: the validated cap (0.988); bm32 n1024_k3072 picked BN256 for all calls
    in the fixed regression. No isolated bm32 A/B -> keep <=128, loosen only if proven.
  bm64/bm128 -> full: conc16 bm64 wash, conc128 unmeasurable -> don't touch until A/B.

Offline verify PASSED: policy on -> bm16 [64], bm32 [64,128], bm64 [64,256], bm128 [128];
BNCAP=0 toggle in-process flips bm16 to [64,128,256] (cache-key fix works).

Net change vs deployed cap (0.988): ONLY bm16 tightens (<=128 -> BN64 only). Sweep tag
`pmpol` running to validate vs pmcapped 0.988 / canonical 0.951; expect decode/bm16 cells
up, no new regressions. bm32/bm64 A/B deferred as follow-ups.

## 2026-06-26 — TP4 bm16 A/B CONTRADICTS e2e -> enforce-eager decode A/B is INVALID

Ran isolated TP4 bm16 A/B (enforce-eager, matched routing) to decide if n2880_k720
wants BN64 or BN128 (to gate the AI/occupancy PM fix):
  n1440_k2880: BN64 15.7 vs BN128 17.7 -> BN64 1.13x
  n2880_k720 : BN64 13.8 vs BN128 15.0 -> BN64 1.09x
=> BOTH TP4 bm16 shapes want BN64 in ISOLATION (contradicts earlier "TP4 wants BN128").

BUT this CONTRADICTS the e2e sweep: pmpol (force bm16->BN64) REGRESSED TP4 (0.920 vs
cap 0.972, systematic across conc). Hard conflict:
  isolated enforce-eager kernel A/B: BN64 faster
  e2e whole-cell sweep (CUDA graphs): forcing BN64 slower
RESOLUTION: --enforce-eager changes decode from GRAPH-REPLAY to EAGER, so the isolated
kernel timing does NOT represent normal graph-captured serving. Documented a8w4
microbench-misleads trap ([[feedback_a8w4_microbench_not_representative]]). Consistent
across TPs: the "BN64 3.3x at TP8" (Gap-1 ee) also did NOT translate -- pmpol TP8 0.957
~= cap 0.956 (neutral, not +3.3%). So NONE of the enforce-eager bm16 "wins" held e2e.

CONSEQUENCES:
  * The whole bm16->BN64 thread was chasing a microbench artifact. enforce-eager
    per-config A/B is INVALID for graph-replayed decode config selection.
  * Only the e2e whole-cell sweep is trustworthy for a8w4 decode. It says: CAP (0.988,
    PM picks within {64,128}) is best; forcing BN64 is worse.
  * The NV-style AI/occupancy PM fix, even if it made PM pick BN64, would likely
    REGRESS e2e -- BN64-faster is a microbench illusion for graph-replayed decode.
DECISION: ship the CAP (_ALLOWED_BN_BY_BM={16:(64,128),32:(64,128)}, 0.988). Stop
optimizing bm16 BN via isolated A/B. Judge any future decode change by e2e sweep ONLY.

## 2026-06-26 — Graph-mode (no enforce-eager) method confirms cap; TP4 resolved

Q: how to measure per-config decode WITHOUT --enforce-eager (which inverts results)?
A: force whole-cell config (AITER_A8W4_AB A==B works in graph mode -- graph bakes it at
capture; only per-call ALTERNATION breaks under graphs), profile in NORMAL graph mode,
read kernel durations via moe_hist (graph-replayed kernels ARE on the GPU timeline; only
the a8w4cfg CPU labels are missing).

TP4 bm16, GRAPH mode, forced whole-cell:
  BN64  : moe avg 13.9us (86% 10-20us, 3% 20-30us tail)
  BN128 : moe avg 11.4us (50% 6-10us, 50% 10-20us, 0% 20-30us)
=> BN128 FASTER in real serving -- OPPOSITE of enforce-eager (which said BN64), AGREES
   with the e2e sweep (force-BN64 regressed TP4). So the cap's BN128-ish TP4 pick is
   CORRECT; enforce-eager was the artifact. TP4 regression resolved by shipping the cap.

VALID non-eager methods (trust order): (1) e2e whole-cell sweep (force config -> tput,
graph mode) = authoritative; (2) graph-mode profile + moe_hist (duration) = per-kernel
signal, BUT separate runs => routing variance (967 vs 1213 calls here), directional only;
graph mode can't do per-call matched-routing alternation. INVALID: enforce-eager per-config
A/B (changes decode exec mode -> mispredicts real serving).
RULE: judge a8w4 DECODE config only in graph mode (e2e primary, moe_hist supporting).
