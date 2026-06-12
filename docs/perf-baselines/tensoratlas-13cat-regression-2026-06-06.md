# TensorAtlas Exhaustive Tuning — 13-cat Regression Shapes

**Date**: 2026-06-06
**Kernel tuned**: `streamk_matmul` (TensorAtlas default; not `persistent_matmul` like the prior pilot)
**Tuning mode**: `--exhaustive_tuning`, 3888 candidate configs per shape, all compiled successfully
**Hardware**: GPUs 4–7 (MI355X, gfx950), 4×3 parallel jobs
**Total tuning time**: 29:29 for 3 shapes (~10 min/shape)

## Shapes (3)

The 3 shapes from `perf-baseline-13cat-2026-06-06.md` where PerfModel
lost to Triton autotune. Common pattern: large M (16K–32K), medium N
(2112–2880), large K (2880–7168).

## Results

| shape | PM | autotune | rocBLAS | **TA exhaustive** | TA vs PM | TA vs rocBLAS |
|---|---:|---:|---:|---:|---:|---:|
| 32768×2880×2880 | 710 | 896 | 1095 | **1127** | **+59%** | **+3%** |
| 32768×2112×7168 | 740 | 877 | 1001 | **1134** | **+53%** | **+13%** |
| 16384×2112×7168 | 755 | 878 | 1071 | **1015** | +34% | -5% |

Units: TFLOPS. TA wins beat rocBLAS on shapes 1 and 2; close on shape 3.

## What TensorAtlas picked

| shape | BM | BN | BK | GM | NUM_SMS | nW | nS | mfma | CHUNK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768×2880×2880 | **256** | **256** | **64** | 16  | 256  | 8 | 2 | 16 | 32 |
| 32768×2112×7168 | **256** | **256** | **64** | 1   | 1152 | 8 | 2 | 16 | 32 |
| 16384×2112×7168 | **128** | **128** | **64** | 16  | 2176 | **4** | 2 | **32** | 32 |

## What PerfModel picked (top-5)

| shape | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 32768×2880×2880 | 128×128×**128** W8 | 128×128×**128** W4 | 128×128×64 W8 | 128×128×64 W4 | **256×256×64 W8** |
| 32768×2112×7168 | 128×128×**128** W8 | 128×128×**128** W4 | 128×128×64 W8 | 128×128×64 W4 | 256×128×64 W8 |
| 16384×2112×7168 | 128×128×**128** W8 | 128×128×**128** W4 | 128×128×64 W8 | 128×128×64 W4 | 256×128×64 W8 |

PerfModel ranks the actual winner at **#5** (shape 1) or **not in top-5**
(shapes 2 & 3, where the winner uses NUM_SMS persistent grid sizing PM
doesn't model).

## Diagnostic findings

### Gap 1 (recurring) — BK=128 ranked above BK=64 at large tiles

Same finding as the 2026-06-05 pilot. PerfModel's top-2 across all 3
shapes is BK=128, but the empirical winners are all BK=64. Fix 2 and
Fix 3 (memory-overlap formula + GPU under-fill) did not change this
ranking — they were tuned for different shape classes.

**Root cause hypothesis**: at BM=BN=256, BK=128 means one stage of the
LDS pipeline is `256×128 + 128×256 = 65 KiB` fp16 — half the
per-CTA LDS budget (128 KiB on gfx950). This caps nStages at 2 and
leaves no room for prefetch overlap. BK=64 cuts that to 32 KiB,
opening room for 3-stage pipelining and better MFMA/load overlap. The
cost model doesn't currently penalize the LDS-budget squeeze that
keeps BK=128 stuck at nStages=2.

**Action**: in `PerfModel.cpp`, add an LDS-pressure term to
`estimatePerf` that penalizes BK choices forcing nStages ≤ 2 at large
tiles. Verify on these 3 shapes before re-sweeping.

### Gap 2 — BM=256/BN=256 underranked vs BM=128/BN=128

For shapes 1 and 2 the winning tile is 256×256, but PerfModel ranks
128×128 in slots #1–#4. Likely coupled with Gap 1: PM scores 128×128
with BK=128 highly because the LDS-budget penalty is missing for
256×256.

### Gap 3 — `matrix_instr_nonkdim=32` not in candidate set

Shape 3's exhaustive winner uses `mfma_instr_nonkdim=32` (the 16×16×**32**
fp16 MFMA instruction). PerfModel's `generateCandidates` only emits
`nonkdim=16` variants. This is a *candidate-set* gap, not a *ranking*
gap — PM can't pick a config it never enumerates.

**Action**: add `nonkdim=32` to `generateCandidates` for fp16 paths on
gfx950, then re-rank.

### Gap 4 (known, unchanged) — Stream-K knobs unmodelled

TA winners use `NUM_SMS` ∈ {256, 1152, 2176} and `CHUNK_SIZE` = 32 —
explicit persistent + stream-K grid sizing. PerfModel doesn't model
these. The 2 winners that beat rocBLAS (32768×2880×2880 at 1127 vs
1095; 32768×2112×7168 at 1134 vs 1001) gain meaningful headroom from
stream-K — worth adding eventually but not in scope for the
data-parallel cost model right now.

## Caveat — resolved

Initial concern was that streamk_matmul's extra knobs (`NUM_SMS`,
`CHUNK_SIZE`) made the TFLOPS not directly comparable. Re-tuned the
same 3 shapes with `--kernel persistent --exhaustive_tuning` and got
**identical winning tiles and TFLOPS within 0.3%**:

| shape | streamk | persistent | best tile (both) |
|---|---:|---:|---|
| 32768×2880×2880 | 1127 | **1128** | BM256 BN256 BK64 nW8 mfma16 |
| 32768×2112×7168 | 1134 | **1133** | BM256 BN256 BK64 nW8 mfma16 |
| 16384×2112×7168 | 1015 | **1012** | BM128 BN128 BK64 nW4 mfma32 |

The "+3% / +13% over rocBLAS" wins are tile-choice driven, not
stream-K cleverness. The tile-choice diagnosis below is kernel-agnostic.

## Next steps

1. ~~Re-tune with `--kernel persistent`~~ — done; results identical to
   streamk within 0.3%. Diagnosis confirmed kernel-agnostic.
2. Implement Gap 1 fix (LDS-pressure / nStages-budget term) in
   `PerfModel.cpp`. Verify the ranking on these 3 shapes flips
   256×256/BK=64 above 128×128/BK=128.
3. Implement Gap 3 fix (add nonkdim=32 candidates for fp16/gfx950).
4. Re-run the 13-cat sweep, confirm these 3 regressions are gone and
   no new regressions appeared elsewhere.
5. Defer Gap 4 (stream-K modelling) — separate work item.

## How to reproduce

```bash
docker exec xguo-triton-tuning bash -c "cd /home/work/TensorAtlas && \
  HIP_VISIBLE_DEVICES=4,5,6,7 \
  python3 ./benchmarks/gemm_bench.py \
    --gemm_size_file ./datasets/perfmodel_13cat_regression.yaml \
    --ngpus 4 --jobs 12 \
    --generate-tuning-cache --exhaustive_tuning"
```

Results YAML: `/home/work/TensorAtlas-gptosstune/tuning_results@streamk_matmul_streamk@xiaohugu_ir-feature-extraction@3acd21e_06-06-2026-15:27:48.yaml`.
