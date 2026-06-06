# TensorAtlas Pilot — Diagnostic for PerfModel Regression Shapes

**Date**: 2026-06-05
**Kernel tuned**: `persistent_matmul` (TensorAtlas's data-parallel persistent kernel)
**Tuning mode**: pruned heuristic space (default), 144 candidate configs per shape
**Hardware**: GPU 0 (MI355X, gfx950), pinned via `HIP_VISIBLE_DEVICES=0`
**Total tuning time**: 4:45 for 4 shapes (~70 sec/shape)

## Shapes (4)

These are the 4 shapes where PerfModel-picked config underperformed
Triton autotune in the 2026-06-05 baseline sweep — explicitly chosen to
diagnose PerfModel calibration misses at large prefill.

## Results

| shape | Baseline PM | Baseline autotune | **TensorAtlas tuned** | rocBLAS | TA tuned vs PM | TA tuned vs rocBLAS |
|---|---|---|---|---|---|---|
| 8192×2880×4096 | 481 | 850 | **971** | 1229 | **+102%** | 79% |
| 3072×3072×3072 | 442 | 598 | **698** | 954 | +58% | 73% |
| 4096×5120×2880 | 653 | 803 | **838** | 1049 | +28% | 80% |
| 4096×2880×4096 | 575 | 739 | **892** | 1159 | +55% | 77% |

Units: TFLOPS. Baseline numbers from `perf-baseline-2026-06-05.md`.

## What TensorAtlas picked

All 4 shapes picked the **same general config region**:

```
BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=2, matrix_instr_nonkdim=16
```

Per-shape differences (just `GROUP_SIZE_M`, `NUM_SMS`, `CHUNK_SIZE`, cache modifiers):

| shape | GM | NUM_SMS | CHUNK_SIZE | CACHE_MOD |
|---|---|---|---|---|
| 8192×2880×4096 | 4 | 384 | 32 | .ca / .ca |
| 3072×3072×3072 | 3 | 144 | 64 | .ca / .ca |
| 4096×5120×2880 | 4 | 320 | 64 | .ca / .ca |
| 4096×2880×4096 | 6 | 192 | 64 | .ca / .ca |

## Diagnostic findings (PerfModel calibration gaps)

The consistency of the TensorAtlas pick across all 4 shapes points at
specific PerfModel calibration misses:

### Gap 1 — `num_warps=8` is undervalued or absent from PerfModel's pick set

All 4 winners use `num_warps=8`. In our baseline, PerfModel was picking
configs with `num_warps=4` (per the `pm_cfg` column in the CSV). On
MI355X with large tiles, 8 warps:
- Fills 2× the SIMDs per CTA → better issue rate
- Doubles the warp slot count for hiding LDS-load latency
- Halves the per-warp register tile (still fits VGPR budget at BM=256/BN=256)

Action item: check `generateCandidates` for kernelType=Standard — does
it emit num_warps=8 candidates at all? If yes, why does `rankConfigs`
score them lower?

### Gap 2 — `BK=64` with `BM=256, BN=256` is missed

All 4 winners use `BK=64`. PerfModel's picks for these shapes likely
favored larger BK (128/256) under the assumption that more K per
iteration amortizes loop overhead better. Reality: at BM=256, BN=256
the LDS budget for one BK=128 buffer is significant; BK=64 lets us
keep 2× more pipeline stages in flight and reduces per-CTA register
pressure.

Action item: revisit the BK preference logic. The L2-XCD simulation
recently added (commit `2b5568bdf9`) should make BK matter to the L2
hit-rate score — verify it's correctly distinguishing BK=64 from
BK=128 at these tile sizes.

### Gap 3 — `GROUP_SIZE_M` may not be optimally selected

TensorAtlas picked GM ∈ {3, 4, 6} per shape. PerfModel's
`selectGroupSizeM` was tuned earlier; it'd be useful to compare its
picks for these shapes against the empirical winners.

### Gap 4 — Stream-K (`NUM_SMS`, `CHUNK_SIZE`) is unmodelled

The TensorAtlas winners use `NUM_SMS` in {144, 192, 320, 384} —
explicit persistent-kernel grid sizing. PerfModel currently doesn't
model `NUM_SMS` as a candidate dimension. This is a known gap but
worth flagging since the 79-80% rocBLAS ratio TensorAtlas achieves
shows there's still headroom against rocBLAS even with optimal
persistent-kernel tuning.

## Next steps

1. Re-bench TensorAtlas-tuned configs through our baseline sweep harness
   to get an apples-to-apples PerfModel vs TensorAtlas comparison on the
   *same* timing methodology (interleaved benches, identical warmup).
2. Investigate Gap 1 (`num_warps=8`) and Gap 2 (`BK=64` at large tiles)
   in PerfModel's `generateCandidates` / `rankConfigs`.
3. Once PerfModel is updated, re-run the baseline sweep and confirm the
   4 regression shapes are no longer regressions.
4. If the pilot is useful, extend to the full 24-shape baseline.

## How to reproduce

```bash
docker exec xguo-triton-tuning bash -c "cd /home/work/TensorAtlas && \
  HIP_VISIBLE_DEVICES=0 \
  python3 benchmarks/gemm_bench.py \
    --kernel persistent \
    --gemm_size_file datasets/perfmodel_regression_pilot.yaml \
    --o datasets/perfmodel_regression_pilot_tuned.yaml"
```

Output YAML: `/home/work/TensorAtlas/datasets/perfmodel_regression_pilot_tuned.yaml`.
