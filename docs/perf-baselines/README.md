# PerfModel Baselines

Tracked snapshots of PerfModel-pick vs Triton autotune vs rocBLAS perf, used
as the ground truth for measuring future PerfModel/kernel changes.

## What's in here

One CSV + markdown pair per sweep date, named `perf-baseline-YYYY-MM-DD.{csv,md}`.

- **CSV**: machine-readable, includes ms + TFLOPS + ratios + selected config per shape
- **MD**: readable summary table, sorted by shape

## How to run

```bash
# From inside the docker container (xguo-gemm-dev):
cd /home/openai/triton_gluon_perfmodel/scripts/perf_baseline
PYTHONPATH=/home/openai/triton_perfmodel_gemm/python python3 sweep.py
```

The sweep:
1. Runs preflight checks (`preflight.py`): GPU PID conflict, clocks, temp, containers, users
2. For each shape in `shapes.py` (24 shapes), measures three backends:
   - **PerfModel pick** (top-1 from `pick_gemm_config`, kernel-only timing — config pre-picked outside loop)
   - **Triton autotune** (tutorial 03's 18-config list, after autotune warmup)
   - **rocBLAS** (via `torch.matmul`, lowers to hipBLASLT on AMD)
3. Writes CSV + markdown to this directory

Wall-clock: ~4-5 minutes for the full 24-shape sweep on MI355X.

## Shape set (24)

See `shapes.py`. Three groups:
- **Square** (6): 256³, 512³, 1024³, 2048³, 3072³, 4096³ — classic GEMM perf curve
- **Decode/skinny** (10): GPT-OSS 120B layer shapes at varied batch (M ∈ {4, 16, 32, 128})
- **Prefill** (8): GPT-OSS varied-M upper end + large square

## How to read the numbers

- All numbers are **TFLOPS** unless noted (`pm_ms`, `auto_ms`, `rb_ms` are in milliseconds)
- `pm_vs_auto > 1` means PerfModel pick beats Triton autotune at that shape
- `pm_vs_rb > 1` means PerfModel pick beats rocBLAS
- At small shapes (256³, M=4 cases), launch overhead dominates and the numbers
  are not very meaningful — they're included for completeness but ratios are
  noisy
- Selection time for PerfModel is **not** included in `pm_ms`. The config is
  pre-picked once per shape and the kernel launch is what's timed. This matches
  production use where the config is cached per shape

## Methodology notes

- Each measurement: 20 warmup iterations + 50 timed iterations, median reported
- All on MI355X (gfx950), CDNA4
- fp16 inputs, fp16 output
- Single GPU, single CTA grid (not multi-GPU)
- Preflight ensures no foreign GPU processes during the run

## Reproducibility / source of truth

For deeper debugging, the CSV has the exact `pm_cfg` (BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages)
PerfModel picked for each shape. Pair with `git rev-parse HEAD` of the
`gluon_perf_model` branch at the time of the sweep to reproduce.

## When to re-run

- After any PerfModel change (calibration fix, new correction, new dtype)
- After any kernel-family change (new SCALE_MODE, new tile path)
- Before submitting perf claims through AMD's Author's Program

## Future additions

- Add fp8 (a8w8) and bf16 sweeps when the Gluon kernel coverage is complete
- Add rocprofv3 path (hardware counters + ms) for shapes where do_bench
  underestimates throughput at small shapes
- Add Gluon kernel comparisons (PerfModel pick for Gluon vs autotune of Gluon variants)
