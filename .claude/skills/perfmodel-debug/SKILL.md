---
name: perfmodel-debug
description: Debug and validate the AMD GEMM PerfModel using reusable aiter, TensorAtlas, GROUP_SIZE_M, MFMA-efficiency, BK, num_stages, and LDS probes. Use when investigating PerfModel ranking regressions, gfx950 calibration changes, TensorAtlas misses, aiter gpt-oss mis-ranks, LDS validity mismatches, or when the user mentions perf_model_debug.py, aiter baseline, gm sweep, BK32, LDS probe, or TensorAtlas residuals.
---

# PerfModel Debug

## Quick Start

Use `.claude/skills/perfmodel-debug/perf_model_debug.py` instead of recreating scratch scripts. Most GPU-running commands assume the repo is installed in the active environment/container and that `aiter` is available.

Before GPU-running subcommands (`aiter-baseline`, `efficiency`, `bk32-*`, `lds-probe`, `tensoratlas-misses`), check the GPU is idle:

```bash
rocm-smi --showuse
rocm-smi --showpids
```

Run commands from the repo root:

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py <subcommand> [args]
```

For the current MI355X container workflow:

```bash
docker exec xguo-perfmodel python3 /home/xiaohugu/openai/triton_gluon_perfmodel/.claude/skills/perfmodel-debug/perf_model_debug.py <subcommand> [args]
```

## Validation Order

When changing `PerfModel.cpp`, validate in this order:

1. `gm-sweep`: confirm `GROUP_SIZE_M` does not swing predicted throughput by multiples.
2. `aiter-baseline`: check the gpt-oss aiter 24-cell table-vs-PM geomean.
3. `tensoratlas-misses`: bucket residual TensorAtlas losses and inspect whether misses are `tile`, `bk`, `nw`, `ns`, or `mag_only`.
4. Use a focused probe only if the buckets point to it:
   - `efficiency` for compute-roof / MFMA efficiency.
   - `bk32-diag` and `bk32-ab` for BK32/BK64 and `num_stages` interactions.
   - `lds-probe` for LDS validity and async padding mismatches.

Do not tune from aggregate metrics alone. Always inspect per-shape changed picks before adding a new model term.

## Subcommands

### aiter-baseline

Runs the aiter gpt-oss 24-cell benchmark: four projections (`attn_QKV`, `attn_O`, `router`, `lm_head`) times six M values. It compares aiter table-picked real runtime to PerfModel-picked real runtime.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py aiter-baseline
```

Use this to track the main aiter geomean and flag cells below the ratio threshold.

### gm-sweep

Prints predicted TFLOPS for `gm in {1,2,4,8}` for selected tiles.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py gm-sweep --M 1024 --N 201088 --K 2880
```

Use this after L2/WGM changes. The expected behavior is a modest, mostly flat change, not multi-x swings.

### tensoratlas-misses

Runs the TensorAtlas residual-miss bucketer using tuned YAML datasets.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py tensoratlas-misses \
  --datasets-dir /home/xiaohugu/mi450/datasets \
  --datasets gpt_oss_120b qwen3_235b_a22b llama3_mlp deepseek_r1 llama4_maverick
```

Use the bucket summary to decide the next probe. If misses are memory/BK/stage related, do not change compute efficiency.

### efficiency

Benchmarks high-arithmetic-intensity square shapes to isolate realized MFMA efficiency.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py efficiency
```

Use this before changing compute roofline constants. The original gfx950 aiter finding was roughly `0.40` of theoretical peak for fat tiles and about `0.20` for min tile dim `64`.

### bk32-diag

Dumps top-ranked aiter configs and BK siblings for known BK32-sensitive cases.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py bk32-diag
```

Use this to distinguish a true BK cost from a wrong `num_stages` choice or invalid candidate filtering.

### bk32-ab

Runs exact real-runtime A/B comparisons for BK32/BK64 and `ns2/ns3`.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py bk32-ab
```

Use this before penalizing BK32. BK32 with `ns3` can be legitimately fast.

### lds-probe

Compares model LDS bytes to compiled `aiter` kernel `metadata.shared`.

```bash
python3 .claude/skills/perfmodel-debug/perf_model_debug.py lds-probe
```

Use this for validity mismatches such as `lds_exceeded=true` while the real kernel runs. Prefer this over obsolete scratch `lds_probe.py`.

## Notes

- The harness is diagnostic and intentionally small-scope. Full benchmark sweeps still live under `scripts/perf_baseline/`.
- Current defaults are `gfx950`/MI355X oriented. For `gfx1250`, gate gfx950-calibrated model terms first and add separate characterization.
- Rebuild or reinstall Triton after native changes before relying on Python bindings.
