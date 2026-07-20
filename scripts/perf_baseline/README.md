# PerfModel dense-GEMM baseline harness

Measures **PerfModel top-1 vs Triton autotune vs rocBLAS (hipBLASLt)** across real
LLM GEMM shapes on gfx950, to validate and debug the analytical config selector
(`triton.backends.amd.amd_gemm_selector.pick_gemm_config`). All runs are fp16
dense; every metric is a **PM/backend ratio** (self-normalized to the backend
measured in the same run, so ratios compare across runs/machines).

Run everything **inside the perf-model container** (fork Triton with
`amd.perf_model` built into libtriton). A `/home/openai` and `/home/work` symlink
to the host checkout is assumed (see the sweep scripts' hardcoded paths).

## GPU cleanliness gate (read this first)

Shared-chassis power/thermal contamination silently corrupts perf sweeps. Every
runner is gated by `preflight.py`, whose **GPU-utilization check** (`rocm-smi
--showuse`, abort if any GPU > 10%) is the reliable guard — note that
`--showpids` does NOT see a neighbor running in another container/namespace, but
`--showuse` does. Always let the gate pass (or wait) before trusting numbers.

```bash
python3 preflight.py            # exit 0 = clean, 1 = contended
python3 preflight.py --force    # report but don't abort
```

## Shape sets

- `shapes_13cat.py`  — 40-shape 13-category stratified sample (the paper set).
  Built by `build_13cat_shapes.py` (caps per category).
- `shapes_all.py`    — 471-shape uncapped superset over the six
  `TensorAtlas/datasets/*_shapes_varied_m.yaml` families (deepseek_r1, gpt_oss,
  llama3_fp8, llama3_mlp, llama4_maverick, qwen3_235b). Built by
  `build_all_shapes.py` (host, CPU-only; vendors the TensorAtlas classifier).

## Sweeps

| script | scope | notes |
|---|---|---|
| `sweep.py`          | `shapes.ALL_SHAPES`     | per-shape `measure_shape` (3 backends) |
| `sweep_13cat.py`    | 40-shape paper set      | |
| `sweep_all.py`      | 471-shape superset      | `--limit N`, `--shuffle`; ~11 min |
| `sweep_category.py` | ONE category            | **preflight-gated**; short (~1–3 min) |
| `sweep_ab_categories.py` | all categories, A/B | one process; **re-checks GPU util before each category**, saves partial-but-clean; compares current binary (with-change) to the committed tie-break baseline (without-change) |

```bash
python3 sweep_category.py --list             # categories + shape counts
python3 sweep_category.py MEDIUM             # gated single category
python3 sweep_category.py LARGE_K --gpu 3
python3 sweep_ab_categories.py --gpu 0        # full with/without per-category table
```

Outputs land in `../../docs/perf-baselines/`.

## Debug / physics tools

- `pm_debug_verylarge.py`, `pm_debug_medium.py` — per-shape decomposition of a
  loss: PM top-1 vs PM oracle (best in PM's own ranked set) vs autotune, and the
  PM-rank of the oracle. Tells RANKING-gap from CANDIDATE-gap.
- `pm_estimate_dump.py`, `pm_estimate_dump2.py` — dump the full `PerfEstimate`
  breakdown for a tile family on one shape (find the mis-ordering term).
- `pm_physics_study.py`, `pm_physics_curve.py` — clean-wave (M=N=16384) study of
  realized MFMA efficiency vs tile size; derives
  `eff ≈ 2.07·(BM/(BM+108))·(BN/(BN+108))` (operand-reuse + accumulator
  latency-hiding, saturating).
- `pm_rerank_validate.py`, `pm_rerank_wavefrac.py` — validate a candidate
  cost-model change WITHOUT rebuilding: re-rank in Python (old vs new tie-break /
  formula), measure both top-1 picks + autotune per shape.

## Method notes

- **Root-cause a wrong pick, don't tie-break blindly.** `pm_debug_*` first
  establishes whether the winner is in PM's candidate set (ranking gap) or not
  (candidate gap); `pm_estimate_dump*` finds which cost term mis-orders it.
- **Validate in Python before a libtriton rebuild** (`pm_rerank_*`): a rebuild is
  a few minutes; a wrong direction wastes it.
- **Prefer bounded, physically-grounded formula changes over fitted constants**
  and always gate a candidate change on a clean full/per-category sweep with a
  no-cross-category-regression rule.
