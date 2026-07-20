# Wide-dataset PerfModel sweep (471 shapes)

Extends the 40-shape paper sweep to **all** TensorAtlas varied-M shapes, using the
**identical** measurement harness (`sweep.py::measure_shape`): per shape it compares
**PerfModel top-1 · Triton autotune · rocBLAS (torch.matmul → hipBLASLt)** in TFLOPS.

## Files
- `build_all_shapes.py` — host, CPU-only. Collects every unique `(M,N,K)` from the six
  `*_shapes_varied_m.yaml` datasets (deepseek_r1, gpt_oss_120b, llama3_fp8, llama3_mlp,
  llama4_maverick, qwen3_235b), applies the same 4 GiB fp16 OOM cap, classifies into the
  13 categories, and writes `shapes_all.py`. **Already run → 471 shapes.**
- `shapes_all.py` — generated `ALL_SHAPES` (471 shapes, uncapped superset of `shapes_13cat.py`).
- `sweep_all.py` — the sweep driver. Same harness as `sweep_13cat.py`, over `shapes_all`.

## Shape distribution (471 shapes, 13 categories)
LARGE_NK 261 · MEDIUM 53 · VERY_LARGE 49 · LARGE_K 28 · LARGE_MK 18 · LARGE 18 ·
LARGE_N 14 · LARGE_MN 13 · SMALL 6 · LARGE_M_SKINNY 3 · LARGE_N_SKINNY 3 ·
LARGE_K_SKINNY 3 · LARGE_M 2.
(LARGE_NK dominates — the natural dataset distribution. Report per-category geomean so
the imbalance is visible; the paper's headline geomean can be reported both raw and
category-balanced.)

## Run (INSIDE the xguo-perfmodel container — fork-triton with amd_gemm_selector)
```bash
# smoke first (20 shapes, ~a couple minutes)
python3 sweep_all.py --limit 20
# full sweep
python3 sweep_all.py --shuffle
```
Outputs `docs/perf-baselines/perf-baseline-all-<date>.{csv,md}` + a geomean/win-rate roll-up.

## Cost & scheduling
- `measure_shape` is do_bench (warmup 20 / repeat 50) for 3 backends; the Triton-autotune
  arm compiles its 18 configs once per unseen shape. Estimate **~0.5–2 min/shape →
  ~4–15 GPU-hours for 471 shapes** (dominated by first-time autotune compiles).
- **The machine is currently in use — do NOT launch yet.** Run the 20-shape smoke to get a
  real per-shape time, then decide the full budget. `--shuffle` spreads the large
  (OOM-risky) shapes so an early failure doesn't waste the run.

## Note on R15 / tuning latency
This harness measures kernel **runtime quality** (PM vs autotune-best vs rocBLAS TFLOPS),
**not** tuning wall-clock. It strengthens the ranking-quality claim (40 → 471 shapes) but
does not by itself produce the "significant tuning latency" number the reviewer asked for.
For that, add a small `time_tuning.py` that times, for a handful of representative shapes:
the Triton autotune first-call (compile+bench of the 18–220 candidates) vs PerfModel's
single compile. That is a separate, quick deliverable — not part of this sweep.
