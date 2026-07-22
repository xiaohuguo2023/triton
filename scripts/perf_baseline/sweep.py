"""PerfModel baseline sweep — model vs autotune vs rocBLAS.

For each shape in shapes.ALL_SHAPES, measure:
  (1) PerfModel-picked config       — single config from pick_gemm_config
  (2) Triton autotune (tutorial 18) — best of get_hip_autotune_config()
  (3) rocBLAS reference             — torch.matmul (hipBLASLT/rocBLAS on AMD)

Two measurement paths per cell:
  - do_bench   (smoke / quick)
  - rocprofv3  (truth — separate run, opt-in via --rocprof)

Output:
  docs/perf-baselines/perf-baseline-<date>.csv
  docs/perf-baselines/perf-baseline-<date>.md
"""
import argparse, csv, importlib.util, os, subprocess, sys, time
from pathlib import Path

# Reuse the tutorial's matmul_model + matmul_kernel_amd + autotune list directly.
TUTORIAL = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"

import torch
import triton

# Import the tutorial as a module so we get matmul_model + matmul (autotune) + matmul_kernel_amd.
# The tutorial runs benchmark.run(...) at module top-level which prints lots of output and
# burns ~30s of GPU time. Suppress it by intercepting __main__ execution if we're imported.
import io, contextlib
spec = importlib.util.spec_from_file_location("tut03", TUTORIAL)
tut = importlib.util.module_from_spec(spec)
# Patch out the tutorial's auto-run benchmark by redirecting stdout during import.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        spec.loader.exec_module(tut)
    except BaseException:
        # The tutorial's top-level benchmark.run() may fail on a lean serving
        # container (e.g. matplotlib not installed). We only need its matmul /
        # matmul_kernel_amd / matmul_model definitions, which are defined before
        # that call, so swallow the failure and validate the symbols below.
        pass
assert hasattr(tut, "matmul") and hasattr(tut, "matmul_kernel_amd"), \
    "tutorial module missing matmul/matmul_kernel_amd after import"

sys.path.insert(0, str(Path(__file__).parent))
from shapes import ALL_SHAPES


import math as _math
import statistics as _stats
from torch.profiler import profile, ProfilerActivity


def filter_outliers_iqr(values, min_samples=5, iqr_multiplier=1.5):
    """Drop UPPER outliers via the IQR (Tukey fence) method — same as TensorAtlas
    (utils/timing_stats.py). Only slow spikes (thermal throttle / scheduling
    jitter) are removed; fast times are never outliers for timing. Falls back to
    the original list if too few samples survive. Returns (filtered, n_removed)."""
    vals = list(values)
    if not vals:
        return [], 0
    if len(vals) < 5:
        return vals, 0
    q = _stats.quantiles(vals, n=4)
    q1, q3 = q[0], q[2]
    upper = q3 + iqr_multiplier * (q3 - q1)
    filt = [v for v in vals if v <= upper]
    if len(filt) >= min_samples:
        return filt, len(vals) - len(filt)
    return vals, 0


def _burst_device_ms(fn, iters):
    """Mean GPU device-time per call (ms) over a short profiled burst."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for ev in prof.key_averages():
        total_us += getattr(ev, "self_device_time_total", 0.0) or 0.0
    return (total_us / iters) / 1000.0 if iters else float("inf")


def interleaved_bench_profiler(fns, rounds=10, iters=6, warmup=10):
    """Interleaved, IQR-filtered device-time benchmark (TensorAtlas methodology,
    on profiler timing). Each ROUND profiles a short burst of every fn in turn —
    fn0, fn1, fn2, fn0, fn1, fn2, ... — so all configs experience the same
    thermal/clock state (fair A/B), unlike measuring one fully then the next.
    Per-round device-times are IQR-filtered (drop throttle spikes) and averaged.
    Device-time (not wall-clock) excludes CPU launch overhead — essential for the
    microsecond SMALL/*_SKINNY kernels where launch overhead dominated the ratio
    and produced phantom regressions. Returns one mean ms per fn (None if failed).
    """
    n = len(fns)
    for fn in fns:                       # warm all (compile, cache)
        for _ in range(warmup):
            try:
                fn()
            except Exception:
                pass
    torch.cuda.synchronize()
    samples = [[] for _ in range(n)]
    failed = set()
    for _ in range(rounds):
        for j, fn in enumerate(fns):
            if j in failed:
                continue
            try:
                samples[j].append(_burst_device_ms(fn, iters))
            except Exception:
                failed.add(j)
    out = []
    for j in range(n):
        finite = [x for x in samples[j] if _math.isfinite(x) and x > 0]
        if not finite:
            out.append(None)
            continue
        filt, _ = filter_outliers_iqr(finite)
        out.append(_stats.mean(filt) if filt else None)
    return out


def bench_ms(fn, warmup=20, repeat=50):
    """GPU kernel time per call (ms), via torch.profiler device-side timing.

    Replaces HIP-event wall-clock timing (2026-07). Rationale: for microsecond
    kernels (SMALL, *_SKINNY, tiny-M shapes — 3-5 us) the per-launch wall-clock
    is dominated by ~2-4 us of CPU launch overhead, which (a) INFLATES the
    PM/autotune ratio (wall-clock ≈ kernel + launch, so the launch term swamps a
    tiny kernel) and (b) is noisy enough at median-of-50 to flip a shape across
    1.0 — the source of the phantom SMALL/*_SKINNY "regressions". The profiler
    reads the device-side kernel duration directly (roctracer), excluding launch
    overhead and CPU jitter, so it is both more accurate and far more stable for
    short kernels; for large (ms-scale) kernels it agrees with event timing.

    Sums self device-time over the profiled window and divides by `repeat` — this
    captures all GPU work per call (incl. any split-K reduction kernel)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for ev in prof.key_averages():
        total_us += getattr(ev, "self_device_time_total", 0.0) or 0.0
    return (total_us / repeat) / 1000.0  # us -> ms/call


def to_tflops(M, N, K, ms):
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def measure_shape(M, N, K, dtype=torch.float16):
    """Run all 3 backends; return dict of ms + TFLOPS + selected config.

    For PerfModel: pre-pick the config once, bench only the kernel launch.
    This isolates kernel perf from the (small but non-zero) selector overhead,
    matching how it would be used in production (cache the config per shape).
    """
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    row = {"M": M, "N": N, "K": K}

    # Build the three callables, then bench them INTERLEAVED so PM / autotune /
    # rocBLAS all see the same thermal state (fair A/B ratio) with IQR-filtered
    # device-time. (2026-07: replaced 3 separate sequential bench_ms calls, which
    # let clock drift bias the ratio and — via wall-clock launch overhead — mis-
    # ranked microsecond kernels.)
    call_pm = call_auto = call_rb = None

    # (1) PerfModel pick — pre-pick config, bench only the kernel launch
    try:
        from triton.backends.amd.amd_gemm_selector import (
            pick_gemm_config, config_to_kernel_kwargs, current_amd_arch,
        )
        arch  = current_amd_arch()
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        cfg   = pick_gemm_config(M, N, K, dtype_str, arch, top_k=1)[0]
        kw    = config_to_kernel_kwargs(cfg)
        c     = torch.empty((M, N), device="cuda", dtype=dtype)
        grid  = (triton.cdiv(M, cfg.block_m) * triton.cdiv(N, cfg.block_n),)

        def call_pm():
            tut.matmul_kernel_amd[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                ACTIVATION="", **kw)
        row["pm_cfg"] = f"BM={cfg.block_m},BN={cfg.block_n},BK={cfg.block_k},W={cfg.num_warps},S={cfg.num_stages}"
    except Exception as ex:
        row["pm_err"] = str(ex)[:80]

    call_auto = lambda: tut.matmul(a, b)
    call_rb   = lambda: torch.matmul(a, b)

    fns   = [call_pm, call_auto, call_rb]
    names = ["pm", "auto", "rb"]
    valid = [(nm, fn) for nm, fn in zip(names, fns) if fn is not None]
    try:
        times = interleaved_bench_profiler([fn for _, fn in valid])
    except Exception as ex:
        times = [None] * len(valid)
        row["bench_err"] = str(ex)[:80]
    for (nm, _), ms in zip(valid, times):
        if ms is None:
            row[f"{nm}_ms"] = None
            continue
        row[f"{nm}_ms"] = ms
        row[f"{nm}_tflops"] = to_tflops(M, N, K, ms)

    # Ratios
    if row.get("pm_tflops") and row.get("auto_tflops"):
        row["pm_vs_auto"] = row["pm_tflops"] / row["auto_tflops"]
    if row.get("pm_tflops") and row.get("rb_tflops"):
        row["pm_vs_rb"] = row["pm_tflops"] / row["rb_tflops"]
    return row


def write_csv(path, rows):
    cols = ["M", "N", "K", "regime",
            "pm_tflops", "auto_tflops", "rb_tflops",
            "pm_vs_auto", "pm_vs_rb",
            "pm_ms", "auto_ms", "rb_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# PerfModel Baseline Sweep\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)\n\n")
        f.write("| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            shape = f"{r['M']}×{r['N']}×{r['K']}"
            pm    = f"{r['pm_tflops']:.0f}"   if r.get("pm_tflops")   else "FAIL"
            au    = f"{r['auto_tflops']:.0f}" if r.get("auto_tflops") else "FAIL"
            rb    = f"{r['rb_tflops']:.0f}"   if r.get("rb_tflops")   else "FAIL"
            r_a   = f"{r['pm_vs_auto']:.2f}×" if r.get("pm_vs_auto") else "—"
            r_b   = f"{r['pm_vs_rb']:.2f}×"   if r.get("pm_vs_rb")   else "—"
            f.write(f"| {shape} | {r['regime']} | {pm} | {au} | {rb} | {r_a} | {r_b} |\n")
        f.write("\nUnits: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/openai/triton_gluon_perfmodel/docs/perf-baselines")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N shapes (debug)")
    args = ap.parse_args()

    # Preflight
    rc = subprocess.run(
        ["python3", str(Path(__file__).parent / "preflight.py")]
    ).returncode
    if rc != 0:
        print("preflight failed — abort", file=sys.stderr)
        sys.exit(1)

    shapes = ALL_SHAPES[: args.limit] if args.limit else ALL_SHAPES
    print(f"Running {len(shapes)} shapes...")
    rows = []
    for i, (M, N, K, regime) in enumerate(shapes, 1):
        print(f"  [{i:2}/{len(shapes)}] {M}×{N}×{K} ({regime})", flush=True)
        row = measure_shape(M, N, K)
        row["regime"] = regime
        rows.append(row)

    stamp = time.strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_p  = out_dir / f"perf-baseline-{stamp}.csv"
    md_p   = out_dir / f"perf-baseline-{stamp}.md"
    write_csv(csv_p, rows); write_markdown(md_p, rows)
    print(f"\nWrote {csv_p}\n      {md_p}")


if __name__ == "__main__":
    main()
