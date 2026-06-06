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
with contextlib.redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(tut)
    except SystemExit:
        pass

sys.path.insert(0, str(Path(__file__).parent))
from shapes import ALL_SHAPES


def bench_ms(fn, warmup=20, repeat=50):
    """Median ms over `repeat` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


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

    # (1) PerfModel pick — pre-pick config, bench only the kernel
    try:
        from triton.backends.amd.amd_gemm_selector import (
            pick_gemm_config, config_to_kernel_kwargs, current_amd_arch,
        )
        arch  = current_amd_arch()
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        cfgs  = pick_gemm_config(M, N, K, dtype_str, arch, top_k=1)
        cfg   = cfgs[0]
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

        ms = bench_ms(call_pm)
        row["pm_ms"] = ms
        row["pm_tflops"] = to_tflops(M, N, K, ms)
        row["pm_cfg"] = f"BM={cfg.block_m},BN={cfg.block_n},BK={cfg.block_k},W={cfg.num_warps},S={cfg.num_stages}"
    except Exception as ex:
        row["pm_ms"] = None
        row["pm_err"] = str(ex)[:80]

    # (2) Triton autotune (tutorial 18 configs)
    try:
        ms = bench_ms(lambda: tut.matmul(a, b))
        row["auto_ms"] = ms
        row["auto_tflops"] = to_tflops(M, N, K, ms)
    except Exception as ex:
        row["auto_ms"] = None
        row["auto_err"] = str(ex)[:80]

    # (3) rocBLAS reference (torch.matmul on AMD lowers to hipBLASLT/rocBLAS)
    try:
        ms = bench_ms(lambda: torch.matmul(a, b))
        row["rb_ms"] = ms
        row["rb_tflops"] = to_tflops(M, N, K, ms)
    except Exception as ex:
        row["rb_ms"] = None
        row["rb_err"] = str(ex)[:80]

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
