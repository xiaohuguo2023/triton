"""13-category sweep: imports ALL_SHAPES from shapes_13cat.py and runs the
standard sweep harness against it."""
import argparse, csv, importlib.util, os, subprocess, sys, time
from pathlib import Path

TUTORIAL = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"

import torch
import triton

import io, contextlib
spec = importlib.util.spec_from_file_location("tut03", TUTORIAL)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(tut)
    except SystemExit:
        pass

sys.path.insert(0, str(Path(__file__).parent))
from shapes_13cat import ALL_SHAPES

# Reuse the per-shape measurement function from sweep.py
import sweep as base_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/openai/triton_gluon_perfmodel/docs/perf-baselines")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rc = subprocess.run(["python3", str(Path(__file__).parent / "preflight.py")]).returncode
    if rc != 0:
        print("preflight failed — abort", file=sys.stderr)
        sys.exit(1)

    shapes = ALL_SHAPES[: args.limit] if args.limit else ALL_SHAPES
    print(f"Running {len(shapes)} shapes (13 categories)...")
    rows = []
    t0 = time.time()
    for i, (M, N, K, regime) in enumerate(shapes, 1):
        elapsed = time.time() - t0
        print(f"  [{i:2}/{len(shapes)}] {M}×{N}×{K} ({regime})  [t+{elapsed:.0f}s]", flush=True)
        row = base_sweep.measure_shape(M, N, K)
        row["regime"] = regime
        rows.append(row)

    stamp = time.strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / f"perf-baseline-13cat-{stamp}.csv"
    md_p  = out_dir / f"perf-baseline-13cat-{stamp}.md"
    base_sweep.write_csv(csv_p, rows)
    base_sweep.write_markdown(md_p, rows)
    print(f"\nWrote {csv_p}\n      {md_p}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
