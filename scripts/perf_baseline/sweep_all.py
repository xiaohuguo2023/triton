"""Full-dataset sweep: imports ALL_SHAPES from shapes_all.py (471 shapes over
all TensorAtlas varied-M datasets) and runs the SAME PerfModel / Triton-autotune
/ rocBLAS measurement as the 40-shape paper sweep (sweep.py::measure_shape).

Identical harness to sweep_13cat.py — only the shape set is the uncapped superset.

Run INSIDE the xguo-perfmodel container (fork-triton with amd_gemm_selector):
    python3 sweep_all.py                 # all 471 shapes
    python3 sweep_all.py --limit 20      # smoke test
    python3 sweep_all.py --shuffle       # randomize order (spread OOM-risky shapes)

Output:
    docs/perf-baselines/perf-baseline-all-<date>.csv
    docs/perf-baselines/perf-baseline-all-<date>.md
"""
import argparse, importlib.util, io, contextlib, subprocess, sys, time
from pathlib import Path

TUTORIAL = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"

import torch  # noqa: F401  (needed by the tutorial module + measure_shape)
import triton  # noqa: F401

# Load the tutorial (matmul autotune + matmul_kernel_amd) quietly.
spec = importlib.util.spec_from_file_location("tut03", TUTORIAL)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(tut)
    except SystemExit:
        pass

sys.path.insert(0, str(Path(__file__).parent))
from shapes_all import ALL_SHAPES
import sweep as base_sweep   # reuse measure_shape / write_csv / write_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="/home/openai/triton_gluon_perfmodel/docs/perf-baselines")
    ap.add_argument("--limit", type=int, default=0, help="first N shapes (smoke)")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize shape order (seed 42)")
    args = ap.parse_args()

    rc = subprocess.run(
        ["python3", str(Path(__file__).parent / "preflight.py")]).returncode
    if rc != 0:
        print("preflight failed — abort", file=sys.stderr); sys.exit(1)

    shapes = list(ALL_SHAPES)
    if args.shuffle:
        import random; random.seed(42); random.shuffle(shapes)
    if args.limit:
        shapes = shapes[: args.limit]

    print(f"Running {len(shapes)} shapes (full TensorAtlas superset)...")
    rows, fails = [], 0
    t0 = time.time()
    for i, (M, N, K, regime) in enumerate(shapes, 1):
        elapsed = time.time() - t0
        print(f"  [{i:3}/{len(shapes)}] {M}×{N}×{K} ({regime})  [t+{elapsed:.0f}s]",
              flush=True)
        try:
            row = base_sweep.measure_shape(M, N, K)
        except Exception as ex:            # never let one bad shape kill the sweep
            row = {"M": M, "N": N, "K": K, "fatal": str(ex)[:100]}
            fails += 1
        row["regime"] = regime
        rows.append(row)

    stamp = time.strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / f"perf-baseline-all-{stamp}.csv"
    md_p  = out_dir / f"perf-baseline-all-{stamp}.md"
    base_sweep.write_csv(csv_p, rows)
    base_sweep.write_markdown(md_p, rows)

    # quick roll-up (win = PM ratio >= 1)
    va = [r["pm_vs_auto"] for r in rows if r.get("pm_vs_auto")]
    vb = [r["pm_vs_rb"]   for r in rows if r.get("pm_vs_rb")]
    def geo(xs):
        import math
        return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")
    print(f"\nWrote {csv_p}\n      {md_p}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min   ({fails} fatal-skips)")
    print(f"PM vs autotune : geomean {geo(va):.3f}  wins {sum(x>=1 for x in va)}/{len(va)}")
    print(f"PM vs rocBLAS  : geomean {geo(vb):.3f}  wins {sum(x>=1 for x in vb)}/{len(vb)}")


if __name__ == "__main__":
    main()
