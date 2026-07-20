"""Per-category A/B baseline: measure the CURRENT binary (with-change) per
category, compare to the committed without-change baseline (tie-break only).

Runs all categories in ONE process (torch imported once). Re-checks GPU
utilization before EACH category via rocm-smi --showuse; if a neighbor is
running (>10% on any GPU) it saves what's done and stops cleanly, so partial
results are always clean. The without-change column is the committed
perf-baseline (PM/auto ratios, comparable across runs since both normalize to
autotune measured in-run).

Usage:  python3 sweep_ab_categories.py [--gpu N] [--force]
Output: docs/perf-baselines/ab_by_category-<date>.csv  + printed table
"""
import argparse, csv, math, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from shapes_all import ALL_SHAPES  # noqa: E402

# committed without-change (tie-break) per-category baseline (from 153ae21c3)
WITHOUT = {
    "LARGE_NK": (261, 1.384, 79), "MEDIUM": (53, 1.076, 75),
    "VERY_LARGE": (49, 1.161, 98), "LARGE_K": (28, 1.897, 100),
    "LARGE": (18, 1.135, 83), "LARGE_MK": (18, 1.421, 94),
    "LARGE_N": (14, 1.059, 71), "LARGE_MN": (13, 1.106, 92),
    "SMALL": (6, 1.507, 100), "LARGE_K_SKINNY": (3, 2.025, 100),
    "LARGE_M_SKINNY": (3, 1.564, 100), "LARGE_N_SKINNY": (3, 1.504, 100),
    "LARGE_M": (2, 1.344, 100),
}
# run smallest categories first -> fast partial results if a neighbor appears
ORDER = sorted(WITHOUT, key=lambda c: WITHOUT[c][0])


def gpu_busy():
    try:
        out = subprocess.run(["rocm-smi", "--showuse"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    busy = []
    for line in out.splitlines():
        m = re.search(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", line)
        if m and int(m.group(2)) > 10:
            busy.append(f"GPU{m.group(1)}={m.group(2)}%")
    return busy


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")

    b = gpu_busy()
    if b and not args.force:
        print(f"ABORT: machine not clean at start: {b}", file=sys.stderr); sys.exit(1)

    import sweep as base_sweep  # heavy import after the gate

    groups = {}
    for (M, N, K, r) in ALL_SHAPES:
        groups.setdefault(r.split("-")[0], []).append((M, N, K, r))

    results = {}
    for cat in ORDER:
        b = gpu_busy()
        if b and not args.force:
            print(f"\n!! neighbor detected before {cat} ({b}) -- stopping; "
                  f"{len(results)}/{len(ORDER)} categories done cleanly.")
            break
        shapes = groups.get(cat, [])
        t0 = time.time()
        va, vb = [], []
        for (M, N, K, r) in shapes:
            try:
                row = base_sweep.measure_shape(M, N, K)
                if row.get("pm_vs_auto"): va.append(row["pm_vs_auto"])
                if row.get("pm_vs_rb"):   vb.append(row["pm_vs_rb"])
            except Exception:
                pass
        results[cat] = (len(va), geo(va), 100*sum(x>=1 for x in va)/len(va) if va else 0,
                        geo(vb))
        print(f"  {cat:<16} n={len(va):>3}  with={geo(va):.3f} "
              f"({100*sum(x>=1 for x in va)/len(va) if va else 0:.0f}% win)  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # table
    print(f"\n{'category':<16}{'n':>4}{'without':>9}{'with':>9}{'delta':>8}{'win%_w/o':>9}{'win%_with':>10}")
    for cat in ORDER:
        if cat not in results: continue
        n, gw, winw, _ = results[cat]
        n0, g0, win0 = WITHOUT[cat]
        print(f"{cat:<16}{n:>4}{g0:>9.3f}{gw:>9.3f}{gw-g0:>+8.3f}{win0:>8}%{winw:>9.0f}%")

    stamp = time.strftime("%Y-%m-%d")
    outp = HERE.parent.parent / "docs" / "perf-baselines" / f"ab_by_category-{stamp}.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "without_geo", "with_geo", "delta", "without_win%", "with_win%"])
        for cat in ORDER:
            if cat not in results: continue
            n, gw, winw, _ = results[cat]
            n0, g0, win0 = WITHOUT[cat]
            w.writerow([cat, n, f"{g0:.4f}", f"{gw:.4f}", f"{gw-g0:+.4f}", win0, f"{winw:.0f}"])
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
