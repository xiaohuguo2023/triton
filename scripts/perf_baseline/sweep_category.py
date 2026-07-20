"""Short per-category sweep with a GPU-cleanliness gate.

Runs preflight FIRST (before importing torch / touching the GPU): aborts unless
the machine is clean (all GPUs idle per rocm-smi --showuse, no foreign PIDs,
temp OK). Then benches ONLY the requested category's shapes -- PM top-1 vs
Triton autotune vs rocBLAS -- and reports per-shape PM/auto, geomean, win rate,
and losers. Per-shape PM/auto is measured back-to-back so it is robust even if a
neighbor starts mid-run (but the preflight gate is the primary guard).

Usage:
    python3 sweep_category.py MEDIUM            # gated; aborts if machine busy
    python3 sweep_category.py LARGE_K --force   # run anyway (report status)
    python3 sweep_category.py VERY_LARGE --gpu 3
    python3 sweep_category.py --list            # show categories + shape counts

Categories: SMALL MEDIUM LARGE VERY_LARGE LARGE_M LARGE_N LARGE_K LARGE_MN
            LARGE_MK LARGE_NK LARGE_M_SKINNY LARGE_N_SKINNY LARGE_K_SKINNY
"""
import argparse, math, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from shapes_all import ALL_SHAPES  # noqa: E402


def categories():
    from collections import Counter
    c = Counter(r.split("-")[0] for (_, _, _, r) in ALL_SHAPES)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("category", nargs="?", help="e.g. MEDIUM, LARGE_K, VERY_LARGE")
    ap.add_argument("--force", action="store_true", help="run even if machine busy")
    ap.add_argument("--gpu", type=int, default=None, help="pin HIP_VISIBLE_DEVICES")
    ap.add_argument("--list", action="store_true", help="list categories and exit")
    args = ap.parse_args()

    if args.list or not args.category:
        print("categories (name: shape count):")
        for k, v in categories().items():
            print(f"  {k:<16} {v}")
        sys.exit(0)

    cat = args.category.upper()
    shapes = [(M, N, K, r) for (M, N, K, r) in ALL_SHAPES if r.split("-")[0] == cat]
    if not shapes:
        print(f"no shapes for category '{cat}'. Use --list.", file=sys.stderr)
        sys.exit(2)

    # ---- GPU cleanliness gate (before any heavy import / GPU touch) ----
    pf = [sys.executable, str(HERE / "preflight.py")]
    if args.force:
        pf.append("--force")
    rc = subprocess.run(pf).returncode
    if rc != 0:
        print("preflight failed -- machine not clean. Aborting "
              "(use --force to override).", file=sys.stderr)
        sys.exit(1)

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # heavy imports only after the gate passes
    import sweep as base_sweep  # noqa: E402

    print(f"\n=== {cat}: {len(shapes)} shapes"
          f"{' on GPU ' + str(args.gpu) if args.gpu is not None else ''} ===")
    rows = []
    for i, (M, N, K, r) in enumerate(shapes, 1):
        try:
            row = base_sweep.measure_shape(M, N, K)
        except Exception as ex:
            row = {"M": M, "N": N, "K": K, "fatal": str(ex)[:80]}
        row["regime"] = r
        rows.append(row)
        va = row.get("pm_vs_auto")
        print(f"  [{i:3}/{len(shapes)}] {M}x{N}x{K:<7} "
              + (f"PM/auto={va:.3f}  PM/rb={row.get('pm_vs_rb', 0):.3f}"
                 if va else f"FAIL {row.get('fatal','')}"), flush=True)

    va = [r["pm_vs_auto"] for r in rows if r.get("pm_vs_auto")]
    vb = [r["pm_vs_rb"] for r in rows if r.get("pm_vs_rb")]

    def geo(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    print(f"\n{cat}  PM/autotune  geomean {geo(va):.3f}  wins {sum(x>=1 for x in va)}/{len(va)}")
    print(f"{cat}  PM/rocBLAS   geomean {geo(vb):.3f}  wins {sum(x>=1 for x in vb)}/{len(vb)}")
    losers = sorted(((r["M"], r["N"], r["K"], r["pm_vs_auto"]) for r in rows
                     if r.get("pm_vs_auto") and r["pm_vs_auto"] < 1.0),
                    key=lambda x: x[3])
    if losers:
        print(f"\nlosers vs autotune ({len(losers)}):")
        for M, N, K, v in losers:
            print(f"  {M}x{N}x{K}  {v:.3f}")


if __name__ == "__main__":
    main()
