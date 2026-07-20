"""Build the FULL shape set from ALL TensorAtlas varied-M datasets.

Same sources and byte-cap as build_13cat_shapes.py, but WITHOUT the per-category
cap: emits every unique (M,N,K) that fits the OOM budget, classified into the
13 categories and tagged by source model. This is the wide-evaluation superset
of shapes_13cat.py; the sweep harness (sweep_all.py) runs the identical
PerfModel / Triton-autotune / rocBLAS measurement over it.

classify_shape / size_tier / is_balanced / dominant_dims are VENDORED verbatim
from TensorAtlas tuning/pruning.py so this runs on a CPU host with no torch.

Run (host, CPU-only):
    python3 build_all_shapes.py
Output: shapes_all.py  (ALL_SHAPES = [(M,N,K,'CAT-src'), ...])
"""
import glob, os, sys, yaml

# Host dataset dir (maps to /home/work/TensorAtlas/datasets in the container).
DATASETS_DIR = os.environ.get(
    "TA_DATASETS", "/home/xiaohugu/work/TensorAtlas/datasets")
DATASETS = sorted(glob.glob(os.path.join(DATASETS_DIR, "*_shapes_varied_m.yaml")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes_all.py")
MAX_BYTES = 4 * 1024**3   # 4 GiB fp16 A+B+C cap (matches build_13cat_shapes.py)
NUM_CUS = 256

# --- vendored classifier (tuning/pruning.py) --------------------------------
def size_tier(x, num_cus=NUM_CUS):
    tiles = (x + 15) // 16
    if tiles <= 4:            return "very_small"
    if tiles <= num_cus // 4: return "small"
    if tiles <= num_cus:      return "medium"
    if tiles <= num_cus * 4:  return "large"
    return "very_large"

def is_balanced(M, N, K, ratio=8.0):
    d = (M, N, K)
    return max(d) / max(1, min(d)) <= ratio

def dominant_dims(M, N, K, ratio=8.0):
    d = {"M": M, "N": N, "K": K}; mx = max(d.values())
    return frozenset(n for n, v in d.items() if v >= mx / ratio)

def classify_shape(M, N, K, num_cus=NUM_CUS):
    if min(M, N, K) <= 0:
        raise ValueError(f"M,N,K must be positive ({M},{N},{K})")
    mt, nt, kt = size_tier(M, num_cus), size_tier(N, num_cus), size_tier(K, num_cus)
    if is_balanced(M, N, K):
        if "very_large" in (mt, nt, kt):                       return "VERY_LARGE"
        if mt == nt == kt == "large":                          return "LARGE"
        if all(t in ("medium", "large") for t in (mt, nt, kt)): return "MEDIUM"
        return "SMALL"
    dom = dominant_dims(M, N, K)
    vs = {"M": mt == "very_small", "N": nt == "very_small", "K": kt == "very_small"}
    if dom == {"M"}: return "LARGE_M_SKINNY" if vs["N"] and vs["K"] else "LARGE_M"
    if dom == {"N"}: return "LARGE_N_SKINNY" if vs["M"] and vs["K"] else "LARGE_N"
    if dom == {"K"}: return "LARGE_K_SKINNY" if vs["M"] and vs["N"] else "LARGE_K"
    if dom == {"M", "N"}: return "LARGE_MN"
    if dom == {"N", "K"}: return "LARGE_NK"
    if dom == {"M", "K"}: return "LARGE_MK"
    if "very_large" in (mt, nt, kt): return "VERY_LARGE"
    if "medium" in (mt, nt, kt):     return "MEDIUM"
    return "SMALL"

# synthetic skinny shapes (same as build_13cat_shapes.py — real datasets have few)
SYNTHETIC_SKINNY = [
    (4096, 64, 64), (8192, 64, 64), (16384, 64, 32),
    (64, 4096, 64), (64, 8192, 64), (32, 16384, 64),
    (64, 64, 4096), (64, 64, 8192), (32, 64, 16384),
]

def total_bytes(M, N, K):
    return 2 * (M*K + K*N + M*N)   # fp16 A+B+C

def main():
    seen, shapes = set(), []
    for path in DATASETS:
        src = os.path.basename(path).replace("_shapes_varied_m.yaml", "")
        try:
            entries = yaml.safe_load(open(path))
        except Exception as ex:
            print(f"  skip {path}: {ex}"); continue
        for e in entries or []:
            key = (e["M"], e["N"], e["K"])
            if key in seen:
                continue
            seen.add(key)
            shapes.append((e["M"], e["N"], e["K"], src))
    for (M, N, K) in SYNTHETIC_SKINNY:
        if (M, N, K) not in seen:
            seen.add((M, N, K)); shapes.append((M, N, K, "synthetic"))

    raw = len(shapes)
    kept = [s for s in shapes if total_bytes(s[0], s[1], s[2]) <= MAX_BYTES]
    pruned = raw - len(kept)

    buckets, chosen = {}, []
    for (M, N, K, src) in kept:
        try:
            cat = classify_shape(M, N, K)
        except Exception:
            continue
        buckets.setdefault(cat, []).append((M, N, K, src))
        chosen.append((M, N, K, f"{cat}-{src}"))
    chosen.sort(key=lambda s: (s[3], s[0]*s[1]*s[2]))

    print(f"Sources ({len(DATASETS)}): " +
          ", ".join(os.path.basename(d) for d in DATASETS))
    print(f"Collected {raw} unique shapes, pruned {pruned} oversized (>4 GiB fp16), "
          f"kept {len(chosen)}")
    print("Per-category counts:")
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        print(f"  {cat:<18} {len(buckets[cat])}")

    with open(OUT, "w") as f:
        f.write('"""Auto-generated FULL shape set from ALL TensorAtlas varied-M '
                'datasets.\n')
        f.write(f"Sources: {', '.join(os.path.basename(d) for d in DATASETS)}\n")
        f.write(f"{len(chosen)} shapes across {len(buckets)} categories "
                f"(uncapped superset of shapes_13cat.py).\n")
        f.write('Regenerate: python3 build_all_shapes.py\n"""\n\n')
        f.write("# (M, N, K, 'CATEGORY-source')\n")
        f.write("ALL_SHAPES = [\n")
        for (M, N, K, label) in chosen:
            f.write(f"    ({M:>7}, {N:>7}, {K:>7}, '{label}'),\n")
        f.write("]\n")
    print(f"\nWrote {OUT}  ({len(chosen)} shapes)")

if __name__ == "__main__":
    main()
