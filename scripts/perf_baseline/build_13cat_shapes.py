"""Build a 13-category shape set by classifying all shapes in TensorAtlas
datasets and picking representative shapes per category.

Output: shapes_13cat.py with ALL_SHAPES list, ~3-5 representative shapes
per category, ~40-60 total.
"""
import glob, sys, yaml
sys.path.insert(0, '/home/work/TensorAtlas')

from tuning.pruning import classify_shape, ShapeCategory

DATASETS = sorted(glob.glob('/home/work/TensorAtlas/datasets/*_varied_m.yaml'))

def collect_all_shapes():
    seen = set()
    shapes = []
    for path in DATASETS:
        try:
            with open(path) as f:
                entries = yaml.safe_load(f)
            for e in entries:
                key = (e['M'], e['N'], e['K'])
                if key in seen: continue
                seen.add(key)
                shapes.append({'M': e['M'], 'N': e['N'], 'K': e['K'],
                               'src': path.split('/')[-1].replace('_shapes_varied_m.yaml', '').replace('.yaml','')})
        except Exception as ex:
            print(f"  skip {path}: {ex}")
    return shapes

SYNTHETIC_SKINNY = [
    # LARGE_M_SKINNY: M dominant, N/K very small (≤64)
    {'M': 4096,  'N': 64, 'K': 64, 'src': 'synthetic'},
    {'M': 8192,  'N': 64, 'K': 64, 'src': 'synthetic'},
    {'M': 16384, 'N': 64, 'K': 32, 'src': 'synthetic'},
    # LARGE_N_SKINNY: N dominant, M/K very small
    {'M': 64, 'N': 4096,  'K': 64, 'src': 'synthetic'},
    {'M': 64, 'N': 8192,  'K': 64, 'src': 'synthetic'},
    {'M': 32, 'N': 16384, 'K': 64, 'src': 'synthetic'},
    # LARGE_K_SKINNY: K dominant
    {'M': 64, 'N': 64, 'K': 4096,  'src': 'synthetic'},
    {'M': 64, 'N': 64, 'K': 8192,  'src': 'synthetic'},
    {'M': 32, 'N': 64, 'K': 16384, 'src': 'synthetic'},
]

# Categories with too many shapes — cap to keep total manageable
CAP_PER_CAT = {
    'LARGE_NK':   3,  # 263 shapes — heavily over-represented
    'VERY_LARGE': 3,
    'MEDIUM':     4,
    'LARGE_K':    3,
    'LARGE_MK':   3,
    'LARGE':      3,
    'LARGE_N':    3,
    'LARGE_MN':   3,
    'SMALL':      4,
    'LARGE_M':    3,
    'LARGE_M_SKINNY': 3,
    'LARGE_N_SKINNY': 3,
    'LARGE_K_SKINNY': 3,
}

def main(max_total=60):
    raw = collect_all_shapes() + SYNTHETIC_SKINNY
    # Cap by total fp16 bytes: A + B + C must fit comfortably (< 8 GB combined)
    # to avoid OOM and crashes during bench setup. The gigantic LLM shapes
    # like 8192×106496×16384 (>6 GB just for B) are pruned here.
    MAX_BYTES = 4 * 1024**3  # 4 GiB cap (conservative — torch overhead + workspace can 2-3x the raw tensor footprint)
    def total_bytes(s):
        return 2 * (s['M']*s['K'] + s['K']*s['N'] + s['M']*s['N'])  # fp16 A+B+C
    shapes = [s for s in raw if total_bytes(s) <= MAX_BYTES]
    pruned = len(raw) - len(shapes)
    print(f"Collected {len(raw)} unique shapes ({len(SYNTHETIC_SKINNY)} synthetic skinny), pruned {pruned} oversized (>8 GiB)")
    # Bucket by category
    buckets = {}
    for s in shapes:
        try:
            cat = classify_shape(s['M'], s['N'], s['K'], num_cus=256)
        except Exception:
            continue
        buckets.setdefault(cat.name, []).append(s)
    print(f"\nPer-category counts:")
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        print(f"  {cat:<20} {len(buckets[cat])}")
    # Pick representatives: prefer geometric-mean-diverse shapes
    import random
    random.seed(42)
    chosen = []
    for cat, items in sorted(buckets.items()):
        per_cat = CAP_PER_CAT.get(cat, 3)
        items_sorted = sorted(items, key=lambda s: s['M']*s['N']*s['K'])
        if len(items_sorted) <= per_cat:
            picks = items_sorted
        else:
            idxs = [int(i * (len(items_sorted)-1) / (per_cat-1)) for i in range(per_cat)]
            picks = [items_sorted[i] for i in sorted(set(idxs))]
        for p in picks:
            chosen.append({**p, 'cat': cat})
    if len(chosen) > max_total:
        # Trim by reducing per_cat further on largest buckets
        # Simple approach: keep first per_cat from each category until under cap
        pass
    print(f"\nChosen {len(chosen)} shapes covering {len(buckets)}/13 categories")
    out = '/home/openai/triton_gluon_perfmodel/scripts/perf_baseline/shapes_13cat.py'
    with open(out, 'w') as f:
        f.write('"""Auto-generated 13-category shape set from TensorAtlas classifier.\n')
        f.write(f'Sources: {", ".join(d.split("/")[-1] for d in DATASETS)}\n')
        f.write('"""\n\n')
        f.write('# (M, N, K, regime_label)\n')
        f.write('ALL_SHAPES = [\n')
        for s in chosen:
            label = f"{s['cat']}-{s['src']}"
            f.write(f"    ({s['M']:>6}, {s['N']:>6}, {s['K']:>6}, '{label}'),\n")
        f.write(']\n')
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()
