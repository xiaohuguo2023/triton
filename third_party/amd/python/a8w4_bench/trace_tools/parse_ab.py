#!/usr/bin/env python3
"""Parse forced-config A/B: per target shape, compare config A (JSON) vs B (PM)
from the SAME run (alternated call-by-call, identical routing/window/thermal).
Reads gpu_user_annotation labels emitted by the line-594 record_function patch.

NOTE: labels only attach to EAGER launches (prefill bm64/bm128). Tiny/graph-
replayed DECODE kernels (bm16/bm32) get no per-call label -> use moe_hist.py
(duration-based) for those. See trace_tools/README.md.

Usage: parse_ab.py <ab_run_dir> [run_subdir ...]   default subdirs run1 run2
"""
import gzip, json, glob, collections, re, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = sys.argv[2:] if len(sys.argv) > 2 else ["run1", "run2"]
def parse(l):
    m=re.match(r"a8w4cfg N(\d+) K(\d+) BM(\d+) BN(\d+) BK(\d+) ns(\d+) nw(\d+)",l)
    if not m: return None
    N,K,BM,BN,BK,ns,nw=map(int,m.groups())
    return (BM,N,K),(BN,BK,ns,nw)
for run in RUNS:
    lab=collections.defaultdict(lambda:[0.0,0])
    for fp in glob.glob(f"{BASE}/{run}/dp0_pp0_tp*.pt.trace.json.gz"):
        for e in json.load(gzip.open(fp))["traceEvents"]:
            if e.get("cat")=="gpu_user_annotation" and str(e.get("name","")).startswith("a8w4cfg"):
                r=parse(e["name"])
                if r: lab[r][0]+=e.get("dur",0.0); lab[r][1]+=1
    if not lab:
        print(f"[{run}] no labels yet"); continue
    print(f"\n=== {run} ===")
    print(f"{'BM,N,K':16} {'cfg(bn/bk/ns/nw)':18} {'avg_us':>8} {'calls':>7} {'total_us':>10}")
    byshape=collections.defaultdict(list)
    for (s,c),(t,n) in lab.items():
        byshape[s].append((c,t/n if n else 0,n,t))
    for s in sorted(byshape, key=lambda s:(s[0],s[1],s[2])):
        rows=sorted(byshape[s])
        for c,a,n,t in rows:
            print(f"{str(s):16} {'/'.join(map(str,c)):18} {a:8.1f} {n:7d} {t:10.0f}")
        if len(rows)==2:  # A/B pair -> verdict
            (ca,aa,na,ta),(cb,ab,nb,tb)=rows
            print(f"    -> {'/'.join(map(str,ca))} vs {'/'.join(map(str,cb))}: "
                  f"avg ratio {ab/aa:.3f} (B/A); {'B faster' if ab<aa else 'A faster'}")
