"""MEDIUM-only re-bench (physics large-tile fix). Runs the SAME measure_shape
harness over just the MEDIUM shapes from shapes_all, reporting per-shape PM/auto
(robust to slow pollution: PM-pick and autotune are measured back-to-back, so
their ratio cancels chassis-level drift). Compare geomean to the tie-break
baseline MEDIUM = 1.076."""
import importlib.util, io, contextlib, sys, math
from pathlib import Path
import torch, triton
TUT = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec = importlib.util.spec_from_file_location("tut03", TUT)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
sys.path.insert(0, str(Path(__file__).parent))
from shapes_all import ALL_SHAPES
import sweep as base_sweep

MED = [(M,N,K,r) for (M,N,K,r) in ALL_SHAPES if r.split('-')[0] == "MEDIUM"]
print(f"MEDIUM shapes: {len(MED)}")
rows=[]
for i,(M,N,K,r) in enumerate(MED,1):
    row = base_sweep.measure_shape(M,N,K); row["regime"]=r
    rows.append(row)
    va = row.get("pm_vs_auto")
    print(f"  [{i:2}/{len(MED)}] {M}x{N}x{K:<6} PM/auto={va:.3f}" if va else f"  [{i}] {M}x{N}x{K} FAIL", flush=True)

va=[r["pm_vs_auto"] for r in rows if r.get("pm_vs_auto")]
vb=[r["pm_vs_rb"] for r in rows if r.get("pm_vs_rb")]
def g(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs))
print(f"\nMEDIUM PM/auto  geomean {g(va):.3f}  wins {sum(x>=1 for x in va)}/{len(va)}   (tie-break baseline 1.076, 40/53)")
print(f"MEDIUM PM/rocBLAS geomean {g(vb):.3f}  wins {sum(x>=1 for x in vb)}/{len(vb)}")
losers=sorted([(r['M'],r['N'],r['K'],r['pm_vs_auto']) for r in rows if r.get('pm_vs_auto') and r['pm_vs_auto']<1], key=lambda x:x[3])
print(f"\nremaining losers ({len(losers)}):")
for M,N,K,v in losers: print(f"  {M}x{N}x{K}  {v:.3f}")
