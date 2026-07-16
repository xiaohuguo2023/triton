import gzip, json, glob, sys, collections, bisect
d=sys.argv[1]
fp=sorted(glob.glob(f"{d}/dp0_pp0_tp0_*.pt.trace.json.gz"))[0]
ev=json.load(gzip.open(fp))["traceEvents"]
steps=sorted((e["ts"],e["ts"]+e.get("dur",0)) for e in ev
    if e.get("cat")=="user_annotation"
    and str(e.get("name","")).startswith("execute_context")
    and "generation_16(16)" in str(e.get("name","")))
starts=[a for a,_ in steps]
def ins(ts):
    i=bisect.bisect_right(starts,ts)-1
    return 0<=i<len(steps) and ts<=steps[i][1]
durs=[e.get("dur",0.0) for e in ev if e.get("cat")=="kernel"
      and e.get("name","").startswith("_moe_gemm_a8w4") and ins(e["ts"])]
h=collections.Counter()
for x in durs:
    b = ("<6us" if x<6 else "6-10" if x<10 else "10-20" if x<20 else
         "20-30" if x<30 else "30-50" if x<50 else ">=50us")
    h[b]+=1
order=["<6us","6-10","10-20","20-30","30-50",">=50us"]
tot=len(durs); s=sum(durs)
print(f"{d}: {tot} moe calls in decode, total {s:.0f}us, avg {s/max(1,tot):.1f}us")
for b in order:
    c=h.get(b,0); print(f"   {b:8} {c:5d} ({100*c/max(1,tot):4.0f}%)")
