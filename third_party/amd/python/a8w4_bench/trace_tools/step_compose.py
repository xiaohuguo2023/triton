import gzip, json, glob, sys, collections, re
d=sys.argv[1]
fp=sorted(glob.glob(f"{d}/dp0_pp0_tp0_*.pt.trace.json.gz"))[0]
ev=json.load(gzip.open(fp))["traceEvents"]
# execute_context_<prefill>(<pf_tok>)_generation_<gen>(<gen_tok>)
pf=collections.Counter(); gen=collections.Counter()
mixed=0; total=0
for e in ev:
    if e.get("cat")=="user_annotation":
        n=str(e.get("name",""))
        m=re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)",n)
        if m and m.group(3)=="16":  # generation_16 steps
            total+=1
            pft=int(m.group(2))
            pf[pft]+=1
            if pft>0: mixed+=1
print(f"{d}: {total} generation_16 steps; {mixed} have prefill tokens mixed in ({100*mixed/max(1,total):.0f}%)")
print("  prefill-token-count histogram (top):", pf.most_common(6))
