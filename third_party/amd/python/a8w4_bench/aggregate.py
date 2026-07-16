#!/usr/bin/env python3
"""Aggregates from per_cell.csv + per_a8w4.csv:
  - PM/JSON geomean by TP, by CONC, by shape, overall
  - worst-20 cells (lowest PM/JSON)
  - worst-20 A8W4 shapes (freq-weighted, PM!=JSON)
  - root-cause histogram (A grid-unreachable; in-grid-diff pending measured B/C)
Plain csv/math (no pandas dependency)."""
import csv, math, collections, os
DB = "/home/xiaohugu/work/sweep_gptoss_output/pm_baseline_db"

def rd(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []

cells = rd(f"{DB}/per_cell.csv")
shapes = rd(f"{DB}/per_a8w4.csv")

def fnum(x):
    try: return float(x)
    except Exception: return None

# pair json vs pmcanon vs pmexact per (tp,conc,isl,osl)
by_key = collections.defaultdict(dict)
for c in cells:
    if c["arm"] in ("json","pmcanon","pmexact","pmbucket","pmbcnw","pmcapped","pmpol"):
        by_key[(c["tp"],c["conc"],c["isl"],c["osl"])][c["arm"]] = fnum(c["total_tput"])
ratios = []   # (key, ratio)
for key,arms in by_key.items():
    j,p = arms.get("json"), arms.get("pmcanon")
    if j and p: ratios.append((key, p/j))

def geo(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

print(f"=== PM/JSON total-throughput ratio ({len(ratios)} paired cells) ===")
print(f"overall geomean = {geo([r for _,r in ratios]):.4f}   below-parity(<0.99) = {sum(1 for _,r in ratios if r<0.99)}/{len(ratios)}")
for dim,idx,label in [("tp",0,"TP"),("conc",1,"CONC"),("isl",2,"shape ISL")]:
    print(f"\n  geomean by {label}:")
    g = collections.defaultdict(list)
    for key,r in ratios: g[key[idx]].append(r)
    for v in sorted(g, key=lambda x:int(x)):
        print(f"    {label}={v:>5}: {geo(g[v]):.4f}  (n={len(g[v])})")

print("\n=== worst 20 cells (lowest PM-canonical/JSON) ===")
for key,r in sorted(ratios, key=lambda x:x[1])[:20]:
    tp,conc,isl,osl = key
    print(f"  tp{tp:>1} conc{conc:>4} isl{isl}/{osl}: {r:.3f}")

# ---- exact-m measured pass: 3-arm comparison + C/B-A/mixed verdict ----
exact_cells = [(k,a) for k,a in by_key.items() if a.get("pmexact") and a.get("json") and a.get("pmcanon")]
if exact_cells:
    print(f"\n=== EXACT-M measured pass ({len(exact_cells)} losing cells) ===")
    print(f"{'cell':22} {'canon/json':>10} {'exact/json':>10} {'exact/canon':>11}  verdict")
    verdicts = collections.Counter()
    for key,a in sorted(exact_cells, key=lambda x: (x[1]['pmcanon']/x[1]['json'])):
        tp,conc,isl,osl = key
        j,pc,pe = a["json"], a["pmcanon"], a["pmexact"]
        cj, ej, ec = pc/j, pe/j, pe/pc
        gap = 1.0 - cj                      # canonical's gap to JSON
        recov = (ej - cj) / gap if gap > 1e-6 else 0.0   # fraction of gap exact recovers
        if ej >= 0.99:                       v = "C-confirmed (exact recovers)"
        elif recov >= 0.6:                   v = "C-mostly (exact recovers most)"
        elif recov <= 0.15:                  v = "B/A-confirmed (exact no help)"
        else:                                v = "mixed (partial recovery)"
        verdicts[v.split(" (")[0]] += 1
        print(f"  tp{tp} conc{conc:>4} {isl}/{osl:<5} {cj:>10.3f} {ej:>10.3f} {ec:>11.3f}  {v}")
    print("\n  verdict histogram:", dict(verdicts))
    ge = geo([a["pmexact"]/a["json"] for _,a in exact_cells])
    gc = geo([a["pmcanon"]/a["json"] for _,a in exact_cells])
    print(f"  over losing cells: geomean canon/json={gc:.4f}  ->  exact/json={ge:.4f}")

# ---- guarded m-bucket selector policy: recovered / unchanged / regressed ----
bucket_cells = [(k,a) for k,a in by_key.items() if a.get("pmbucket") and a.get("json") and a.get("pmcanon")]
if bucket_cells:
    print(f"\n=== GUARDED M-BUCKET selector ({len(bucket_cells)} losing cells) ===")
    print(f"{'cell':22} {'canon/json':>10} {'bucket/json':>11} {'bkt/canon':>10}  outcome")
    out = collections.Counter()
    for key,a in sorted(bucket_cells, key=lambda x: x[1]['pmcanon']/x[1]['json']):
        tp,conc,isl,osl = key
        j,pc,pb = a["json"], a["pmcanon"], a["pmbucket"]
        cj, bj, bc = pc/j, pb/j, pb/pc
        if bc >= 1.02:   o = "recovered"
        elif bc <= 0.98: o = "REGRESSED"
        else:            o = "unchanged"
        out[o] += 1
        print(f"  tp{tp} conc{conc:>4} {isl}/{osl:<5} {cj:>10.3f} {bj:>11.3f} {bc:>10.3f}  {o}")
    print("\n  outcome histogram:", dict(out))
    gb = geo([a["pmbucket"]/a["json"] for _,a in bucket_cells])
    gc = geo([a["pmcanon"]/a["json"] for _,a in bucket_cells])
    print(f"  over losing cells: geomean canon/json={gc:.4f}  ->  bucket/json={gb:.4f}")
    reg = [(k,a) for k,a in bucket_cells if a["pmbucket"]/a["pmcanon"] <= 0.98]
    if reg:
        print(f"  !! {len(reg)} REGRESSED cells (bucket worse than canonical) -> guard needs tightening:")
        for key,a in reg: print(f"     tp{key[0]} conc{key[1]} {key[2]}/{key[3]}: bkt/canon={a['pmbucket']/a['pmcanon']:.3f}")

# ---- bucket+nwcap (B fix) e2e: the decision checkpoint ----
bcnw = [(k,a) for k,a in by_key.items() if a.get("pmbcnw") and a.get("json") and a.get("pmcanon")]
if bcnw:
    print(f"\n=== BUCKET + NW-CAP (B fix) vs baselines ({len(bcnw)} losing cells) ===")
    print(f"{'cell':22} {'canon/j':>8} {'exact/j':>8} {'bkt/j':>7} {'bcnw/j':>7} {'bcnw/canon':>10}  verdict")
    out=collections.Counter()
    for key,a in sorted(bcnw, key=lambda x: x[1]['pmcanon']/x[1]['json']):
        tp,conc,isl,osl=key; j=a['json']
        cj=a['pmcanon']/j; bn=a['pmbcnw']/j
        ex=(a['pmexact']/j) if a.get('pmexact') else None
        bk=(a['pmbucket']/j) if a.get('pmbucket') else None
        bc=a['pmbcnw']/a['pmcanon']
        if bn>=0.99: v="tie/beats JSON"
        elif bc>=1.02: v="improves vs canon"
        elif bc<=0.98: v="REGRESSED vs canon"
        else: v="~same as canon"
        out[v]+=1
        print(f"  tp{tp} conc{conc:>4} {isl}/{osl:<5} {cj:>8.3f} {(f'{ex:.3f}' if ex else '-'):>8} "
              f"{(f'{bk:.3f}' if bk else '-'):>7} {bn:>7.3f} {bc:>10.3f}  {v}")
    print("\n  verdict histogram:", dict(out))
    gc=geo([a['pmcanon']/a['json'] for _,a in bcnw]); gb=geo([a['pmbcnw']/a['json'] for _,a in bcnw])
    print(f"  over losing cells: geomean canon/json={gc:.4f}  ->  bcnw/json={gb:.4f}")

# ---- BN-cap fix (block_m<=32 -> BN<=128) whole-surface vs JSON + vs bucket ----
capped = [(k,a) for k,a in by_key.items() if a.get("pmcapped") and a.get("json")]
if capped:
    gcap = geo([a["pmcapped"]/a["json"] for _,a in capped])
    print(f"\n=== BN-CAP FIX (pmcapped/json) whole surface ({len(capped)} cells) ===")
    print(f"  overall geomean = {gcap:.4f}   below-parity(<0.99) = "
          f"{sum(1 for _,a in capped if a['pmcapped']/a['json']<0.99)}/{len(capped)}")
    for dim,idx,label in [("tp",0,"TP"),("conc",1,"CONC"),("isl",2,"shape ISL")]:
        g=collections.defaultdict(list)
        for key,a in capped: g[key[idx]].append(a["pmcapped"]/a["json"])
        print(f"  by {label}: " + "  ".join(f"{v}={geo(g[v]):.4f}" for v in sorted(g,key=lambda x:int(x))))
    # did the 5 previously-regressed bucket cells recover?
    prev_reg = [("8","16","1024","1024"),("8","16","8192","1024"),("8","8","8192","1024"),
                ("2","256","1024","1024"),("2","64","1024","1024")]
    print("\n  previously bucket-regressed cells (capped vs bucket vs json):")
    print(f"    {'cell':20} {'bkt/json':>9} {'cap/json':>9} {'cap/bkt':>8}")
    for key in prev_reg:
        a=by_key.get(key,{})
        if a.get("pmcapped") and a.get("json"):
            bk=(a['pmbucket']/a['json']) if a.get('pmbucket') else float('nan')
            cj=a['pmcapped']/a['json']; cb=(a['pmcapped']/a['pmbucket']) if a.get('pmbucket') else float('nan')
            print(f"    tp{key[0]} conc{key[1]:>4} {key[2]}/{key[3]:<5} {bk:>9.3f} {cj:>9.3f} {cb:>8.3f}")
    # worst cells under the cap fix
    print("\n  worst 10 cells (lowest pmcapped/json):")
    for key,a in sorted(capped, key=lambda x:x[1]['pmcapped']/x[1]['json'])[:10]:
        print(f"    tp{key[0]} conc{key[1]:>4} {key[2]}/{key[3]}: {a['pmcapped']/a['json']:.3f}")

# ---- A8W4 shape root-cause ----
print("\n=== A8W4 shape attribution (unique block_m,N,K,swizzle across all cells) ===")
uniq = {}
for s in shapes:
    key=(s["block_m"],s["N"],s["K"],s["swizzle"])
    f=int(s["freq"])
    if key not in uniq or f>uniq[key]["freq"]:
        uniq[key]=dict(freq=f, pm=s["pm_cfg"], json=s["json_cfg"], in_grid=s["json_in_grid"],
                       excl=s["json_excl_field"], pm_est=s["pm_est_tflops"], json_est=s["json_est_tflops"],
                       eq=s["pm_eq_json"])
def fnum(x):
    try: return float(x)
    except Exception: return None
buckets=collections.Counter()
for key,u in uniq.items():
    if u["eq"]=="True":
        buckets["match (PM==JSON)"]+=1; u["cls"]="match"
    elif u["in_grid"]=="False":
        b="A: grid-unreachable ("+ (u["excl"] or "?") +")"; buckets[b]+=1; u["cls"]="A"
    else:
        pe,je=fnum(u["pm_est"]),fnum(u["json_est"])
        if pe is not None and je is not None and je>pe*1.02:
            buckets["C: cache/canonical (model prefers JSON, selector didn't)"]+=1; u["cls"]="C"
        else:
            buckets["B: ranking (model over-ranks PM pick)"]+=1; u["cls"]="B"
print("  root-cause histogram:")
for k,v in buckets.most_common(): print(f"    {v:>3}  {k}")

print("\n=== worst A8W4 shapes: PM!=JSON, by freq ===")
diff=[(key,u) for key,u in uniq.items() if u["eq"]!="True"]
for key,u in sorted(diff, key=lambda x:-x[1]["freq"])[:20]:
    bm,n,k,sw=key
    print(f"  bm{bm} n{n} k{k} sw={sw} freq={u['freq']:>7}  PM={u['pm']} JSON={u['json']}  "
          f"est PM={u['pm_est']} JSON={u['json_est']}  [{u.get('cls','?')}]")
