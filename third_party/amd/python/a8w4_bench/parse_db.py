#!/usr/bin/env python3
"""Normalize the baseline-DB sweep into two CSVs.

per_cell.csv     : one row per (arm, tp, conc, isl, osl) with throughput+latency+meta.
per_a8w4.csv     : one row per (cell, block_m, N, K, swizzle) with PM cfg, JSON cfg,
                   grid membership + excluding field, PM estimate(PM cfg) & (JSON cfg),
                   runtime frequency.

Run inside xguo-nightly-latest (needs perf_model.so + aiter gfx950-A8W4.json).
Reads the DB dir directly; safe to run on a partially-complete sweep.
"""
import json, glob, os, re, csv, collections, statistics, sys
DB = "/home/xiaohugu/work/sweep_gptoss_output/pm_baseline_db"
sys.path.insert(0, "/home/xiaohugu/openai")
import perfmodel_a8w4_select as S
pm = S._pm; HW = S._hw("gfx950")
F8, F4, F32 = pm.ElemKind.FP8, pm.ElemKind.FP4, pm.ElemKind.FP32
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import AITER_TRITON_CONFIGS_PATH as P
JSON = json.load(open(f"{P}/moe/gfx950-A8W4.json"))

meta = {}
if os.path.exists(f"{DB}/run_meta.txt"):
    for line in open(f"{DB}/run_meta.txt"):
        if "=" in line:
            k, v = line.strip().split("=", 1); meta[k] = v
IMAGE = meta.get("image", "?"); REV = meta.get("perf_model_revision", "?")

TAG_RE = re.compile(r"(?P<arm>json|pmcanon|pmexact|pmbucket|pmbcnw|pmcapped|pmpol)_isl(?P<isl>\d+)_osl(?P<osl>\d+)_tp(?P<tp>\d+)_conc(?P<conc>\d+)")

def parse_stdout_tput(f):
    tot = out = None
    try:
        t = open(f).read()
    except Exception:
        return None, None
    m = re.search(r"Total Token throughput \(tok/s\):\s*([0-9.]+)", t);  tot = float(m.group(1)) if m else None
    m = re.search(r"Output token throughput \(tok/s\):\s*([0-9.]+)", t); out = float(m.group(1)) if m else None
    return tot, out

def load_result(tag):
    for cand in (f"{DB}/{tag}.result.json", f"{DB}/{tag}.json.json"):
        if os.path.exists(cand):
            try: return json.load(open(cand))
            except Exception: return None
    return None

# ---------------- per-cell CSV ----------------
cells = {}
for f in sorted(glob.glob(f"{DB}/*.stdout")):
    m = TAG_RE.search(os.path.basename(f))
    if not m: continue
    d = m.groupdict(); tag = os.path.basename(f)[:-len(".stdout")]
    tot, out = parse_stdout_tput(f)
    r = load_result(tag) or {}
    cells[tag] = dict(arm=d["arm"], tp=int(d["tp"]), conc=int(d["conc"]),
                      isl=int(d["isl"]), osl=int(d["osl"]),
                      image=IMAGE, revision=REV,
                      total_tput=tot, output_tput=out or r.get("output_throughput"),
                      total_token_throughput=r.get("total_token_throughput"),
                      request_throughput=r.get("request_throughput"),
                      completed=r.get("completed"), num_prompts=r.get("num_prompts"),
                      duration=r.get("duration"),
                      mean_ttft_ms=r.get("mean_ttft_ms"), p99_ttft_ms=r.get("p99_ttft_ms"),
                      mean_tpot_ms=r.get("mean_tpot_ms"), p99_tpot_ms=r.get("p99_tpot_ms"),
                      mean_e2el_ms=r.get("mean_e2el_ms"),
                      success=(tot is not None),
                      server_log=f"{tag}.server.log", stdout=f"{tag}.stdout",
                      picklog=(f"{tag}.picklog" if d["arm"].startswith("pm") else ""))
cols = ["arm","tp","conc","isl","osl","image","revision","total_tput","output_tput",
        "total_token_throughput","request_throughput","completed","num_prompts","duration",
        "mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","p99_tpot_ms","mean_e2el_ms","success",
        "server_log","stdout","picklog"]
with open(f"{DB}/per_cell.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for tag in sorted(cells): w.writerow({k: cells[tag].get(k) for k in cols})
print(f"per_cell.csv: {len(cells)} rows")

# ---------------- per-A8W4-shape CSV ----------------
def grid_excl(v):
    bad = []
    if v["BLOCK_SIZE_K"] not in (256,512): bad.append(f"BK={v['BLOCK_SIZE_K']}")
    if v["num_stages"] not in (2,3):        bad.append(f"ns={v['num_stages']}")
    if v.get("matrix_instr_nonkdim",16)!=16: bad.append(f"nkd={v['matrix_instr_nonkdim']}")
    if v.get("waves_per_eu",0)!=0:          bad.append(f"wpe={v['waves_per_eu']}")
    if v["BLOCK_SIZE_N"] not in (64,128,256,512): bad.append(f"BN={v['BLOCK_SIZE_N']}")
    if v["num_warps"] not in (4,8):         bad.append(f"nw={v['num_warps']}")
    return bad

def est_tf(bm, N, K, m, cfg):
    try:
        e = pm.estimate_perf(pm.GemmProblem(int(m),N,K,F8,F4,F32,8,4,32), S._mk(bm,*cfg), HW)
        return round(e.predicted_tflops,1)
    except Exception:
        return None

shape_rows = []
for f in sorted(glob.glob(f"{DB}/pmcanon_*.picklog")):
    m = TAG_RE.search(os.path.basename(f))
    if not m: continue
    d = m.groupdict()
    agg = collections.defaultdict(lambda: {"ms": [], "pm": None})
    for line in open(f):
        p = line.split()
        if len(p) < 9: continue
        mm,n,k,bm,bn,bk,ns,nw,sw = int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5]),int(p[6]),int(p[7]),p[8]
        key = (bm,n,k,sw); agg[key]["ms"].append(mm); agg[key]["pm"]=(bn,bk,ns,nw)
    for (bm,n,k,sw),info in agg.items():
        mmed = int(statistics.median(info["ms"])); freq = len(info["ms"]); pmc = info["pm"]
        jv = JSON.get(f"bm{bm}_n{n}_k{k}")
        jcfg = (jv["BLOCK_SIZE_N"],jv["BLOCK_SIZE_K"],jv["num_stages"],jv["num_warps"]) if jv else None
        excl = grid_excl(jv) if jv else ["<no-json-entry>"]
        shape_rows.append(dict(tp=int(d["tp"]),conc=int(d["conc"]),isl=int(d["isl"]),osl=int(d["osl"]),
            block_m=bm,N=n,K=k,swizzle=sw,freq=freq,m_median=mmed,
            pm_cfg="/".join(map(str,pmc)),
            json_cfg=("/".join(map(str,jcfg)) if jcfg else ""),
            json_in_grid=(jv is not None and not excl),
            json_excl_field=(",".join(excl) if excl else ""),
            pm_est_tflops=est_tf(bm,n,k,mmed,pmc),
            json_est_tflops=(est_tf(bm,n,k,mmed,jcfg) if jcfg else None),
            pm_eq_json=(jcfg is not None and pmc==jcfg)))
scols = ["tp","conc","isl","osl","block_m","N","K","swizzle","freq","m_median",
         "pm_cfg","json_cfg","json_in_grid","json_excl_field","pm_est_tflops","json_est_tflops","pm_eq_json"]
with open(f"{DB}/per_a8w4.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=scols); w.writeheader()
    for r in shape_rows: w.writerow(r)
print(f"per_a8w4.csv: {len(shape_rows)} rows")
