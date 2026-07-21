import importlib.util, io, contextlib, csv, math
import torch, triton
from triton.backends.amd import amd_gemm_selector as sel
from shapes_all import ALL_SHAPES
TUT="/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec=importlib.util.spec_from_file_location("tut03",TUT); tut=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
arch=sel.current_amd_arch()
SHAPES=[(M,N,K) for (M,N,K,r) in ALL_SHAPES if r.split('-')[0]=='LARGE_NK']
def bench_ms(fn,w=12,r=30):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(r):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
def cfgstr(c): return f"BM{c.block_m}BN{c.block_n}BK{c.block_k}W{c.num_warps}S{c.num_stages}"
def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception: return None
res=[]
for (M,N,K) in SHAPES:
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    pmc=sel.pick_gemm_config(M,N,K,"fp16",arch,top_k=1)[0]
    t1=bench_cfg(a,b,M,N,K,pmc)
    _=tut.matmul(a,b); ta=bench_ms(lambda: tut.matmul(a,b))
    spd=(ta/t1) if (t1 and ta) else 0   # >1 = PM faster (win)
    res.append((spd,M,N,K,cfgstr(pmc)))
    del a,b; torch.cuda.empty_cache()
with open("losers_nk.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["speedup","M","N","K","pm_cfg"])
    for spd,M,N,K,c in res: w.writerow([f"{spd:.4f}",M,N,K,c])
res.sort()
losers=[x for x in res if x[0]<0.98]
print(f"=== {len(losers)} LOSERS (PM slower, speedup<0.98) of {len(res)} ===")
for spd,M,N,K,c in losers:
    print(f"  {spd:5.2f}  {M:>5}x{N:>6}x{K:<6}  {c}")
gm=math.exp(sum(math.log(x[0]) for x in res if x[0]>0)/sum(1 for x in res if x[0]>0))
print(f"geomean_speedup={gm:.3f}  win%={100*sum(1 for x in res if x[0]>=1)/len(res):.0f}")
