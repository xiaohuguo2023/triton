import importlib.util, io, contextlib
import torch, triton
from triton.backends.amd import amd_gemm_selector as sel
TUT="/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec=importlib.util.spec_from_file_location("tut03",TUT); tut=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
arch=sel.current_amd_arch()
LOSERS=[(64,36864,7168),(64,28672,4096),(64,28672,8192),(128,53248,16384),(64,18432,16384),
(128,57344,8192),(128,106496,16384),(64,10240,8192),(128,36864,7168),(64,9216,4096)]
def bench_ms(fn,w=12,r=30):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(r):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12
def cfgstr(c): return f"BM{c.block_m}BN{c.block_n}BK{c.block_k}W{c.num_warps}S{c.num_stages}"
def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception: return None
print(f"{'shape':>18}{'PMtop1':>8}{'oracle':>8}{'auto':>7}{'t1/au':>7}{'or/au':>7}  PM_cfg -> oracle_cfg(rank)")
for (M,N,K) in LOSERS:
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    ranked=sel.pick_gemm_config(M,N,K,"fp16",arch,top_k=300)
    pmc=ranked[0]; t1=bench_cfg(a,b,M,N,K,pmc)
    bo=t1; boc=pmc; bor=0
    for i,c in enumerate(ranked[:80]):
        ms=bench_cfg(a,b,M,N,K,c)
        if ms and (bo is None or ms<bo): bo,boc,bor=ms,c,i
    _=tut.matmul(a,b); ta=bench_ms(lambda: tut.matmul(a,b))
    T1,BO,TA=(tflops(M,N,K,x) if x else 0 for x in (t1,bo,ta))
    print(f"{f'{M}x{N}x{K}':>18}{T1:>8.0f}{BO:>8.0f}{TA:>7.0f}{T1/TA if TA else 0:>7.2f}{BO/TA if TA else 0:>7.2f}  {cfgstr(pmc)} -> {cfgstr(boc)}(#{bor})")
    del a,b; torch.cuda.empty_cache()
