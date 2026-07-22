import importlib.util, io, contextlib
import torch, triton
from torch.profiler import profile, ProfilerActivity
from triton.backends.amd import amd_gemm_selector as sel
TUT="/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec=importlib.util.spec_from_file_location("tut03",TUT); tut=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
arch=sel.current_amd_arch()
SHAPES=[(64,512,128),(64,128,512),(128,512,128),(128,128,512),(256,512,128),(256,128,512),
(4096,64,64),(8192,64,64),(16384,64,32),(64,4096,64),(64,8192,64),(32,16384,64)]
def bench_ms_prof(fn, warmup=20, active=60):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(active): fn()
        torch.cuda.synchronize()
    tot=0.0
    for ev in prof.key_averages():
        tot += getattr(ev,"self_device_time_total",0.0) or 0.0
    return (tot/active)/1000.0  # us -> ms
def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12
def cfgstr(c): return f"BM{c.block_m}BN{c.block_n}BK{c.block_k}W{c.num_warps}S{c.num_stages}"
def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms_prof(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION=""  ,**kw))
    except Exception: return None
print(f"{'shape':>16}{'PMus':>7}{'auus':>7}{'t1/au':>7}  PM_cfg")
for (M,N,K) in SHAPES:
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    pmc=sel.pick_gemm_config(M,N,K,"fp16",arch,top_k=1)[0]
    t1=bench_cfg(a,b,M,N,K,pmc)
    _=tut.matmul(a,b); ta=bench_ms_prof(lambda: tut.matmul(a,b))
    r=(ta/t1) if (t1 and ta) else 0   # both are times; >1 = PM faster
    print(f"{f'{M}x{N}x{K}':>16}{t1*1000:>7.1f}{ta*1000:>7.1f}{r:>7.2f}  {cfgstr(pmc)}")
    del a,b; torch.cuda.empty_cache()
