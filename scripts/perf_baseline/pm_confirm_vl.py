"""Confirm the VERY_LARGE regression reason: the boosted pick 256x256 is slower
than the smaller 128x256 / 256x128 that the model over-ranked past it."""
import importlib.util, io, contextlib
import torch, triton
from triton.backends.amd import amd_gemm_selector as sel
from triton._C.libtriton import amd
pm = amd.perf_model
TUT = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec = importlib.util.spec_from_file_location("tut03", TUT)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
def bench_ms(fn,w=10,r=30):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(r):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
def mk(bm,bn,bk):
    c=pm.TritonGemmConfig(); c.block_m=bm;c.block_n=bn;c.block_k=bk;c.num_warps=8;c.num_stages=3;c.group_size_m=1;c.mfma_non_k_dim=16;c.waves_per_eu=0; return c
def bench(a,b,M,N,K,bm,bn,bk):
    try:
        c=mk(bm,bn,bk); kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,bm)*triton.cdiv(N,bn),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception: return None
def tf(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12
arch=sel.current_amd_arch()
for (M,N,K) in [(4096,28672,4096),(4096,32768,5120),(16384,8192,28672)]:
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    pk=sel.pick_gemm_config(M,N,K,"fp16",arch,top_k=1)[0]
    print(f"\n{M}x{N}x{K}  physics-pick=BM{pk.block_m}BN{pk.block_n}BK{pk.block_k}S{pk.num_stages}")
    for (bm,bn,bk) in [(256,256,32),(128,256,64),(256,128,64),(128,128,64)]:
        ms=bench(a,b,M,N,K,bm,bn,bk)
        print(f"   {bm}x{bn}x{bk}: {tf(M,N,K,ms):.0f} TF" if ms else f"   {bm}x{bn}x{bk}: FAIL")
    del a,b; torch.cuda.empty_cache()
