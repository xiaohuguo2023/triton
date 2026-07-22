"""Debug the 6 tiny-K skinny losses with profiler device-time. For each shape:
PM top-1 pick, oracle (best candidate by device-time), autotune, + the model's
predicted_tflops for PM-pick vs oracle to see if it's a ranking or candidate gap."""
import importlib.util, io, contextlib
import torch, triton
from triton.backends.amd import amd_gemm_selector as sel
from triton._C.libtriton import amd
pm=amd.perf_model
TUT="/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec=importlib.util.spec_from_file_location("tut03",TUT); tut=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
import sweep as S  # profiler device-time bench_ms
arch=sel.current_amd_arch(); hw=pm.HardwareInfo.get(arch)
SHAPES=[(4096,64,64),(8192,64,64),(16384,64,32),(64,4096,64),(64,8192,64),(32,16384,64)]
def cfgstr(c): return f"BM{c.block_m}BN{c.block_n}BK{c.block_k}W{c.num_warps}S{c.num_stages}"
def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return S.bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception: return None
def predtf(M,N,K,c):
    prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
    return pm.estimate_perf(prob,c,hw).predicted_tflops
print(f"{'shape':>15}{'PMus':>7}{'orus':>7}{'auus':>7}{'t1/au':>7}{'or/au':>7}{'PMpredTF':>9}{'orpredTF':>9}  PM->oracle(rank)")
for (M,N,K) in SHAPES:
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    ranked=sel.pick_gemm_config(M,N,K,"fp16",arch,top_k=400)
    pmc=ranked[0]; t1=bench_cfg(a,b,M,N,K,pmc)
    bo=t1; boc=pmc; bor=0
    for i,c in enumerate(ranked[:120]):
        ms=bench_cfg(a,b,M,N,K,c)
        if ms and (bo is None or ms<bo): bo,boc,bor=ms,c,i
    _=tut.matmul(a,b); ta=S.bench_ms(lambda: tut.matmul(a,b))
    print(f"{f'{M}x{N}x{K}':>15}{t1*1e3:>7.1f}{bo*1e3:>7.1f}{ta*1e3:>7.1f}{ta/t1 if t1 else 0:>7.2f}{ta/bo if bo else 0:>7.2f}{predtf(M,N,K,pmc):>9.0f}{predtf(M,N,K,boc):>9.0f}  {cfgstr(pmc)} -> {cfgstr(boc)}(#{bor})")
    del a,b; torch.cuda.empty_cache()
