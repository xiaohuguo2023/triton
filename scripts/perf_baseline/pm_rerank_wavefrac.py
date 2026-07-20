"""Validate the dense fractional-wave fix WITHOUT rebuild: re-score every
candidate charging FRACTIONAL waves (totalCycles *= waveEfficiency, i.e. use
tiles/CUs instead of ceil) on the dense compute-bound path, pick the new top-1,
and measure it vs the current deployed pick and autotune.

Current deploy = candidate with max predicted_tflops.
New            = min (effective_tile_cycles * num_waves * occP * waveEff),
                 occP = 1 (compute-bound) else 1/max(occupancy,0.25).

Run on MEDIUM losers + MEDIUM winners + VERY_LARGE + LARGE_K + skinny to confirm
the fix helps MEDIUM with no cross-category regression.
"""
import importlib.util, io, contextlib
import torch, triton
from triton._C.libtriton import amd
pm = amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel

TUT = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec = importlib.util.spec_from_file_location("tut03", TUT)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
arch = sel.current_amd_arch()

GROUPS = {
 "MEDIUM-lose": [(4096,9216,4096),(4096,5120,2880),(4096,5120,5120),(4096,3072,4096),
                 (4096,5120,8192),(4096,2880,4096),(8192,2112,7168),(4096,5120,16384),
                 (8192,3072,4096),(8192,2880,4096),(4096,2880,2880),(4096,14336,4096),(8192,2880,2880)],
 "MEDIUM-win":  [(8192,8192,8192),(4096,4096,14336),(8192,16384,5120)],
 "VERY_LARGE":  [(4096,32768,5120),(16384,8192,28672),(4096,7168,18432)],
 "LARGE_K":     [(4,128,2880),(256,256,7168)],
 "SKINNY":      [(4096,64,64),(64,64,8192)],
}

def bench_ms(fn, warmup=8, rep=25):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12
def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),
            ACTIVATION="", **kw))
    except Exception: return None

def new_cycles(e):
    occP = 1.0 if e.is_compute_bound else 1.0/max(e.occupancy,0.25)
    we = e.wave_efficiency if e.wave_efficiency>0 else 1.0
    return e.effective_tile_cycles * e.num_waves * occP * we

hw=pm.HardwareInfo.get(arch)
for gname, shapes in GROUPS.items():
    print(f"\n===== {gname} =====")
    print(f"{'M':>6}{'N':>7}{'K':>7}  {'old':>6}{'new':>6}{'auto':>6}  {'old/au':>7}{'new/au':>7}  new_cfg")
    for (M,N,K) in shapes:
        prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
        cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
        scored=[(c,pm.estimate_perf(prob,c,hw)) for c in cands]
        old=max(scored,key=lambda r:r[1].predicted_tflops)[0]
        new=min(scored,key=lambda r:new_cycles(r[1]))[0]
        a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
        mo=bench_cfg(a,b,M,N,K,old); mn=bench_cfg(a,b,M,N,K,new)
        _=tut.matmul(a,b); ma=bench_ms(lambda: tut.matmul(a,b))
        to,tn,ta=(tflops(M,N,K,x) if x else 0 for x in (mo,mn,ma))
        print(f"{M:>6}{N:>7}{K:>7}  {to:>6.0f}{tn:>6.0f}{ta:>6.0f}  {to/ta if ta else 0:>7.3f}{tn/ta if ta else 0:>7.3f}  BM{new.block_m}BN{new.block_n}BK{new.block_k}S{new.num_stages}")
        del a,b; torch.cuda.empty_cache()
