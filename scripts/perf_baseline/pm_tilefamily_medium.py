"""Ground-truth the tile-shape sweet spot for MEDIUM losers: for each shape,
MEASURE the {128,256}x{128,256} x BK{32,64} W8 S3 family and print measured
TFLOPS next to the model's predicted_tflops. Reveals the systematic bias
(model prefers small clean-wave tiles; measured prefers a mid sweet spot)."""
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

SHAPES = [
    (4096, 9216, 4096), (4096, 5120, 2880), (4096, 5120, 5120),
    (4096, 3072, 4096), (4096, 2880, 4096), (8192, 2112, 7168),
]
FAM = [(bm, bn, bk) for bm in (128, 256) for bn in (128, 256) for bk in (32, 64)]

def bench_ms(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts=[]
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12

def mk(bm,bn,bk):
    c=pm.TritonGemmConfig(); c.block_m=bm; c.block_n=bn; c.block_k=bk
    c.num_warps=8; c.num_stages=3; c.group_size_m=1; c.mfma_non_k_dim=16; c.waves_per_eu=0
    return c

def bench_cfg(a,b,M,N,K,c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),
            ACTIVATION="", **kw))
    except Exception as ex:
        return None

hw=pm.HardwareInfo.get(arch)
for (M,N,K) in SHAPES:
    prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    print(f"\n{M}x{N}x{K}")
    print(f"{'BMxBNxBK':>14}{'meas_TF':>9}{'pred_TF':>9}{'nWav':>6}{'wEff':>7}")
    res=[]
    for (bm,bn,bk) in FAM:
        c=mk(bm,bn,bk); e=pm.estimate_perf(prob,c,hw)
        ms=bench_cfg(a,b,M,N,K,c); tf=tflops(M,N,K,ms) if ms else 0
        res.append((bm,bn,bk,tf,e.predicted_tflops,e.num_waves,e.wave_efficiency))
    meas_best=max(res,key=lambda r:r[3]); pred_best=max(res,key=lambda r:r[4])
    for (bm,bn,bk,tf,pt,nw,we) in res:
        mk_meas=" <MEAS" if (bm,bn,bk)==meas_best[:3] else ""
        mk_pred=" <PRED" if (bm,bn,bk)==pred_best[:3] else ""
        print(f"{f'{bm}x{bn}x{bk}':>14}{tf:>9.0f}{pt:>9.0f}{nw:>6}{we:>7.3f}{mk_meas}{mk_pred}")
    del a,b; torch.cuda.empty_cache()
