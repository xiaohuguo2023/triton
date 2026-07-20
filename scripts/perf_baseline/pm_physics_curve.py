"""Map compute efficiency vs MFMA accumulator count m=(BM/16)*(BN/16) in the
clean-wave regime (M=N=16384), to derive the latency-hiding saturation curve.

Hypothesis: eff(tile) = kPeak * laneUtil(minTileDim) * m/(m+m_half)
  laneUtil = min(1, minTileDim/128)     (narrow-tile MFMA lane waste)
  m/(m+m_half)                          (MFMA accumulator latency hiding, saturating)
Fit m_half and kPeak from measured TFLOPS; check 128x256 ~= 256x128 (m-driven).
"""
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

M = N = 16384
K = 8192
# tiles spanning m = (BM/16)*(BN/16); use BK32 so big tiles fit LDS
TILES = [(64,64),(64,128),(128,64),(128,128),(64,256),(256,64),
         (128,256),(256,128),(256,256)]
BK = 32

def bench_ms(fn, warmup=10, rep=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def mk(bm,bn,bk):
    c=pm.TritonGemmConfig(); c.block_m=bm; c.block_n=bn; c.block_k=bk
    c.num_warps=8; c.num_stages=3; c.group_size_m=1; c.mfma_non_k_dim=16; c.waves_per_eu=0
    return c

def bench(a,b,bm,bn,bk):
    try:
        c=mk(bm,bn,bk); kw=sel.config_to_kernel_kwargs(c)
        out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,bm)*triton.cdiv(N,bn),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),
            ACTIVATION="", **kw))
    except Exception as ex:
        return None

a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
rows=[]
for (bm,bn) in TILES:
    ms=bench(a,b,bm,bn,BK)
    if ms is None: continue
    tf=2*M*N*K/(ms*1e-3)/1e12
    m=(bm//16)*(bn//16)
    rows.append((bm,bn,m,min(bm,bn),tf))
peak=max(r[4] for r in rows)
print(f"clean-wave M=N={M} K={K} BK={BK}   (peak observed {peak:.0f} TF)")
print(f"{'tile':>10}{'m=acc':>7}{'minDim':>8}{'TFLOP/s':>9}{'frac_pk':>9}")
for (bm,bn,m,mind,tf) in sorted(rows,key=lambda r:r[2]):
    print(f"{f'{bm}x{bn}':>10}{m:>7}{mind:>8}{tf:>9.0f}{tf/peak:>9.3f}")

# fit eff = kPeak * min(1,minDim/128) * m/(m+mh) via grid search on mh, least-sq kPeak
import math
best=None
for mh in [10,20,30,40,50,59,70,90,120,160,200]:
    # solve kPeak by least squares: eff_i = kPeak * f_i ; f_i = laneutil*m/(m+mh)
    num=den=0.0
    for (bm,bn,m,mind,tf) in rows:
        f=min(1.0,mind/128.0)*m/(m+mh)
        num+=f*(tf/peak); den+=f*f
    kP=num/den
    err=sum((tf/peak-kP*min(1.0,mind/128.0)*m/(m+mh))**2 for (bm,bn,m,mind,tf) in rows)
    if best is None or err<best[0]: best=(err,mh,kP)
err,mh,kP=best
print(f"\nBEST FIT: eff/peak = {kP:.3f} * min(1,minDim/128) * m/(m+{mh})   (SSE={err:.4f})")
print(f"{'tile':>10}{'meas':>8}{'fit':>8}")
for (bm,bn,m,mind,tf) in sorted(rows,key=lambda r:r[2]):
    fit=kP*min(1.0,mind/128.0)*m/(m+mh)
    print(f"{f'{bm}x{bn}':>10}{tf/peak:>8.3f}{fit:>8.3f}")
