"""Physics study: real per-tile cycle cost of 128x128 / 128x256 / 256x256 as a
controlled function of M, N, K. Goal: understand WHY larger tiles are faster
(MFMA-op amortization of fixed per-K-iter overhead), and derive the functional
form for the cost model -- NOT fit a constant.

Method: measure on LARGE shapes so every tile has many clean waves (M=N=16384 ->
64 waves for 128x128, 16 for 256x256; tail negligible), isolating compute
efficiency from wave-quantization. Convert time -> cycles using the measured
GPU clock. Report:
  * cycles/FLOP (steady-state compute efficiency) per tile
  * K-sweep: time = slope*K + intercept  ->  slope = per-K-iter cost,
    intercept = fill/epilogue overhead. Amortization = how the fixed overhead
    shrinks per-FLOP as the tile does more MFMA ops per K-iter.
"""
import importlib.util, io, contextlib, subprocess
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

# GPU clock (GHz) for cycle conversion
def gpu_ghz():
    try:
        out = subprocess.check_output(["rocm-smi", "--showsclk"], text=True)
        for ln in out.splitlines():
            if "Mhz" in ln or "MHz" in ln:
                import re
                m = re.search(r"(\d+)\s*[Mm]hz", ln)
                if m: return int(m.group(1)) / 1000.0
    except Exception:
        pass
    return 2.0  # gfx950 nominal fallback
GHZ = gpu_ghz()

TILES = [(128,128,64),(128,256,64),(256,128,64),(256,256,64),
         (128,128,32),(128,256,32),(256,256,32)]

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

def bench_tile(a,b,M,N,K,bm,bn,bk):
    try:
        c=mk(bm,bn,bk); kw=sel.config_to_kernel_kwargs(c)
        out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,bm)*triton.cdiv(N,bn),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),
            ACTIVATION="", **kw))
    except Exception:
        return None

def measure(M,N,K):
    a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
    res={}
    for (bm,bn,bk) in TILES:
        ms=bench_tile(a,b,M,N,K,bm,bn,bk); res[(bm,bn,bk)]=ms
    del a,b; torch.cuda.empty_cache()
    return res

def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12
def cyc_per_flop(M,N,K,ms):  # total device cycles / total FLOPs
    return (ms*1e-3*GHZ*1e9) / (2*M*N*K)

print(f"GPU clock ~{GHZ:.2f} GHz (cycle conversion)")

# ---- K-sweep at large clean M=N (isolates compute; many waves) ----
M=N=16384
Ks=[2048,4096,8192,16384]
print(f"\n=== K-sweep  M=N={M}  (clean waves; steady-state compute) ===")
print(f"{'tile':>13}  " + "  ".join(f"K{K}:TF/cpf" for K in Ks))
data={}
for K in Ks:
    data[K]=measure(M,N,K)
for t in TILES:
    row=[]
    for K in Ks:
        ms=data[K][t]
        row.append(f"{tflops(M,N,K,ms):.0f}/{cyc_per_flop(M,N,K,ms)*1e3:.3f}" if ms else "FAIL")
    print(f"{f'{t[0]}x{t[1]}x{t[2]}':>13}  " + "  ".join(f"{r:>13}" for r in row))
print("(TF = TFLOP/s ; cpf = milli-cycles/FLOP, lower=more efficient)")

# ---- per-K-iter cost via linear fit time = slope*K + intercept (BK64 tiles) ----
print(f"\n=== linear fit time(ms) = slope*K + intercept  (M=N={M}) ===")
print(f"{'tile':>13}{'slope(us/Kunit)':>17}{'intercept(us)':>15}{'fill_frac@K8192':>17}")
for t in [(128,128,64),(128,256,64),(256,256,64)]:
    xs=Ks; ys=[data[K][t]*1e3 for K in Ks]  # us
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    slope=(n*sxy-sx*sy)/(n*sxx-sx*sx); intercept=(sy-slope*sx)/n
    fill=intercept/(slope*8192+intercept)
    print(f"{f'{t[0]}x{t[1]}':>13}{slope*1000:>17.4f}{intercept:>15.2f}{fill:>16.1%}")

# ---- N-sweep (M=16384,K=8192): N-tile / padding effect ----
print(f"\n=== N-sweep  M=16384 K=8192 ===")
print(f"{'tile':>13}  " + "  ".join(f"N{N}" for N in [4096,8192,16384,32768]))
for N2 in [4096,8192,16384,32768]:
    pass
ndata={N2:measure(16384,N2,8192) for N2 in [4096,8192,16384,32768]}
for t in [(128,128,64),(128,256,64),(256,256,64)]:
    row=[f"{tflops(16384,N2,8192,ndata[N2][t]):.0f}" if ndata[N2][t] else "F" for N2 in [4096,8192,16384,32768]]
    print(f"{f'{t[0]}x{t[1]}':>13}  " + "  ".join(f"{r:>6}" for r in row) + "  (TFLOP/s)")
