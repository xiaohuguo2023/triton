"""Validate the tie-break fix WITHOUT rebuilding libtriton: re-rank candidates in
Python with the OLD vs NEW dense tie-break, measure both top-1 picks + autotune.

OLD dense tie-break: predicted -> AI -> blockK(larger) -> numWarps(larger) -> numStages(fewer)
NEW dense tie-break: predicted -> AI -> numStages(MORE) -> blockK(larger) -> numWarps(larger)
  (mirrors the MX path's 'prefer deeper pipeline', ordered before blockK so the
   coupled smaller-BK/deeper-stage winner isn't lost to prefer-larger-BK.)

Losers should recover toward/above autotune; winners must not regress.
"""
import functools, importlib.util, io, contextlib
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

# losers (PM/auto<0.9) + winners (should stay ahead) from perf-baseline-all
LOSERS = [
    (4096, 16384,  5120), (4096, 32768, 5120), (16384, 8192, 28672),
    (4096,  7168, 16384), (4096,  4096,  4096),
]
WINNERS = [
    (4, 128, 2880), (256, 256, 7168), (64, 64, 8192),   # LARGE_K / skinny (PM wins big)
    (16, 24576, 1536), (8192, 5120, 5120),              # LARGE_N / LARGE
]

def bench_ms(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts=[]
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def tflops(M,N,K,ms): return 2*M*N*K/(ms*1e-3)/1e12

def bench_cfg(a,b,M,N,K,c):
    try:
        kw = sel.config_to_kernel_kwargs(c)
        out = torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),
            ACTIVATION="", **kw))
    except Exception:
        return None

def cmp_old(A,B):
    ca,ea=A; cb,eb=B
    if ea.is_valid!=eb.is_valid: return -1 if ea.is_valid>eb.is_valid else 1
    if abs(ea.predicted_tflops-eb.predicted_tflops)>1e-3:
        return -1 if ea.predicted_tflops>eb.predicted_tflops else 1
    if abs(ea.arithmetic_intensity-eb.arithmetic_intensity)>1e-3:
        return -1 if ea.arithmetic_intensity>eb.arithmetic_intensity else 1
    if ca.block_k!=cb.block_k: return -1 if ca.block_k>cb.block_k else 1
    if ca.num_warps!=cb.num_warps: return -1 if ca.num_warps>cb.num_warps else 1
    if ca.num_stages!=cb.num_stages: return -1 if ca.num_stages<cb.num_stages else 1
    return 0

def cmp_new(A,B):
    ca,ea=A; cb,eb=B
    if ea.is_valid!=eb.is_valid: return -1 if ea.is_valid>eb.is_valid else 1
    if abs(ea.predicted_tflops-eb.predicted_tflops)>1e-3:
        return -1 if ea.predicted_tflops>eb.predicted_tflops else 1
    if abs(ea.arithmetic_intensity-eb.arithmetic_intensity)>1e-3:
        return -1 if ea.arithmetic_intensity>eb.arithmetic_intensity else 1
    if ca.num_stages!=cb.num_stages: return -1 if ca.num_stages>cb.num_stages else 1  # MOVED UP, prefer MORE
    if ca.block_k!=cb.block_k: return -1 if ca.block_k>cb.block_k else 1
    if ca.num_warps!=cb.num_warps: return -1 if ca.num_warps>cb.num_warps else 1
    return 0

def top1(prob, hw, cands, cmp):
    scored=[(c, pm.estimate_perf(prob,c,hw)) for c in cands]
    scored.sort(key=functools.cmp_to_key(cmp))
    return scored[0][0]

def run(tag, shapes):
    hw = pm.HardwareInfo.get(arch)
    print(f"\n===== {tag} =====")
    print(f"{'M':>6}{'N':>7}{'K':>7}  {'old':>6}{'new':>6}{'auto':>6}  {'old/au':>7}{'new/au':>7}  new_cfg")
    for (M,N,K) in shapes:
        prob = pm.GemmProblem(M,N,K, pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
        cands = pm.generate_candidates(prob, hw, kernel_type=pm.KernelType.Standard)
        co = top1(prob,hw,cands,cmp_old); cn = top1(prob,hw,cands,cmp_new)
        a = torch.randn((M,K),device="cuda",dtype=torch.float16)
        b = torch.randn((K,N),device="cuda",dtype=torch.float16)
        mo=bench_cfg(a,b,M,N,K,co); mn=bench_cfg(a,b,M,N,K,cn)
        _=tut.matmul(a,b); ma=bench_ms(lambda: tut.matmul(a,b))
        to,tn,ta = (tflops(M,N,K,x) if x else 0 for x in (mo,mn,ma))
        ncfg=f"BK{cn.block_k}S{cn.num_stages}W{cn.num_warps}"
        print(f"{M:>6}{N:>7}{K:>7}  {to:>6.0f}{tn:>6.0f}{ta:>6.0f}  {to/ta if ta else 0:>7.3f}{tn/ta if ta else 0:>7.3f}  {ncfg}")
        del a,b; torch.cuda.empty_cache()

run("LOSERS", LOSERS)
run("WINNERS", WINNERS)
