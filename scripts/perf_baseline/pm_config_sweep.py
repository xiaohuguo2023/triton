"""Full config sweep for a fixed shape+BM×BN: every (BK,ns,warps,async,bypass_lds)
variant, measured vs predicted TFLOPS. Reveals BK×num_stages×async coupling."""
import sys, importlib.util, io, contextlib
import torch, triton
from triton.backends.amd import amd_gemm_selector as sel
from triton._C.libtriton import amd
pm=amd.perf_model
TUT="/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec=importlib.util.spec_from_file_location("tut03",TUT); tut=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass
arch=sel.current_amd_arch()
M,N,K,BM,BN=(int(x) for x in sys.argv[1:6])
hw=pm.HardwareInfo.get(arch)
prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
from torch.profiler import profile as _profile, ProfilerActivity as _PA
def bench_ms(fn,w=15,r=60):
    # profiler device-time (excludes launch overhead) — required for sub-ms kernels
    for _ in range(w): fn()
    torch.cuda.synchronize()
    with _profile(activities=[_PA.CUDA]) as prof:
        for _ in range(r): fn()
        torch.cuda.synchronize()
    tot=0.0
    for ev in prof.key_averages(): tot += getattr(ev,"self_device_time_total",0.0) or 0.0
    return (tot/r)/1000.0
def tflops(ms): return 2*M*N*K/(ms*1e-3)/1e12
a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
def bench_cfg(c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception as ex: return None
print(f"{M}x{N}x{K}  BM{BM}xBN{BN}")
print(f"{'BK':>4}{'ns':>3}{'W':>3}{'async':>6}{'byp':>4}{'meas':>7}{'pred':>7}{'cb':>3}{'effTile':>9}")
rows=[c for c in cands if c.block_m==BM and c.block_n==BN]
rows.sort(key=lambda c:(c.block_k,c.num_stages,c.num_warps))
best=(0,None)
for c in rows:
    e=pm.estimate_perf(prob,c,hw); ms=bench_cfg(c); mt=tflops(ms) if ms else 0
    if mt>best[0]: best=(mt,c)
    print(f"{c.block_k:>4}{c.num_stages:>3}{c.num_warps:>3}{str(c.use_async_copy):>6}{str(c.bypass_lds):>4}{mt:>7.0f}{e.predicted_tflops:>7.0f}{int(e.is_compute_bound):>3}{e.effective_tile_cycles:>9.0f}")
mc=best[1]; pmpick=max(rows,key=lambda c:pm.estimate_perf(prob,c,hw).predicted_tflops)
print(f"MEAS-BEST: BK{mc.block_k} ns{mc.num_stages} W{mc.num_warps} async{mc.use_async_copy} = {best[0]:.0f}")
print(f"PM-PICK:   BK{pmpick.block_k} ns{pmpick.num_stages} W{pmpick.num_warps} async{pmpick.use_async_copy}")
