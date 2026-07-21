"""For a shape + fixed BM×BN, sweep BLOCK_K: measured TFLOPS vs model predicted,
to expose the BK tie-break divergence. Args: M N K BM BN"""
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
def bench_ms(fn,w=15,r=50):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(r):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
def tflops(ms): return 2*M*N*K/(ms*1e-3)/1e12
a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
def bench_cfg(c):
    try:
        kw=sel.config_to_kernel_kwargs(c); out=torch.empty((M,N),device="cuda",dtype=torch.float16)
        grid=(triton.cdiv(M,c.block_m)*triton.cdiv(N,c.block_n),)
        return bench_ms(lambda: tut.matmul_kernel_amd[grid](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out.stride(0),out.stride(1),ACTIVATION="",**kw))
    except Exception as ex: return None
# best config per BK for this BM×BN (best over warps/stages by measured)
print(f"{M}x{N}x{K}  BM{BM}xBN{BN}")
print(f"{'BK':>5}{'W':>3}{'S':>3}{'meas_TF':>9}{'pred_TF':>9}{'cb':>3}{'effTile':>10}{'numWave':>8}")
bybk={}
for c in cands:
    if c.block_m==BM and c.block_n==BN:
        bybk.setdefault(c.block_k,[]).append(c)
for bk in sorted(bybk):
    # pick the warp/stage variant the model ranks best for this BK
    best=max(bybk[bk], key=lambda c: pm.estimate_perf(prob,c,hw).predicted_tflops)
    e=pm.estimate_perf(prob,best,hw)
    ms=bench_cfg(best)
    mt=tflops(ms) if ms else 0
    print(f"{bk:>5}{best.num_warps:>3}{best.num_stages:>3}{mt:>9.0f}{e.predicted_tflops:>9.0f}{int(e.is_compute_bound):>3}{e.effective_tile_cycles:>10.0f}{e.num_waves:>8}")
