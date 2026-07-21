import sys
from triton._C.libtriton import amd
pm=amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel
M,N,K=(int(x) for x in sys.argv[1:4])
want=[(64,64),(128,128),(128,256),(256,256),(64,256)]
hw=pm.HardwareInfo.get(sel.current_amd_arch())
prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
ff=["predicted_tflops","num_waves","wave_efficiency","total_output_tiles","effective_tile_cycles","compute_cycles","memory_cycles","is_compute_bound"]
print(f"{M}x{N}x{K}")
print(f"{'BMxBNxBKxWxS':>18}"+"".join(f"{n[:11]:>12}" for n in ff))
best={}
for c in cands:
    if (c.block_m,c.block_n) in want and c.num_warps==8 and c.num_stages==3 and c.block_k in (64,128):
        e=pm.estimate_perf(prob,c,hw)
        k=(c.block_m,c.block_n)
        if k not in best or e.predicted_tflops>best[k][1].predicted_tflops: best[k]=(c,e)
for k in want:
    if k not in best: continue
    c,e=best[k]; cells=[]
    for n in ff:
        v=getattr(e,n); cells.append(f"{v:12.4g}" if isinstance(v,float) else f"{str(v):>12}")
    print(f"{f'{c.block_m}x{c.block_n}x{c.block_k}x{c.num_warps}x{c.num_stages}':>18}"+"".join(cells))
