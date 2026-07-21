import sys
from triton._C.libtriton import amd
pm=amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel
M,N,K=(int(x) for x in sys.argv[1:4])
hw=pm.HardwareInfo.get(sel.current_amd_arch())
prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
ff=["predicted_tflops","num_waves","wave_efficiency","total_output_tiles","occupancy","effective_tile_cycles","compute_cycles","memory_cycles","is_compute_bound"]
rows=[(c,pm.estimate_perf(prob,c,hw)) for c in cands]
rows.sort(key=lambda r:-r[1].predicted_tflops)
print(f"{M}x{N}x{K}  (top-8 + notable)")
print(f"{'BMxBNxBKxWxS':>18}"+"".join(f"{n[:11]:>12}" for n in ff))
def show(c,e,tag=""):
    cells=[]
    for n in ff:
        v=getattr(e,n)
        cells.append(f"{v:12.4g}" if isinstance(v,float) else f"{str(v):>12}")
    print(f"{f'{c.block_m}x{c.block_n}x{c.block_k}x{c.num_warps}x{c.num_stages}':>18}"+"".join(cells)+tag)
for i,(c,e) in enumerate(rows[:8]): show(c,e,f"  #{i}")
for i,(c,e) in enumerate(rows):
    if c.block_m>=128 and c.block_n==256: show(c,e,f"  <-BM128BN256 #{i}"); break
for i,(c,e) in enumerate(rows):
    if c.block_m==32 and c.block_n==256: show(c,e,f"  <-BM32BN256 #{i}"); break
