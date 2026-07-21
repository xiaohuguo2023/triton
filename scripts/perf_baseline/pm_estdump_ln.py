import sys
from triton._C.libtriton import amd
pm=amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel
M,N,K=(int(x) for x in (sys.argv[1:4] or (4,24576,1536)))
hw=pm.HardwareInfo.get(sel.current_amd_arch())
prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
ff=["predicted_tflops","num_waves","wave_efficiency","total_output_tiles","occupancy","effective_tile_cycles","is_compute_bound","arithmetic_intensity","vgpr_count"]
fam=[c for c in cands if c.block_m in (16,32) and c.block_n in (32,64,128,256) and c.num_warps==8 and c.num_stages==3 and c.block_k in (64,128)]
rows=[(c,pm.estimate_perf(prob,c,hw)) for c in fam]
rows.sort(key=lambda r:-r[1].predicted_tflops)
print(f"{M}x{N}x{K}")
print(f"{'BMxBNxBK':>13}"+"".join(f"{n[:12]:>13}" for n in ff))
for c,e in rows:
    cells=[]
    for n in ff:
        v=getattr(e,n)
        cells.append(f"{v:13.4g}" if isinstance(v,float) else f"{str(v):>13}")
    print(f"{f'{c.block_m}x{c.block_n}x{c.block_k}':>13}"+"".join(cells))
