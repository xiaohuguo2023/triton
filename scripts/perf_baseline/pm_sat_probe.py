import sys
from triton._C.libtriton import amd
pm=amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel
M,N,K=(int(x) for x in sys.argv[1:4])
want=[(int(x.split('x')[0]),int(x.split('x')[1])) for x in sys.argv[4:]]
hw=pm.HardwareInfo.get(sel.current_amd_arch())
prob=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
cands=pm.generate_candidates(prob,hw,kernel_type=pm.KernelType.Standard)
seen=set()
for c in cands:
    if (c.block_m,c.block_n) in want and c.num_warps==8 and c.num_stages==3 and c.block_k==64 and (c.block_m,c.block_n) not in seen:
        seen.add((c.block_m,c.block_n)); pm.estimate_perf(prob,c,hw)
