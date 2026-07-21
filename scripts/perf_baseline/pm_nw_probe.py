from triton._C.libtriton import amd
pm=amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel
hw=pm.HardwareInfo.get(sel.current_amd_arch())
print("HW fields:", [a for a in dir(hw) if not a.startswith("_")])
prob=pm.GemmProblem(8192,2112,7168,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
e0=pm.estimate_perf(prob,pm.generate_candidates(prob,hw)[0],hw)
print("EST fields:", [a for a in dir(e0) if not a.startswith("_")])
print(f"{'tile ns W':>18}{'ctas/CU':>9}{'occ':>6}{'wav/simd':>9}{'vgpr':>6}{'pred':>7}")
for (M,N,K,BM,BN) in [(8192,2112,7168,256,256),(16384,256,7168,128,128),(4096,24576,1536,128,256)]:
    p=pm.GemmProblem(M,N,K,pm.ElemKind.FP16,pm.ElemKind.FP16,pm.ElemKind.FP32,16,16,32)
    print(f"--- {M}x{N}x{K} BM{BM}BN{BN} ---")
    for c in pm.generate_candidates(p,hw):
        if c.block_m==BM and c.block_n==BN and c.block_k==32 and c.num_stages==3:
            e=pm.estimate_perf(p,c,hw)
            wps=getattr(e,"waves_per_simd",getattr(e,"wavesPerSimd",-1))
            print(f"{f'{BM}x{BN}x32 s3 W{c.num_warps}':>18}{e.ctas_per_cu:>9}{e.occupancy:>6.2f}{wps:>9}{e.vgpr_count:>6}{e.predicted_tflops:>7.0f}")
