"""Why does PerfModel under-rank the larger N-tile (BN256) vs BN128 on MEDIUM
shapes? Dump the estimate for the BM128/256 x {BN} x {BK} W8 S3 family."""
import sys
from triton._C.libtriton import amd
pm = amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel

M, N, K = (int(x) for x in (sys.argv[1:4] or (4096, 5120, 5120)))
hw = pm.HardwareInfo.get(sel.current_amd_arch())
prob = pm.GemmProblem(M, N, K, pm.ElemKind.FP16, pm.ElemKind.FP16, pm.ElemKind.FP32, 16, 16, 32)
cands = pm.generate_candidates(prob, hw, kernel_type=pm.KernelType.Standard)
fields = ["predicted_tflops", "num_waves", "wave_efficiency", "total_output_tiles",
          "occupancy", "ctas_per_cu", "effective_tile_cycles", "is_compute_bound",
          "arithmetic_intensity", "vgpr_count", "lds_bytes"]
fam = [c for c in cands if c.num_warps == 8 and c.num_stages == 3
       and c.block_m in (128, 256) and c.block_k in (32, 64)]
rows = [(c, pm.estimate_perf(prob, c, hw)) for c in fam]
rows.sort(key=lambda r: -r[1].predicted_tflops)
print(f"shape {M}x{N}x{K}")
print(f"{'BMxBN':>9}{'BK':>4}  " + "  ".join(f"{f[:13]:>13}" for f in fields))
for c, e in rows:
    vals = []
    for f in fields:
        v = getattr(e, f)
        vals.append(f"{v:13.4g}" if isinstance(v, float) else f"{str(v):>13}")
    print(f"{str(c.block_m)+'x'+str(c.block_n):>9}{c.block_k:>4}  " + "  ".join(vals))
