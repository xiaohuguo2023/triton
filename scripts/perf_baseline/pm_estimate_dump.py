"""Dump the PerfEstimate breakdown for the BM256/BN256/W8 config family on a
large dense shape, to find which cost term mis-ranks BK64/S2 (PM top-1) above
BK32/S3 (the ~28%-faster true winner)."""
from triton._C.libtriton import amd
pm = amd.perf_model
from triton.backends.amd import amd_gemm_selector as sel

arch = sel.current_amd_arch()
hw = pm.HardwareInfo.get(arch)
prob = pm.GemmProblem(4096, 4096, 4096, pm.ElemKind.FP16, pm.ElemKind.FP16,
                      pm.ElemKind.FP32, 16, 16, 32)
cands = pm.generate_candidates(prob, hw, kernel_type=pm.KernelType.Standard)

def fields(e):
    return sorted(n for n in dir(e) if not n.startswith("_")
                  and not callable(getattr(e, n)))

fam = [c for c in cands if c.block_m == 256 and c.block_n == 256 and c.num_warps == 8]
print(f"{len(cands)} candidates, {len(fam)} in BM256/BN256/W8 family")
rows = [(c, pm.estimate_perf(prob, c, hw)) for c in fam]
if not rows:
    raise SystemExit("no BM256/BN256/W8 candidates")
ff = fields(rows[0][1])
print("fields:", ff)

def gv(e, n):
    try: return getattr(e, n)
    except Exception: return None

rows.sort(key=lambda r: -(gv(r[1], "predicted_tflops") or 0))
hdr = f"{'BK':>4}{'S':>3}{'wpe':>4}  " + "  ".join(f"{n[:16]:>16}" for n in ff)
print(hdr)
for c, e in rows:
    cells = []
    for n in ff:
        v = gv(e, n)
        cells.append(f"{v:16.4g}" if isinstance(v, float) else f"{str(v):>16}")
    wpe = getattr(c, "waves_per_eu", -1)
    print(f"{c.block_k:>4}{c.num_stages:>3}{wpe:>4}  " + "  ".join(cells))
