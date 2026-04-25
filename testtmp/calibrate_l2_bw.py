"""
L2 Bandwidth Calibration Experiment for gfx950 PerfModel
=========================================================

Goal
----
Determine the correct value of `peakL2BwBytesPerCycle` for gfx950 (MI355X)
to use in PerfModel's `HardwareInfo`. This value controls how much benefit
the model assigns to L2 cache hits when ranking GEMM configurations.

Methodology
-----------
The key insight: for a fixed problem (M=N=K, BM=BN fixed), varying BK changes
the L2 working set per K-block but NOT the total DRAM traffic (both have the
same total FLOPs and total bytes). The performance difference between BK values
must come from L2 cache effects:

  - Small BK: many K-iterations, each loading a small A/B slice. More passes
    over the same data → better L2 temporal reuse for A (column panels)
    but more loop overhead.

  - Large BK: fewer K-iterations, each loading a large A/B slice. Less L2
    reuse opportunity but lower loop overhead.

The transition point where performance flattens (or drops) as BK increases
tells us when the A/B tiles stop fitting in L2.

Calibration procedure
---------------------
1. Fix M=N=K=1024, BM=BN=64, nW=8, nS=2, mfma=16 on gfx950
2. Sweep BK ∈ {16, 32, 64, 96, 128, 192, 256}
3. Measure actual TFLOPS for each BK
4. Compute L2 tile sizes from the model for each BK
5. Find the BK where performance drops — this is when A/B tiles exceed L2
6. Use the L2 capacity and tile sizes to back-calculate L2 bandwidth:
   L2_BW = (L2_hit_traffic) / (measured_cycles - DRAM_cycles)
   where DRAM_cycles = DRAM_traffic / peakDRAMBwPerCU

Expected outcome
----------------
- For small BK (fits in L2): TFLOPS should be relatively constant (memory
  overhead hidden by L2 hits)
- For large BK (exceeds L2): TFLOPS drops as more traffic goes to DRAM
- The crossover BK gives us the L2 working set size, from which we infer
  the L2 bandwidth needed to match observed performance

Hardware: gfx950 (AMD Instinct MI355X)
  - 256 CUs, 8 XCDs (32 CUs per XCD)
  - DRAM BW: ~7.2 TB/s → 3000 bytes/cycle at 2.4 GHz
  - L2 per XCD: ~32 MB (estimated; verify from AMD docs)
  - L2 BW: unknown → this experiment determines it

How to use the results
----------------------
After running this experiment:
1. Plot TFLOPS vs BK
2. Find the "knee" where performance drops
3. At the knee BK=BK_knee: A tile = BM*BK*2 bytes, B tile = BN*BK*2 bytes
   → L2 capacity per XCD ≈ (l2_m * A_tile + l2_n * B_tile) at the knee
4. For BK < BK_knee (L2 hits): measure TFLOPS_l2
   For BK > BK_knee (DRAM bound): measure TFLOPS_dram
5. L2 hit rate at knee ≈ estimate_l2_hit_rate(BK_knee) from model
6. Effective BW at BK < knee = total_bytes / (total_bytes/TFLOPS_l2 * peak_flops)
7. L2_BW = solve: TFLOPS_l2 matches model with L2_BW as free parameter

Usage
-----
  python testtmp/calibrate_l2_bw.py

Requirements: Triton installed with AMD perf_model bindings (pip install -e .)
"""

import torch
import triton
import triton.language as tl
from triton._C.libtriton.amd import perf_model as pm

DEVICE = triton.runtime.driver.active.get_active_torch_device()
ARCH   = triton.runtime.driver.active.get_current_target().arch
HW     = pm.HardwareInfo.get(ARCH)

print(f"Device arch: {ARCH}")
print(f"numCUs: {HW.num_cus}, clock: {HW.clock_mhz} MHz")
print(f"Peak DRAM BW: {HW.peak_mem_bw_bytes_per_cycle:.0f} bytes/cycle")
print(f"DRAM BW at clock: {HW.peak_mem_bw_bytes_per_cycle * HW.clock_mhz / 1e6:.1f} TB/s")
print()

# ── Kernel ────────────────────────────────────────────────────────────────────
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, matrix_instr_nonkdim: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k*BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm*offs_cm[:, None] + stride_cn*offs_cn[None, :]
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def run(a, b, bm, bn, bk, nw, ns, mfma, gsm):
    M, K = a.shape; K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GROUP_SIZE_M=gsm, matrix_instr_nonkdim=mfma,
        num_warps=nw, num_stages=ns)
    return c

# ── Experiment parameters ─────────────────────────────────────────────────────
BM, BN  = 64, 64
NW, NS  = 8, 2
MFMA    = 16
GSM     = 8   # fixed GROUP_SIZE_M for fair comparison

# BK values to sweep — must be powers of 2 (Triton tl.arange constraint)
# and multiples of MFMA kDim (32 for gfx950 16x16x32)
BK_VALUES = [32, 64, 128, 256]

# Problem sizes to sweep
M_VALUES = [512, 1024, 2048, 4096]

print("=" * 80)
print("L2 Bandwidth Calibration: TFLOPS vs BK for fixed BM=BN=64 on gfx950")
print("=" * 80)
print(f"{'M':>6}  " + "  ".join(f"BK={bk:3d}" for bk in BK_VALUES))
print("-" * 80)

results = {}
for M in M_VALUES:
    N = K = M
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    row = {}
    for bk in BK_VALUES:
        if bk > K:
            row[bk] = float('nan')
            continue
        # warmup
        run(a, b, BM, BN, bk, NW, NS, MFMA, GSM)
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(lambda: run(a, b, BM, BN, bk, NW, NS, MFMA, GSM))
        tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
        row[bk] = tflops

    results[M] = row
    vals = "  ".join(f"{row[bk]:7.1f}" for bk in BK_VALUES)
    print(f"{M:>6}  {vals}")

print()
print("=" * 80)
print("Analysis: L2 working set per XCD vs L2 capacity")
print("=" * 80)
print(f"\nFor BM=BN={BM}, numXCDs=8, l2SizeBytes={HW.l2_per_cu/(1<<20) if hasattr(HW,'l2_per_cu') else '?'} MB per XCD")
print(f"\nA tile size: BM*BK*2 bytes  (per K-block)")
print(f"B tile size: BN*BK*2 bytes  (per K-block)")
print()

prob_1024 = pm.GemmProblem(1024, 1024, 1024,
    pm.ElemKind.FP16, pm.ElemKind.FP16, pm.ElemKind.FP32, 16, 16, 32)

print(f"{'BK':>5}  {'A_tile_KB':>10}  {'B_tile_KB':>10}  {'total_KB':>10}  "
      f"{'L2_hit_rate':>12}  {'BK=64 ratio':>12}")
bk64_tflops = {M: results[M].get(64, float('nan')) for M in M_VALUES}

for bk in BK_VALUES:
    cfg = pm.TritonGemmConfig()
    cfg.block_m=BM; cfg.block_n=BN; cfg.block_k=bk; cfg.num_warps=NW
    cfg.num_stages=NS; cfg.mfma_non_k_dim=MFMA; cfg.use_async_copy=True
    cfg.bypass_lds=False; cfg.k_pack=1; cfg.group_size_m=GSM

    # Estimate L2 hit rate for M=1024
    try:
        est = pm.estimate_perf(prob_1024, cfg, HW)
    except Exception:
        est = None

    a_tile_kb = BM * bk * 2 / 1024
    b_tile_kb = BN * bk * 2 / 1024
    total_kb  = a_tile_kb + b_tile_kb

    # Ratio vs BK=64 for M=1024
    ratio_1024 = (results[1024].get(bk, float('nan')) /
                  results[1024].get(64, float('nan'))
                  if results[1024].get(64, 0) > 0 else float('nan'))

    print(f"{bk:>5}  {a_tile_kb:>10.1f}  {b_tile_kb:>10.1f}  {total_kb:>10.1f}  "
          f"{'(see model)':>12}  {ratio_1024:>12.3f}")

print()
print("=" * 80)
print("Calibration guidance")
print("=" * 80)
print("""
To calibrate peakL2BwBytesPerCycle:

1. Find BK_knee = smallest BK where TFLOPS(BK) / TFLOPS(BK=64) drops below 0.95
   This is where the A/B tiles start missing L2.

2. At BK_knee: A_working_set = l2_m * BM * BK_knee * 2 bytes per XCD
               B_working_set = l2_n * BN * BK_knee * 2 bytes per XCD
   These should be close to l2SizeBytes (32 MB per XCD on gfx950).

3. For BK < BK_knee (L2-bound), measure TFLOPS_l2.
   For BK > BK_knee (DRAM-bound), measure TFLOPS_dram.

4. L2 hit rate at BK_small = estimate_l2_hit_rate(l2_m, l2_n, BM, BK, BN)
   from the model.

5. Effective DRAM traffic at BK_small = tileBytesAB * (1 - L2_hit_rate) * numKIter
   Effective L2 traffic = tileBytesAB * L2_hit_rate * numKIter

6. totalSeconds_l2 = totalFlops / (TFLOPS_l2 * 1e12)
   memSeconds_l2 = totalSeconds_l2 - compute_seconds  (subtract compute time)
   dramSeconds = DRAM_traffic / (peakDRAMBw * numCUs)
   l2Seconds = memSeconds_l2 - dramSeconds
   peakL2BwPerCU = L2_traffic_per_CU / l2Seconds * (1 / clockHz)

7. Set hw.peakL2BwBytesPerCycle = peakL2BwPerCU * numCUs / clockHz * clockHz
   (= L2_bw_bytes_per_second / clockHz)

Typical values expected:
  gfx950 L2 BW ≈ 30-60 TB/s → 30e12/2.4e9 = 12500 to 25000 bytes/cycle
""")
