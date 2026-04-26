"""
Test AITER tile sizes vs PerfModel on gfx950.

AITER uses tiny tiles with very large BK (e.g., 1×4×512) for small-M GEMM.
BM=1, BN=4 means each tile covers literally 1 row and 4 cols of output —
enabled by using scalar/vector operations rather than MFMA for very small M.

For M≤16 shapes where BM < mfmaNonKDim=16, Triton must use a smaller MFMA
or fallback to FMA. We test with matrix_instr_nonkdim=0 (auto) for AITER tiles.
"""

import torch, triton
import triton.language as tl
from triton.backends.amd.amd_gemm_selector import pick_gemm_config, current_amd_arch, config_to_kernel_kwargs

DEVICE = triton.runtime.driver.active.get_active_torch_device()
ARCH   = current_amd_arch()

@triton.jit
def matmul_k(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, matrix_instr_nonkdim: tl.constexpr,
    ACTIVATION: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    npig = GROUP_SIZE_M * num_pid_n
    gid = pid // npig; fpm = gid * GROUP_SIZE_M
    gsm = min(num_pid_m - fpm, GROUP_SIZE_M)
    pm = fpm + ((pid % npig) % gsm); pn = (pid % npig) // gsm
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)
    ap = a_ptr + rm[:, None]*stride_am + rk[None, :]*stride_ak
    bp = b_ptr + rk[:, None]*stride_bk + rn[None, :]*stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(ap, mask=rk[None, :] < K-k*BLOCK_K, other=0.0)
        b = tl.load(bp, mask=rk[:, None] < K-k*BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        ap += BLOCK_K * stride_ak; bp += BLOCK_K * stride_bk
    rc = pm*BLOCK_M + tl.arange(0, BLOCK_M)
    sc = pn*BLOCK_N + tl.arange(0, BLOCK_N)
    cp = c_ptr + stride_cm*rc[:, None] + stride_cn*sc[None, :]
    tl.store(cp, acc.to(tl.float16), mask=(rc[:, None] < M) & (sc[None, :] < N))

def run(a, b, bm, bn, bk, nw, ns, mfma, gsm):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_k[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GROUP_SIZE_M=gsm, matrix_instr_nonkdim=mfma, ACTIVATION='',
        num_warps=nw, num_stages=ns)
    return c

def bench(fn, M, N, K):
    ms = triton.testing.do_bench(fn, rep=5)
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)

# AITER tile configs from the table (literal BM, BN, BK)
AITER_CONFIGS = {
    # (M, N, K): (BM, BN, BK, nW, nS, mfma, gsm, NUM_WGs, TFLOPS_ref)
    (4,   128, 2880): (1,  4,  512, 4, 2, 0, 1, 128,  0.35),
    (8,   128, 2880): (1,  4,  512, 4, 2, 0, 1, 256,  0.65),
    (4,  5120, 2880): (32, 32, 512, 8, 2, 0, 1, 160,  8.66),
    (8,  5120, 2880): (32, 32, 512, 8, 2, 0, 1, 160, 17.37),
    (16, 5120, 2880): (32, 32, 512, 8, 2, 0, 1, 160, 33.59),
    (32, 5120, 2880): (32, 32, 512, 8, 2, 0, 1, 160, 63.45),
    (4,  2880, 4096): (16, 16, 512, 8, 2, 0, 1, 180,  7.88),
    (8,  2880, 4096): (16, 16, 512, 8, 2, 0, 1, 180, 15.97),
    (16, 2880, 4096): (16, 16, 512, 8, 2, 0, 1, 180, 31.54),
    (32, 2880, 4096): (32, 16, 512, 8, 2, 0, 1, 180, 46.08),
}

print(f"GPU: {ARCH}")
print(f"\n{'M':>5} {'N':>5} {'K':>5}  {'AITER config':25s}  {'AITER':>7}  {'PerfModel config':25s}  {'PM':>7}  {'Ratio':>6}")
print("-" * 105)

for (M, N, K), (bm, bn, bk, nw, ns, mfma, gsm, num_wgs, tflops_ref) in AITER_CONFIGS.items():
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    # Test AITER config
    aiter_str = f"BM={bm:2d} BN={bn:2d} BK={bk:3d} nW={nw} nS={ns}"
    try:
        run(a, b, bm, bn, bk, nw, ns, mfma, gsm)  # warmup
        tf_aiter = bench(lambda: run(a, b, bm, bn, bk, nw, ns, mfma, gsm), M, N, K)
    except Exception as e:
        tf_aiter = float('nan')
        aiter_str += f" ERR:{str(e)[:20]}"

    # Test PerfModel top-1 config
    cfgs = pick_gemm_config(M, N, K, 'fp16', ARCH, top_k=1)
    if cfgs:
        cfg = cfgs[0]
        kw = config_to_kernel_kwargs(cfg)
        pm_str = f"BM={cfg.block_m:2d} BN={cfg.block_n:2d} BK={cfg.block_k:3d} nW={cfg.num_warps} nS={cfg.num_stages}"
        try:
            run(a, b, cfg.block_m, cfg.block_n, cfg.block_k,
                cfg.num_warps, cfg.num_stages, cfg.mfma_non_k_dim, cfg.group_size_m)
            tf_pm = bench(lambda: run(a, b, cfg.block_m, cfg.block_n, cfg.block_k,
                                       cfg.num_warps, cfg.num_stages,
                                       cfg.mfma_non_k_dim, cfg.group_size_m), M, N, K)
        except Exception as e:
            tf_pm = float('nan')
    else:
        tf_pm = float('nan')
        pm_str = "no config"

    ratio = tf_pm / tf_aiter if tf_aiter > 0 and tf_pm > 0 else float('nan')
    tiles = (M+bm-1)//bm * ((N+bn-1)//bn)
    print(f"{M:>5} {N:>5} {K:>5}  {aiter_str:25s}  {tf_aiter:>7.2f}  {pm_str:25s}  {tf_pm:>7.2f}  {ratio:>6.2f}  (tiles={tiles})")

print()
print("Notes:")
print("- AITER config tile sizes are literal (BM=1 means 1 row, BN=4 means 4 cols)")
print("- mfma=0 means auto-select (Triton picks based on tile size)")
print("- NUM_WGs in AITER = outputTiles (no stream-K splitting)")
