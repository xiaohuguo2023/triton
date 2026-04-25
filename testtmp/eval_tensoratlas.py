"""
TensorAtlas Dataset Evaluation: PerfModel vs Tuned Configs
===========================================================

Compares our analytical PerfModel config selection against the tuned configs
in TensorAtlas's *_tuned.yaml files. Runs both configs on the current GPU
and reports TFLOPS, config match rate, and summary statistics.

Usage
-----
  python testtmp/eval_tensoratlas.py [--dataset <name>] [--top-k <k>] [--no-bench]

  --dataset: one of llama3_mlp, deepseek_r1, qwen3_235b_a22b, gpt_oss_120b,
             llama4_maverick  (default: qwen3_235b_a22b)
  --top-k:   how many top PerfModel configs to benchmark (default: 1)
  --no-bench: skip actual GPU benchmarking, show config comparison only

Notes
-----
- Tuned configs use stream-K (NUM_SMS, CHUNK_SIZE) which PerfModel doesn't support.
  We run the tuned kernel WITHOUT stream-K (standard tiled GEMM) for fair comparison.
- TFLOPS in the YAML were measured on gfx942 (MI300X). We re-benchmark on the
  current GPU (gfx950) for both tuned and PerfModel configs.
- Very small M (≤16) may show poor PerfModel performance since our model doesn't
  yet optimise for tiny batch sizes where kernel launch overhead dominates.
"""

import argparse, sys, yaml, torch, triton, time
import triton.language as tl
from triton.backends.amd.amd_gemm_selector import (
    pick_gemm_config, config_to_kernel_kwargs, current_amd_arch,
)

DATASETS_DIR = "/home/work/TensorAtlas/datasets"
DEVICE = triton.runtime.driver.active.get_active_torch_device()
ARCH   = current_amd_arch()

# ── Kernel ────────────────────────────────────────────────────────────────────
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

def run_kernel(a, b, bm, bn, bk, nw, ns, mfma, gsm):
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

def bench(fn, M, N, K, reps=3):
    ms = triton.testing.do_bench(fn, rep=reps)
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)

def is_power_of_two(n):
    return n > 0 and (n & (n-1)) == 0

def load_tuned(dataset):
    # Handle naming inconsistency: gpt_oss_120b uses _shapes_tuned suffix
    for suffix in ["_tuned.yaml", "_shapes_tuned.yaml"]:
        path = f"{DATASETS_DIR}/{dataset}{suffix}"
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No tuned YAML found for dataset '{dataset}'")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="qwen3_235b_a22b",
                   choices=["llama3_mlp", "deepseek_r1", "qwen3_235b_a22b",
                            "gpt_oss_120b", "llama4_maverick"])
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--max-shapes", type=int, default=0,
                   help="Limit number of shapes to evaluate (0=all)")
    args = p.parse_args()

    entries = load_tuned(args.dataset)
    print(f"\nDataset: {args.dataset}  ({len(entries)} shapes)")
    print(f"GPU: {ARCH}  |  PerfModel top-k: {args.top_k}")
    print(f"{'benchmark' if not args.no_bench else 'config-compare only'}")
    print("=" * 110)

    fmt = (f"{'M':>6} {'N':>6} {'K':>6}  "
           f"{'Tuned config':35s}  {'PerfModel config':35s}  "
           f"{'Tuned':>8}  {'PerfModel':>9}  {'Ratio':>6}  {'Match':>5}")
    print(fmt)
    print("-" * 110)

    total = skipped = matched_bm = matched_bn = matched_bk = 0
    tflops_tuned_sum = tflops_pm_sum = 0.0
    ratios = []

    shapes = entries[:args.max_shapes] if args.max_shapes else entries

    for entry in shapes:
        M, N, K = entry['M'], entry['N'], entry['K']
        tuned_bm = entry['BLOCK_SIZE_M']
        tuned_bn = entry['BLOCK_SIZE_N']
        tuned_bk = entry['BLOCK_SIZE_K']
        tuned_nw = entry['num_warps']
        tuned_ns = entry['num_stages']
        tuned_mfma = entry['matrix_instr_nonkdim']
        tuned_gsm  = entry.get('GROUP_SIZE_M', 8)
        tuned_tflops_ref = float(entry['TFLOPS'])  # reference from YAML (gfx942)

        # Skip shapes where tuned config uses non-power-of-2 BLOCK_K (rare)
        if not is_power_of_two(tuned_bk):
            skipped += 1; continue

        # PerfModel config selection
        pm_cfgs = pick_gemm_config(M, N, K, 'fp16', ARCH, top_k=args.top_k)
        if not pm_cfgs:
            skipped += 1; continue
        pm_cfg = pm_cfgs[0]
        pm_kw  = config_to_kernel_kwargs(pm_cfg)

        total += 1

        # Config match
        bm_match = tuned_bm == pm_cfg.block_m
        bn_match = tuned_bn == pm_cfg.block_n
        bk_match = tuned_bk == pm_cfg.block_k
        if bm_match: matched_bm += 1
        if bn_match: matched_bn += 1
        if bk_match: matched_bk += 1
        tile_match = bm_match and bn_match

        tuned_str = (f"BM={tuned_bm:3d} BN={tuned_bn:3d} BK={tuned_bk:3d} "
                     f"nW={tuned_nw} nS={tuned_ns}")
        pm_str    = (f"BM={pm_cfg.block_m:3d} BN={pm_cfg.block_n:3d} "
                     f"BK={pm_cfg.block_k:3d} nW={pm_cfg.num_warps} nS={pm_cfg.num_stages}")
        match_str = "✓BK" if (bm_match and bn_match and bk_match) else \
                    "✓MN" if tile_match else ""

        if args.no_bench:
            print(f"{M:>6} {N:>6} {K:>6}  {tuned_str:35s}  {pm_str:35s}  "
                  f"{'ref:'+f'{tuned_tflops_ref:.0f}':>8}  {'N/A':>9}  {'N/A':>6}  {match_str:>5}")
            continue

        # Benchmark both configs on current GPU
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        try:
            # Warm up
            run_kernel(a, b, tuned_bm, tuned_bn, tuned_bk, tuned_nw, tuned_ns,
                       tuned_mfma, tuned_gsm)
            run_kernel(a, b, pm_cfg.block_m, pm_cfg.block_n, pm_cfg.block_k,
                       pm_cfg.num_warps, pm_cfg.num_stages,
                       pm_cfg.mfma_non_k_dim, pm_cfg.group_size_m)
            torch.cuda.synchronize()

            tf_tuned = bench(lambda: run_kernel(a, b, tuned_bm, tuned_bn, tuned_bk,
                                                 tuned_nw, tuned_ns, tuned_mfma, tuned_gsm),
                              M, N, K)
            tf_pm    = bench(lambda: run_kernel(a, b, pm_cfg.block_m, pm_cfg.block_n,
                                                 pm_cfg.block_k, pm_cfg.num_warps,
                                                 pm_cfg.num_stages, pm_cfg.mfma_non_k_dim,
                                                 pm_cfg.group_size_m),
                              M, N, K)

            ratio = tf_pm / tf_tuned if tf_tuned > 0 else 0
            tflops_tuned_sum += tf_tuned
            tflops_pm_sum    += tf_pm
            ratios.append(ratio)

            print(f"{M:>6} {N:>6} {K:>6}  {tuned_str:35s}  {pm_str:35s}  "
                  f"{tf_tuned:>8.1f}  {tf_pm:>9.1f}  {ratio:>6.2f}  {match_str:>5}")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6}  {tuned_str:35s}  {pm_str:35s}  "
                  f"  ERROR: {str(e)[:40]}")

    print("=" * 110)
    print(f"\nSummary ({total} shapes evaluated, {skipped} skipped):")
    print(f"  Config match rate:  BM={matched_bm/total:.1%}  "
          f"BN={matched_bn/total:.1%}  BK={matched_bk/total:.1%}  "
          f"MN-tile={sum(1 for _ in range(total) if matched_bm > 0)/total:.0%}")

    if ratios:
        import statistics
        print(f"  TFLOPS ratio (PM/Tuned):  "
              f"median={statistics.median(ratios):.2f}  "
              f"mean={statistics.mean(ratios):.2f}  "
              f"min={min(ratios):.2f}  max={max(ratios):.2f}")
        pct_within_10 = sum(1 for r in ratios if r >= 0.90) / len(ratios)
        pct_within_20 = sum(1 for r in ratios if r >= 0.80) / len(ratios)
        pct_better    = sum(1 for r in ratios if r >= 1.00) / len(ratios)
        print(f"  Within 10% of tuned: {pct_within_10:.1%}")
        print(f"  Within 20% of tuned: {pct_within_20:.1%}")
        print(f"  Better than tuned:   {pct_better:.1%}")
        print(f"\n  Total TFLOPS: Tuned={tflops_tuned_sum:.0f}  "
              f"PerfModel={tflops_pm_sum:.0f}  "
              f"Overall ratio={tflops_pm_sum/tflops_tuned_sum:.2f}")

if __name__ == "__main__":
    main()
