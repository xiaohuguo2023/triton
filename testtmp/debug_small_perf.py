"""
Debug: why does same config (32x32x64, nW=8, nS=2) give different TFLOPS
between autotune and PerfModel for small M/N/K?
"""
import torch, triton, triton.language as tl
from triton.backends.amd.amd_gemm_selector import pick_gemm_config, current_amd_arch

DEVICE = triton.runtime.driver.active.get_active_torch_device()
arch = current_amd_arch()

@triton.jit
def matmul_kernel_test(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, matrix_instr_nonkdim: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k*BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm*offs_cm[:, None] + stride_cn*offs_cn[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def run_kernel(a, b, bm, bn, bk, nw, ns, mfma, group_size_m):
    M, K = a.shape; K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_kernel_test[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=group_size_m, matrix_instr_nonkdim=mfma,
        num_warps=nw, num_stages=ns)
    return c

print(f"\nFixed config (32x32x64, nW=8, nS=2, mfma=16) - varying GROUP_SIZE_M:")
print(f"{'M':>5} {'GROUP_SIZE_M':>12}  {'TFLOPS':>8}")
print("-" * 35)
for M in [256, 512, 1024, 2048]:
    N = K = M
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    run_kernel(a, b, 32, 32, 64, 8, 2, 16, 8)  # warmup
    torch.cuda.synchronize()
    for gsm in [1, 4, 6, 8]:
        ms = triton.testing.do_bench(lambda: run_kernel(a, b, 32, 32, 64, 8, 2, 16, gsm))
        tflops = 2*M*N*K*1e-12 / (ms*1e-3)
        print(f"{M:>5} {gsm:>12}  {tflops:>8.2f}")
    print()

print(f"\nBK=32 vs BK=64 comparison (64x64 tile, M=1024):")
M = N = K = 1024
a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
for bk in [32, 64]:
    for gsm in [4, 8]:
        run_kernel(a, b, 64, 64, bk, 8, 2, 16, gsm)  # warmup
        ms = triton.testing.do_bench(lambda: run_kernel(a, b, 64, 64, bk, 8, 2, 16, gsm))
        tflops = 2*M*N*K*1e-12 / (ms*1e-3)
        print(f"  BK={bk:2d} GROUP_SIZE_M={gsm}: {tflops:.2f} TFLOPS")
