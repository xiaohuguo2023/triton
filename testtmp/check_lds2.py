import torch, triton, triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, matrix_instr_nonkdim: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n; pid_n = pid % num_pid_n
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None]*stride_am + offs_k[None,:]*stride_ak
    b_ptrs = b_ptr + offs_k[:,None]*stride_bk + offs_bn[None,:]*stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm*offs_cm[:,None] + stride_cn*offs_cn[None,:]
    tl.store(c_ptrs, acc.to(tl.float16))

M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

print("BK=128, nS=2 LDS usage:")
for bm, bn in [(32,32), (64,64), (128,128), (256,128), (256,256)]:
    for bk, ns, mfma in [(128, 2, 16), (64, 2, 16)]:
        try:
            pgm = matmul_kernel.warmup(a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=8,
                matrix_instr_nonkdim=mfma, num_warps=8, num_stages=ns, grid=(1,))
            shared = pgm.metadata.shared
            status = "✅" if shared <= 163840 else "❌ OVERFLOW"
            print(f"  BM={bm:3d} BN={bn:3d} BK={bk:3d} nS={ns}: {shared:7d} bytes {status}")
        except Exception as e:
            print(f"  BM={bm:3d} BN={bn:3d} BK={bk:3d} nS={ns}: FAILED - {e}")
