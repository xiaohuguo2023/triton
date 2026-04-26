import torch, triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_k(a_ptr, b_ptr, c_ptr, M, N, K,
    sa0, sa1, sb0, sb1, sc0, sc1,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GSM: tl.constexpr, mfma: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M,BM); nn = tl.cdiv(N,BN)
    npig = GSM*nn; gid = pid//npig; fpm = gid*GSM
    gsm = min(nm-fpm,GSM); pm = fpm+((pid%npig)%gsm); pn = (pid%npig)//gsm
    rm = (pm*BM+tl.arange(0,BM))%M; rn = (pn*BN+tl.arange(0,BN))%N
    rk = tl.arange(0,BK)
    ap = a_ptr+rm[:,None]*sa0+rk[None,:]*sa1
    bp = b_ptr+rk[:,None]*sb0+rn[None,:]*sb1
    acc = tl.zeros((BM,BN),dtype=tl.float32)
    for k in range(0, tl.cdiv(K,BK)):
        a = tl.load(ap, mask=rk[None,:]<K-k*BK, other=0.0)
        b = tl.load(bp, mask=rk[:,None]<K-k*BK, other=0.0)
        acc = tl.dot(a,b,acc)
        ap += BK*sa1; bp += BK*sb0
    rc = pm*BM+tl.arange(0,BM); sc2 = pn*BN+tl.arange(0,BN)
    cp = c_ptr+sc0*rc[:,None]+sc1*sc2[None,:]
    tl.store(cp, acc.to(tl.float16), mask=(rc[:,None]<M)&(sc2[None,:]<N))

def run(a, b, bm, bn, bk, nw, ns, mfma, gsm):
    M,K = a.shape; K2,N = b.shape
    c = torch.empty((M,N),device=a.device,dtype=torch.float16)
    grid = (triton.cdiv(M,bm)*triton.cdiv(N,bn),)
    matmul_k[grid](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
        c.stride(0),c.stride(1),BM=bm,BN=bn,BK=bk,GSM=gsm,mfma=mfma,ACTIVATION='',
        num_warps=nw,num_stages=ns)
    return c

def bench(fn, M, N, K):
    ms = triton.testing.do_bench(fn, rep=5)
    return 2*M*N*K*1e-12/(ms*1e-3)

print('M=1536, M=1664: BK sweep (64x64 tile, nW=8, nS=2, mfma=16, GSM=8)')
print(f"{'M':>5}  {'BK=64':>8}  {'BK=128':>8}  {'BK=256':>8}  {'BK=512':>8}  winner")
for M in [1536, 1664]:
    N = K = M
    a = torch.randn((M,K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K,N), device=DEVICE, dtype=torch.float16)
    results = {}
    for bk in [64, 128, 256, 512]:
        try:
            run(a,b,64,64,bk,8,2,16,8)
            tf = bench(lambda: run(a,b,64,64,bk,8,2,16,8), M,N,K)
            results[bk] = tf
        except Exception as e:
            results[bk] = None
    vals = "  ".join(f"{results[bk]:>8.1f}" if results[bk] else f"{'ERR':>8}"
                     for bk in [64,128,256,512])
    best = max((bk for bk in results if results[bk]), key=lambda b: results[b])
    print(f"{M:>5}  {vals}  BK={best}")

# Also check what generates for M=1536
print()
from triton._C.libtriton.amd import perf_model as pm
hw = pm.HardwareInfo.get('gfx950')
for M in [1536, 1664]:
    prob = pm.GemmProblem(M, M, M, pm.ElemKind.FP16, pm.ElemKind.FP16, pm.ElemKind.FP32, 16, 16, 32)
    cands = pm.generate_candidates(prob, hw)
    ranked = pm.rank_configs(prob, cands, hw, top_k=3)
    tiles = [(M+64-1)//64 * ((M+64-1)//64)]
    print(f"M={M}: {len(cands)} candidates, top-3:")
    for c in ranked:
        e = pm.estimate_perf(prob, c, hw)
        print(f"  BM={c.block_m} BN={c.block_n} BK={c.block_k} nS={c.num_stages}: "
              f"{e.predicted_tflops:.0f} TFLOPS")
