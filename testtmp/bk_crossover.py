"""Find the M crossover where BK=64 starts beating BK=128 for 64x64 tile."""
import torch, triton
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def mk(a_ptr, b_ptr, c_ptr, M, N, K, sa0,sa1,sb0,sb1,sc0,sc1,
    BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr,
    GSM:tl.constexpr, mfma:tl.constexpr, ACTIVATION:tl.constexpr):
    pid=tl.program_id(0); nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    npig=GSM*nn; gid=pid//npig; fpm=gid*GSM
    gsm=min(nm-fpm,GSM); pm=fpm+((pid%npig)%gsm); pn=(pid%npig)//gsm
    rm=(pm*BM+tl.arange(0,BM))%M; rn=(pn*BN+tl.arange(0,BN))%N; rk=tl.arange(0,BK)
    ap=a_ptr+rm[:,None]*sa0+rk[None,:]*sa1; bp=b_ptr+rk[:,None]*sb0+rn[None,:]*sb1
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k in range(0,tl.cdiv(K,BK)):
        a=tl.load(ap,mask=rk[None,:]<K-k*BK,other=0.0)
        b=tl.load(bp,mask=rk[:,None]<K-k*BK,other=0.0)
        acc=tl.dot(a,b,acc); ap+=BK*sa1; bp+=BK*sb0
    rc=pm*BM+tl.arange(0,BM); sc2=pn*BN+tl.arange(0,BN)
    cp=c_ptr+sc0*rc[:,None]+sc1*sc2[None,:]
    tl.store(cp,acc.to(tl.float16),mask=(rc[:,None]<M)&(sc2[None,:]<N))

def run(a,b,bk):
    M,K=a.shape; c=torch.empty((M,M),device=a.device,dtype=torch.float16)
    grid=(triton.cdiv(M,64)*triton.cdiv(M,64),)
    mk[grid](a,b,c,M,M,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
        c.stride(0),c.stride(1),BM=64,BN=64,BK=bk,GSM=8,mfma=16,ACTIVATION='',
        num_warps=8,num_stages=2)
    return c

print(f"{'M':>5}  {'BK=64':>8}  {'BK=128':>8}  winner")
for M in [512, 768, 1024, 1280, 1536, 2048]:
    a=torch.randn((M,M),device=DEVICE,dtype=torch.float16)
    b=torch.randn((M,M),device=DEVICE,dtype=torch.float16)
    r = {}
    for bk in [64, 128]:
        run(a,b,bk)
        ms=triton.testing.do_bench(lambda: run(a,b,bk), rep=5)
        r[bk]=2*M**3*1e-12/(ms*1e-3)
    w=64 if r[64]>r[128] else 128
    print(f"{M:>5}  {r[64]:>8.1f}  {r[128]:>8.1f}  BK={w}")
