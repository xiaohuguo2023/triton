"""Perf-debug: why does PerfModel only tie autotune on VERY_LARGE / MEDIUM
(large-M dense) shapes? Decompose each loss into:

  * PM top-1        -- what PerfModel actually serves (ranked[0])
  * PM oracle       -- best config WITHIN PerfModel's own ranked candidate set
  * autotune winner -- best of the tutorial's HIP autotune grid

Interpretation:
  PM-top1 << PM-oracle            -> RANKING bug (PM has a good config, ranks it low)
  PM-oracle << autotune winner    -> CANDIDATE-GEN gap (PM's space misses the winner)
For each loser we also report where the autotune winner sits in PM's ranking.

Run INSIDE xguo-perfmodel, GPU 0:
    HIP_VISIBLE_DEVICES=0 MPLBACKEND=Agg python3 pm_debug_verylarge.py
"""
import importlib.util, io, contextlib, sys, time
from pathlib import Path
import torch, triton

TUT = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
spec = importlib.util.spec_from_file_location("tut03", TUT)
tut = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try: spec.loader.exec_module(tut)
    except BaseException: pass

from triton.backends.amd import amd_gemm_selector as sel

# MEDIUM losers (pm_vs_auto<1) after tie-break fix
SHAPES = [
    (4096,9216,4096,"M-qwen"),(4096,5120,2880,"M-gptoss"),(4096,5120,5120,"M-llama4"),(4096,3072,4096,"M-qwen"),(4096,5120,8192,"M-llama4"),(4096,2880,4096,"M-gptoss"),(8192,2112,7168,"M-deepseek"),
]

def bench_ms(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def tflops(M, N, K, ms): return 2*M*N*K/(ms*1e-3)/1e12

def cfg_str(c): return f"BM{c.block_m} BN{c.block_n} BK{c.block_k} W{c.num_warps} S{c.num_stages}"

def bench_cfg(a, b, M, N, K, c, dtype):
    try:
        kw = sel.config_to_kernel_kwargs(c)
        out = torch.empty((M, N), device="cuda", dtype=dtype)
        grid = (triton.cdiv(M, c.block_m) * triton.cdiv(N, c.block_n),)
        def run():
            tut.matmul_kernel_amd[grid](a, b, out, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1), ACTIVATION="", **kw)
        return bench_ms(run)
    except Exception as ex:
        return None

def main():
    arch = sel.current_amd_arch()
    dtype = torch.float16
    for (M, N, K, tag) in SHAPES:
        print(f"\n{'='*70}\n{M}x{N}x{K}  ({tag})")
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b = torch.randn((K, N), device="cuda", dtype=dtype)

        # PM full ranking
        ranked = sel.pick_gemm_config(M, N, K, "fp16", arch, top_k=300)
        pm_top1 = ranked[0]
        t_top1 = bench_cfg(a, b, M, N, K, pm_top1, dtype)
        # PM oracle: best within PM's ranked candidate set (bench all, cap 60 for time)
        best_ms, best_c, best_rank = t_top1, pm_top1, 0
        for i, c in enumerate(ranked[:60]):
            ms = bench_cfg(a, b, M, N, K, c, dtype)
            if ms and (best_ms is None or ms < best_ms):
                best_ms, best_c, best_rank = ms, c, i

        # autotune winner (tutorial grid) via its autotuner cache
        _ = tut.matmul(a, b)  # warms the autotuner
        auto_c = None
        try:
            at = tut.matmul_kernel_amd
            cache = getattr(at, "cache", {})
            if cache: auto_c = list(cache.values())[-1]
        except Exception: pass
        t_auto = bench_ms(lambda: tut.matmul(a, b))

        def tf(ms): return f"{tflops(M,N,K,ms):6.0f}" if ms else "  FAIL"
        print(f"  PM top-1     : {tf(t_top1)} TF   {cfg_str(pm_top1)}")
        print(f"  PM oracle    : {tf(best_ms)} TF   {cfg_str(best_c)}   (PM rank #{best_rank})")
        print(f"  autotune     : {tf(t_auto)} TF   {auto_c if auto_c else '(cfg n/a)'}")
        if t_top1 and t_auto:
            print(f"  --> PM-top1/auto = {tflops(M,N,K,t_top1)/tflops(M,N,K,t_auto):.3f}"
                  f"   PM-oracle/auto = {tflops(M,N,K,best_ms)/tflops(M,N,K,t_auto):.3f}"
                  f"   top1/oracle = {tflops(M,N,K,t_top1)/tflops(M,N,K,best_ms):.3f}")
        del a, b; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
