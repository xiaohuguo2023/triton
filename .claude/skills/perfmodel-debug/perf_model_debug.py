#!/usr/bin/env python3
"""Reusable PerfModel debugging harnesses.

This consolidates the scratch scripts used during the gfx950 perf-model
refinement.  The subcommands intentionally keep their benchmark scopes small:
they are diagnostic probes, not full perf baselines.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# This file lives at <repo>/.claude/skills/perfmodel-debug/perf_model_debug.py.
# Resolve the repo root robustly (walk up to the dir containing third_party/)
# rather than a fixed parents[N], so moving the skill dir doesn't silently break
# the third_party.amd.backend.amd_gemm_selector import.
def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "third_party" / "amd").is_dir():
            return parent
    return start.parents[3]  # fallback: <repo>/.claude/skills/perfmodel-debug


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DEFAULT_DATASETS_DIR = Path(
    os.environ.get("PERFMODEL_DATASETS_DIR", "/home/xiaohugu/mi450/datasets")
)


GPTOSS_PROJECTIONS = [
    ("attn_QKV", 5120, 2880),
    ("attn_O", 2880, 4096),
    ("router", 128, 2880),
    ("lm_head", 201088, 2880),
]
GPTOSS_MS = [16, 64, 256, 1024, 4096, 8192]


@dataclass(frozen=True)
class Tile:
    bm: int
    bn: int
    bk: int
    nw: int = 8
    ns: int = 2
    gm: int = 1

    @classmethod
    def parse(cls, text: str) -> "Tile":
        # Accept BMxBNxBK, BMxBNxBK:nw:ns, or BMxBNxBK:nw:ns:gm.
        lhs, *rhs = text.split(":")
        bm, bn, bk = [int(v) for v in lhs.lower().split("x")]
        vals = [8, 2, 1]
        for i, value in enumerate(rhs[:3]):
            vals[i] = int(value)
        return cls(bm, bn, bk, vals[0], vals[1], vals[2])

    def label(self) -> str:
        return f"{self.bm}x{self.bn}x{self.bk} nw{self.nw} ns{self.ns} gm{self.gm}"


def add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def perf_model():
    add_repo_to_path()
    from triton._C.libtriton import amd

    return amd.perf_model


def elem_kind(pm, dtype: str):
    dtype = dtype.lower()
    if dtype in {"fp16", "float16", "half"}:
        return pm.ElemKind.FP16, 16
    if dtype in {"bf16", "bfloat16"}:
        return pm.ElemKind.BF16, 16
    if dtype in {"fp8"}:
        return pm.ElemKind.FP8, 8
    raise ValueError(f"unsupported dtype: {dtype}")


def make_estimate(
    M: int,
    N: int,
    K: int,
    tile: Tile,
    arch: str,
    dtype: str = "bf16",
    c_bits: int = 32,
    mfma_non_k_dim: int = 16,
):
    pm = perf_model()
    kind, bits = elem_kind(pm, dtype)
    prob = pm.GemmProblem(M, N, K, kind, kind, pm.ElemKind.FP32, bits, bits, c_bits)
    cfg = pm.TritonGemmConfig()
    cfg.block_m, cfg.block_n, cfg.block_k = tile.bm, tile.bn, tile.bk
    cfg.num_warps, cfg.num_stages, cfg.group_size_m = tile.nw, tile.ns, tile.gm
    cfg.mfma_non_k_dim = mfma_non_k_dim
    return pm.estimate_perf(prob, cfg, pm.HardwareInfo.get(arch))


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def bench(fn, warmup: int, rep: int) -> float:
    import triton

    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    except Exception:
        return float("inf")


def import_aiter():
    add_repo_to_path()
    import aiter.ops.triton.gemm.basic.gemm_a16w16 as gmod
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    return gmod, gemm_a16w16, gmod._get_triton_config


def bench_aiter_tile(M: int, N: int, K: int, tile: Tile, x, w, warmup: int, rep: int):
    gmod, gemm_a16w16, orig = import_aiter()
    base, _ = orig(M, N, K)
    cfg = dict(base)
    cfg.update(
        BLOCK_SIZE_M=tile.bm,
        BLOCK_SIZE_N=tile.bn,
        BLOCK_SIZE_K=tile.bk,
        num_warps=tile.nw,
        num_stages=tile.ns,
        GROUP_SIZE_M=tile.gm,
        waves_per_eu=0,
    )
    gmod._get_triton_config = lambda *args, _cfg=cfg: (_cfg, True)
    try:
        gemm_a16w16(x, w)
        ms = bench(lambda: gemm_a16w16(x, w), warmup, rep)
        return tflops(M, N, K, ms), ms, None
    except Exception as exc:  # keep diagnostics moving through invalid configs
        return -1.0, float("inf"), str(exc).split("\n", 1)[0][:120]
    finally:
        gmod._get_triton_config = orig


def pick_configs(M: int, N: int, K: int, dtype: str, arch: str, top_k: int = 1):
    add_repo_to_path()
    try:
        from third_party.amd.backend.amd_gemm_selector import pick_gemm_config, config_to_kernel_kwargs
    except ModuleNotFoundError:
        from triton.backends.amd.amd_gemm_selector import pick_gemm_config, config_to_kernel_kwargs

    return pick_gemm_config(M, N, K, dtype, arch, top_k=top_k), config_to_kernel_kwargs


def cmd_aiter_baseline(args) -> None:
    import torch

    gmod, gemm_a16w16, orig = import_aiter()
    ratios = []
    print(f"{'proj':9}{'M':>7}{'tbl_ms':>9}{'pm_ms':>9}{'pm/tbl':>8}  pick")
    print("-" * 72)
    for name, N, K in GPTOSS_PROJECTIONS:
        for M in args.ms:
            x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
            w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

            gmod._get_triton_config = orig
            table_ms = bench(lambda: gemm_a16w16(x, w), args.warmup, args.rep)

            configs, to_kwargs = pick_configs(M, N, K, "bf16", args.arch, top_k=1)
            if not configs:
                print(f"{name:9}{M:>7}  no PM config")
                continue
            picked = configs[0]
            kwargs = to_kwargs(picked)
            base, _ = orig(M, N, K)
            cfg = dict(base)
            cfg.update(
                BLOCK_SIZE_M=kwargs["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=kwargs["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=kwargs["BLOCK_SIZE_K"],
                matrix_instr_nonkdim=kwargs["matrix_instr_nonkdim"],
                GROUP_SIZE_M=kwargs["GROUP_SIZE_M"],
                num_warps=kwargs["num_warps"],
                num_stages=kwargs["num_stages"],
                waves_per_eu=kwargs["waves_per_eu"],
            )
            gmod._get_triton_config = lambda *a, _cfg=cfg: (_cfg, True)
            pm_ms = bench(lambda: gemm_a16w16(x, w), args.warmup, args.rep)
            ratio = table_ms / pm_ms if pm_ms < float("inf") else 0.0
            ratios.append(ratio)
            flag = " <<" if ratio < args.flag_ratio else ""
            print(
                f"{name:9}{M:>7}{table_ms:>9.3f}{pm_ms:>9.3f}{ratio:>7.2f}x  "
                f"{picked.block_m}x{picked.block_n}x{picked.block_k} "
                f"nw{picked.num_warps} ns{picked.num_stages} gm{max(1, picked.group_size_m)}{flag}"
            )
            del x, w
            torch.cuda.empty_cache()
    gmod._get_triton_config = orig
    print("-" * 72)
    if ratios:
        print(f"BASELINE geomean pm/table = {math.prod(ratios) ** (1 / len(ratios)):.3f}x")


def cmd_gm_sweep(args) -> None:
    tiles = [Tile.parse(t) for t in args.tiles]
    print(f"{args.M}x{args.N}x{args.K} — predicted TFLOPS vs group_size_m:")
    print(f"  {'tile':18}{'gm1':>8}{'gm2':>8}{'gm4':>8}{'gm8':>8}")
    for tile in tiles:
        row = []
        for gm in [1, 2, 4, 8]:
            e = make_estimate(args.M, args.N, args.K, Tile(tile.bm, tile.bn, tile.bk, tile.nw, tile.ns, gm), args.arch)
            row.append(e.predicted_tflops)
        print(f"  {tile.bm}x{tile.bn}x{tile.bk:<8}" + "".join(f"{v:>8.0f}" for v in row))


def cmd_efficiency(args) -> None:
    import torch

    shapes = [tuple(int(v) for v in s.lower().split("x")) for s in args.shapes]
    tiles = [Tile.parse(t) for t in args.tiles]
    pm = perf_model()
    hw = pm.HardwareInfo.get(args.arch)
    clock = hw.clock_mhz * 1e6
    by_tile = defaultdict(list)
    print(f"{'MxNxK':>18}{'tile':>20}{'real':>7}{'ideal':>7}{'eff':>6}{'cb':>3}{'nW':>5}  cc/mc")
    for M, N, K in shapes:
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        for tile in tiles:
            e = make_estimate(M, N, K, tile, args.arch)
            denom = e.compute_cycles * e.num_waves
            ideal = 2 * M * N * K / (denom / clock) / 1e12 if denom > 0 else 0.0
            real, _, _ = bench_aiter_tile(M, N, K, tile, x, w, args.warmup, args.rep)
            eff = real / ideal if ideal > 0 and real > 0 else 0.0
            # This is intentionally the raw theoretical-compute-vs-memory test,
            # not e.is_compute_bound, so the probe measures realized MFMA
            # efficiency against the unde-rated compute roofline.
            cb = e.compute_cycles >= e.memory_cycles
            print(
                f"{f'{M}x{N}x{K}':>18}{tile.label():>20}"
                f"{real:>7.0f}{ideal:>7.0f}{eff:>6.2f}{int(cb):>3}{e.num_waves:>5.0f}  "
                f"{e.compute_cycles:.0f}/{e.memory_cycles:.0f}"
            )
            if cb and real > 0 and ideal > 0:
                by_tile[(tile.bm, tile.bn, tile.bk, tile.nw)].append(eff)
        del x, w
        torch.cuda.empty_cache()
    print("\n=== MFMA efficiency per tile (compute-bound rows only) ===")
    for key in sorted(by_tile):
        vals = by_tile[key]
        print(f"  {key[0]}x{key[1]}x{key[2]} nw{key[3]}: n={len(vals)} mean={sum(vals)/len(vals):.3f} min={min(vals):.3f} max={max(vals):.3f}")


def cmd_bk32_ab(args) -> None:
    import torch

    gmod, _, orig = import_aiter()
    base, _ = orig(args.base_M, args.base_N, args.base_K)
    interesting = {
        key: base[key]
        for key in base
        if key in {"kpack", "matrix_instr_nonkdim", "waves_per_eu", "num_stages", "cache_modifier", "NUM_KSPLIT"}
        or "ping" in key.lower()
        or "async" in key.lower()
        or "stage" in key.lower()
    }
    print("base cfg keys:", interesting)
    cases = [
        ("lm_head", 201088, 2880, 8192, 4),
        ("lm_head", 201088, 2880, 256, 1),
        ("attn_QKV", 5120, 2880, 8192, 8),
    ]
    tiles = [Tile.parse(t) for t in args.tiles]
    for name, N, K, M, gm in cases:
        print(f"\n### {name} M={M} N={N} K={K} gm={gm}")
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        for tile in tiles:
            bm = 128 if M == 256 and tile.bm == 256 else tile.bm
            bn = 128 if M == 256 and tile.bn == 256 else tile.bn
            real, _, err = bench_aiter_tile(M, N, K, Tile(bm, bn, tile.bk, tile.nw, tile.ns, gm), x, w, args.warmup, args.rep)
            note = "" if real > 0 else f" ({err})"
            print(f"   {bm}x{bn}x{tile.bk} nw{tile.nw} ns{tile.ns}: REAL={real:.0f}{note}")
        del x, w
        torch.cuda.empty_cache()
    gmod._get_triton_config = orig


def cmd_lds_probe(args) -> None:
    import torch

    gmod, gemm_a16w16, orig = import_aiter()
    fn = gmod._gemm_a16_w16_kernel
    while not hasattr(fn, "device_caches") and hasattr(fn, "fn"):
        fn = fn.fn
    if not hasattr(fn, "device_caches"):
        raise RuntimeError(f"could not unwrap JITFunction: {type(fn)}")

    def real_shared():
        best = 0
        for dc in fn.device_caches.values():
            cache = dc[0] if isinstance(dc, (tuple, list)) else dc
            values = cache.values() if hasattr(cache, "values") else []
            for compiled in values:
                shared = getattr(getattr(compiled, "metadata", None), "shared", None)
                if shared:
                    best = max(best, shared)
        return best

    M, N, K = args.M, args.N, args.K
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    print(f"{'config':24}{'model':>8}{'real':>8}{'ratio':>8}")
    for tile in [Tile.parse(t) for t in args.tiles]:
        model = make_estimate(M, N, K, tile, args.arch).lds_bytes
        fn.device_caches.clear()
        base, _ = orig(M, N, K)
        cfg = dict(base)
        cfg.update(
            BLOCK_SIZE_M=tile.bm,
            BLOCK_SIZE_N=tile.bn,
            BLOCK_SIZE_K=tile.bk,
            num_warps=tile.nw,
            num_stages=tile.ns,
            GROUP_SIZE_M=tile.gm,
            waves_per_eu=0,
        )
        gmod._get_triton_config = lambda *a, _cfg=cfg: (_cfg, True)
        try:
            gemm_a16w16(x, w)
            real = real_shared()
            ratio = model / real if real else 0.0
            print(f"{tile.label():24}{model:>8}{real:>8}{ratio:>8.3f}")
        except Exception as exc:
            print(f"{tile.label():24}{model:>8}   ERR {str(exc).split('.')[0][-70:]}")
    gmod._get_triton_config = orig


def cmd_bk32_diag(args) -> None:
    import torch

    cases = [
        ("lm_head", 201088, 2880, 8192),
        ("lm_head", 201088, 2880, 256),
        ("attn_QKV", 5120, 2880, 8192),
    ]
    for name, N, K, M in cases:
        print(f"\n### {name} M={M} N={N} K={K}")
        ranked, _ = pick_configs(M, N, K, "bf16", args.arch, top_k=args.top_k)
        if not ranked:
            print("  no ranked configs")
            continue
        top = ranked[0]
        top_gm = max(1, top.group_size_m)
        print(f" top-1 pick: {top.block_m}x{top.block_n}x{top.block_k} nw{top.num_warps} ns{top.num_stages} gm{top_gm}")
        print(f" top-{args.top_k} ranked:")
        for cfg in ranked:
            tile = Tile(cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_warps, cfg.num_stages, max(1, cfg.group_size_m))
            e = make_estimate(M, N, K, tile, args.arch)
            print(
                f"   {tile.label()}: pred={e.predicted_tflops:.0f} valid={int(e.is_valid)} "
                f"ldsX={int(e.lds_exceeded)} spill={int(e.likely_spills)} lds={e.lds_bytes} cb={int(e.is_compute_bound)}"
            )
        print(f" BK siblings of {top.block_m}x{top.block_n} nw{top.num_warps} ns{top.num_stages}:")
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        for bk in [32, 64, 128, 256]:
            tile = Tile(top.block_m, top.block_n, bk, top.num_warps, top.num_stages, top_gm)
            e = make_estimate(M, N, K, tile, args.arch)
            real, _, _ = bench_aiter_tile(M, N, K, tile, x, w, args.warmup, args.rep)
            print(
                f"   BK{bk}: pred={e.predicted_tflops:.0f} valid={int(e.is_valid)} "
                f"ldsX={int(e.lds_exceeded)} spill={int(e.likely_spills)} lds={e.lds_bytes} REAL={real:.0f}"
            )
        del x, w
        torch.cuda.empty_cache()


def cmd_tensoratlas_misses(args) -> None:
    import torch
    import triton
    import triton.language as tl
    import yaml

    @triton.jit
    def matmul_k(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        matrix_instr_nonkdim: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    def load_dataset(dataset: str):
        for suffix in ("_tuned.yaml", "_shapes_tuned.yaml"):
            path = args.datasets_dir / f"{dataset}{suffix}"
            if path.exists():
                with path.open() as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError(f"no tuned YAML found for {dataset} under {args.datasets_dir}")

    def run_matmul(a, b, cfg):
        bm, bn, bk, nw, ns, mfma, gm = cfg
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        matmul_k[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_SIZE_M=gm,
            matrix_instr_nonkdim=mfma,
            ACTIVATION="",
            num_warps=nw,
            num_stages=ns,
        )
        return c

    def real_tf(a, b, cfg, M, N, K):
        try:
            run_matmul(a, b, cfg)
            ms = triton.testing.do_bench(lambda: run_matmul(a, b, cfg), rep=args.rep)
            return tflops(M, N, K, ms)
        except Exception:
            return 0.0

    def est_cfg(M, N, K, cfg):
        bm, bn, bk, nw, ns, mfma, gm = cfg
        tile = Tile(bm, bn, bk, nw, ns, gm)
        return make_estimate(M, N, K, tile, args.arch, dtype=args.dtype, c_bits=args.c_bits, mfma_non_k_dim=mfma)

    def diag(dataset: str):
        entries = load_dataset(dataset)
        buckets = {"tile": 0, "bk": 0, "nw": 0, "ns": 0, "mag_only": 0}
        misses = []
        for entry in entries:
            M, N, K = entry["M"], entry["N"], entry["K"]
            tuned_bk = entry["BLOCK_SIZE_K"]
            if tuned_bk & (tuned_bk - 1):
                continue
            picked, _ = pick_configs(M, N, K, args.dtype, args.arch, top_k=1)
            if not picked:
                continue
            pc = picked[0]
            tuned_cfg = (
                entry["BLOCK_SIZE_M"],
                entry["BLOCK_SIZE_N"],
                tuned_bk,
                entry["num_warps"],
                entry["num_stages"],
                entry["matrix_instr_nonkdim"],
                entry.get("GROUP_SIZE_M", 8),
            )
            picked_cfg = (
                pc.block_m,
                pc.block_n,
                pc.block_k,
                pc.num_warps,
                pc.num_stages,
                pc.mfma_non_k_dim,
                max(1, pc.group_size_m),
            )
            a = torch.randn((M, K), device=args.device, dtype=torch.float16)
            b = torch.randn((K, N), device=args.device, dtype=torch.float16)
            tuned_real = real_tf(a, b, tuned_cfg, M, N, K)
            picked_real = real_tf(a, b, picked_cfg, M, N, K)
            del a, b
            torch.cuda.empty_cache()
            ratio = picked_real / tuned_real if tuned_real > 0 else 0.0
            if ratio >= args.thresh or tuned_real == 0:
                continue
            tuned_est = est_cfg(M, N, K, tuned_cfg)
            picked_est = est_cfg(M, N, K, picked_cfg)
            if (picked_cfg[0], picked_cfg[1]) != (tuned_cfg[0], tuned_cfg[1]):
                bucket = "tile"
            elif picked_cfg[2] != tuned_cfg[2]:
                bucket = "bk"
            elif picked_cfg[3] != tuned_cfg[3]:
                bucket = "nw"
            elif picked_cfg[4] != tuned_cfg[4]:
                bucket = "ns"
            else:
                bucket = "mag_only"
            buckets[bucket] += 1
            misses.append((M, N, K, ratio, bucket, tuned_cfg, picked_cfg, tuned_real, picked_real, tuned_est, picked_est))
        return buckets, misses

    for dataset in args.datasets:
        buckets, misses = diag(dataset)
        print(f"\n### {dataset}: {len(misses)} misses (ratio<{args.thresh}) buckets={buckets}")
        for M, N, K, ratio, bucket, tuned, picked, rt, rp, et, ep in misses:
            print(f"\n {M}x{N}x{K} ratio={ratio:.2f} [{bucket}]")
            print(
                f"   tuned {tuned[0]}x{tuned[1]}x{tuned[2]} nw{tuned[3]} ns{tuned[4]} gm{tuned[6]}: "
                f"real={rt:.0f} pred={et.predicted_tflops:.0f} cb={int(et.is_compute_bound)} "
                f"nW={et.num_waves:.0f} occ={et.occupancy:.2f} "
                f"c/m/l={et.compute_cycles:.0f}/{et.memory_cycles:.0f}/{et.lds_cycles:.0f}"
            )
            print(
                f"   PM    {picked[0]}x{picked[1]}x{picked[2]} nw{picked[3]} ns{picked[4]} gm{picked[6]}: "
                f"real={rp:.0f} pred={ep.predicted_tflops:.0f} cb={int(ep.is_compute_bound)} "
                f"nW={ep.num_waves:.0f} occ={ep.occupancy:.2f} "
                f"c/m/l={ep.compute_cycles:.0f}/{ep.memory_cycles:.0f}/{ep.lds_cycles:.0f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("TRITON_ARCH", "gfx950"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("aiter-baseline", help="Run the aiter gpt-oss 24-cell table-vs-PM benchmark.")
    p.add_argument("--ms", nargs="*", type=int, default=GPTOSS_MS)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rep", type=int, default=60)
    p.add_argument("--flag-ratio", type=float, default=0.90)
    p.set_defaults(func=cmd_aiter_baseline)

    p = sub.add_parser("gm-sweep", help="Check predicted TFLOPS sensitivity to GROUP_SIZE_M.")
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=201088)
    p.add_argument("--K", type=int, default=2880)
    p.add_argument("--tiles", nargs="*", default=["64x256x64:8:2", "256x128x64:8:2", "256x256x64:8:2", "128x128x128:8:2"])
    p.set_defaults(func=cmd_gm_sweep)

    p = sub.add_parser("efficiency", help="Measure realized MFMA efficiency on high-AI square shapes.")
    p.add_argument("--shapes", nargs="*", default=["4096x4096x4096", "8192x8192x8192", "8192x8192x4096", "6144x6144x6144"])
    p.add_argument("--tiles", nargs="*", default=[
        "256x256x64:8:2", "256x128x64:8:2", "128x256x64:8:2", "128x128x64:8:2",
        "256x256x64:4:2", "256x128x64:4:2", "128x128x64:4:2",
        "64x256x64:8:2", "256x64x64:8:2", "64x128x64:4:2", "128x64x64:4:2", "64x64x64:4:2",
    ])
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rep", type=int, default=40)
    p.set_defaults(func=cmd_efficiency)

    p = sub.add_parser("bk32-diag", help="Dump top-ranked aiter configs and BK siblings for BK32 investigations.")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rep", type=int, default=40)
    p.set_defaults(func=cmd_bk32_diag)

    p = sub.add_parser("bk32-ab", help="Benchmark exact BK32/BK64 and ns2/ns3 aiter A/B configs.")
    p.add_argument("--base-M", type=int, default=8192)
    p.add_argument("--base-N", type=int, default=201088)
    p.add_argument("--base-K", type=int, default=2880)
    p.add_argument("--tiles", nargs="*", default=[
        "256x256x32:8:2", "256x256x32:8:3", "256x256x64:8:2", "256x256x64:8:3",
        "256x128x32:8:3", "256x128x64:8:2", "256x128x64:8:3",
    ])
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=60)
    p.set_defaults(func=cmd_bk32_ab)

    p = sub.add_parser("lds-probe", help="Compare model LDS bytes against compiled aiter metadata.shared.")
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=201088)
    p.add_argument("--K", type=int, default=2880)
    p.add_argument("--tiles", nargs="*", default=[
        "256x256x64:8:2", "128x128x64:8:2", "256x128x64:8:2", "256x128x64:8:3",
        "128x256x64:8:3", "128x128x64:8:3", "256x256x32:8:3", "64x64x128:8:2",
        "128x128x128:8:2", "256x128x32:8:3",
    ])
    p.set_defaults(func=cmd_lds_probe)

    p = sub.add_parser("tensoratlas-misses", help="Bucket TensorAtlas residual misses against tuned YAML datasets.")
    p.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--thresh", type=float, default=0.95)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--c-bits", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rep", type=int, default=5)
    p.set_defaults(func=cmd_tensoratlas_misses)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
