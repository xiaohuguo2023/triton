"""On-the-fly a8w4 config selection helper (benchmark injection).

Exposes pick_a8w4(...) returning the aiter kernel-config dict, chosen by the
standalone perf_model.so via rank_configs over the feasible tile grid. Lazy,
cached, no monkeypatch side effects -- meant to be called from an env-gated
branch inside aiter's get_kernel_config_triton for A/B benchmarking.
"""
import functools
import os
import sys

# Import priority for the perf-model bindings:
#   1. triton._C.libtriton.amd.perf_model  -- the in-Triton submodule (fork build).
#      It already owns the pybind types, so importing the standalone .so on top
#      would raise "ElemKind already registered".
#   2. standalone perf_model.so from AITER_A8W4_PERFMODEL_SO_DIR (default:
#      the directory of this file) -- for stock Triton with no amd.perf_model.
#   3. fail loudly with an actionable message.
# Both paths expose the SAME API (PerfModelBindings.cpp is the single source):
#   ElemKind, KernelType, HardwareInfo, GemmProblem, rank_configs,
#   estimate_perf, select_group_size_m, __perf_model_revision__.
_REQUIRED_SYMS = (
    "ElemKind", "KernelType", "HardwareInfo", "GemmProblem",
    "rank_configs", "estimate_perf", "select_group_size_m",
    "__perf_model_revision__",
)


def _check_syms(mod, source):
    missing = [s for s in _REQUIRED_SYMS if not hasattr(mod, s)]
    if missing:
        raise ImportError(
            f"perf_model loaded from {source} is missing symbols {missing}. "
            "The bindings are likely stale -- rebuild via "
            "third_party/amd/build_standalone_perf_model.sh (standalone) or "
            "rebuild libtriton (in-Triton submodule)."
        )
    return mod


def _load_perf_model():
    # 1. in-Triton submodule
    try:
        from triton._C.libtriton import amd as _amd
        _mod = _amd.perf_model
        _ = _mod.ElemKind  # AttributeError if the submodule isn't built in
        return _check_syms(_mod, "triton._C.libtriton.amd.perf_model")
    except Exception:
        pass
    # 2. standalone perf_model.so
    _so_dir = os.environ.get(
        "AITER_A8W4_PERFMODEL_SO_DIR", os.path.dirname(os.path.abspath(__file__))
    )
    if _so_dir not in sys.path:
        sys.path.insert(0, _so_dir)
    try:
        import perf_model as _mod
    except Exception as e:  # 3. fail loudly
        raise ImportError(
            "Could not load the perf_model bindings. Tried the in-Triton "
            "submodule (triton._C.libtriton.amd.perf_model) and the standalone "
            f"perf_model.so in AITER_A8W4_PERFMODEL_SO_DIR={_so_dir!r}. "
            "Build the standalone with "
            "third_party/amd/build_standalone_perf_model.sh inside the serving "
            f"container. Underlying error: {e!r}"
        ) from e
    return _check_syms(_mod, f"{_so_dir}/perf_model.so")


_pm = _load_perf_model()

_F8, _F4, _F32 = _pm.ElemKind.FP8, _pm.ElemKind.FP4, _pm.ElemKind.FP32
_MFMA = 16
# ns=1 is intentionally excluded: aiter's own generate_candidates never emits
# num_stages=1 on CDNA4 (it disables the load/compute software pipeline), so it
# must not be a candidate here either.
_GRID = [(bn, bk, ns, nw)
         for bn in (64, 128, 256, 512)
         for bk in (256, 512)
         for ns in (2, 3)
         for nw in (4, 8)]
_HW = {}
# routing-skew capture state (only used when AITER_A8W4_HIST_LOG is set)
_HIST_SEEN = {}
_HIST_CAP = 8   # samples per (n,k,block_m)


def _hw(arch):
    hw = _HW.get(arch)
    if hw is None:
        hw = _HW[arch] = _pm.HardwareInfo.get(arch)
    return hw


def _mk(bm, bn, bk, ns, nw):
    c = _pm.TritonGemmConfig()
    c.block_m, c.block_n, c.block_k = bm, bn, bk
    c.num_stages, c.num_warps, c.group_size_m = ns, nw, 1
    c.mfma_non_k_dim = _MFMA
    c.waves_per_eu = 0
    return c


# Representative routed-rows M per block_m regime, used ONLY as a ranking proxy
# so the A8W4 MoE config is selected per generic shape (arch, block_m, N, K,
# swizzle) instead of per exact routed rows. Selecting per-shape:
#   * avoids exact-m cache misses / per-call rank_configs during serving, and
#   * yields a small fixed config set that warmup fully precompiles (no JIT at
#     inference), matching the tuned JSON's per-(bm,N,K) granularity.
#
# The rule is MODEL-AGNOSTIC — derived only from block_m, with no expert-count,
# top-k, or hidden/intermediate assumptions:
#   * block_m <= 16 is the routing DECODE floor (tokens_per_expert ~= 1); a small
#     representative m preserves the under-filled / narrow-BN decode behavior.
#   * block_m > 16 is the prefill regime; ~32 M-blocks (block_m*32) puts ranking
#     in the saturated / wide-BN regime.
# Calibrated + validated on gpt-oss (the available A8W4 model); should be
# re-validated on other A8W4 MoE shape families as they appear. Escape hatch:
# AITER_A8W4_EXACT_M=1 falls back to exact-m ranking (per-call, may JIT) if a
# future model proves the generic proxy wrong for some shape.
def _canonical_m(block_m):
    return block_m * 2 if block_m <= 16 else block_m * 32


def _next_pow2(x):
    x = int(x)
    return 1 << max(0, (x - 1)).bit_length()


# C-class selector fix (measured 2026-07-13, see project_a8w4_pm_vs_json_measured_attribution):
# the single canonical_m (block_m*32) is systematically ~2-23x below the real routed-rows M,
# which mis-ranks BN/BK for small block_m; an exact-m measured pass recovered those cells
# (C-class) but REGRESSED some block_m==128 cells (B-class num_warps over-ranking).
# So: for block_m<=64 rank at next_pow2(real m) -- near-exact but a BOUNDED bucket count
# (bm32/64 see only ~2 buckets, so warmup still covers them / no per-call JIT).
# For block_m==128 keep canonical_m (defer to the perf-model num_warps calibration).
# AITER_A8W4_MBUCKET=0 disables (pure canonical); AITER_A8W4_EXACT_M=1 forces exact everywhere.
def _rank_m(block_m, m):
    import os as _os
    if _os.environ.get("AITER_A8W4_MBUCKET") == "0":
        return _canonical_m(block_m)
    if block_m <= 64:
        return _next_pow2(m)
    return _canonical_m(block_m)


# --- Single-run forced-config A/B (measurement only) ---------------------------
# AITER_A8W4_AB pins a target shape to two configs, alternating call-by-call in
# ONE serve so both are measured under identical routing / window / thermal.
# Format: "bm:n:k=bn/bk/ns/nw|bn/bk/ns/nw;..."  (left=A e.g. JSON, right=B e.g. PM)
# The 587 record_function label encodes the returned config, so parse_labeled
# disaggregates A vs B. No effect unless the env is set.
_AB_SPEC = None
_AB_CTR = {}


def _ab_spec():
    global _AB_SPEC
    if _AB_SPEC is None:
        _AB_SPEC = {}
        raw = os.environ.get("AITER_A8W4_AB", "")
        for part in raw.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, cfgs = part.split("=", 1)
            bm, n, k = (int(x) for x in key.split(":"))
            a, b = cfgs.split("|", 1)
            def _p(s):
                bn, bk, ns, nw = (int(x) for x in s.split("/"))
                return (bn, bk, ns, nw)
            _AB_SPEC[(bm, n, k)] = (_p(a), _p(b))
    return _AB_SPEC


def _ab_override(block_m, n, k, chosen):
    spec = _ab_spec().get((int(block_m), int(n), int(k)))
    if spec is None:
        return chosen
    key = (int(block_m), int(n), int(k))
    i = _AB_CTR.get(key, 0)
    _AB_CTR[key] = i + 1
    return spec[i & 1]


# BN candidate policy by block_m: only REMOVE block_n options that are STRUCTURALLY
# bad for a regime; let PM rank within the rest. NOT a tuned table by TP/conc.
#   bm16,bm32 -> BN<=128 (drop BN256). Validated whole-surface 0.988 (pmcapped).
#     bm16/bm32 are low-tokens/expert => BN256 wastes CU supply on padded work.
# WHY NOT bm16 -> BN64-only: tried it (pmpol sweep) and it REGRESSED to 0.972 --
#   bm16's BN64-vs-BN128 optimum is SHAPE/TP-dependent: BN64 wins at TP8 (small N,K:
#   n2880_k360 3.31x, n720_k2880 1.28x) but BN128 wins at TP4 (larger N,K:
#   n1440_k2880, n2880_k720). A blanket block_m rule cannot capture that. The right
#   place is PM ranking within {64,128}: PM HAS N,K, it just needs a memory-bound
#   occupancy term (reward block-supply at low arithmetic intensity) to pick BN64@TP8
#   and BN128@TP4 itself. That is the next experiment (PerfModel.cpp), not a rule here.
#   bm64,bm128 -> full grid (conc16 bm64 A/B = wash; conc128 unmeasurable) -> leave to PM.
# AITER_A8W4_BNCAP=0 disables (full grid) for A/B.
_ALLOWED_BN_BY_BM = {16: (64, 128), 32: (64, 128)}


@functools.lru_cache(maxsize=None)
def _pick(arch, block_m, n, k, swizzle, rank_m, bncap):
    # a8w4 MoE (pick_a8w4) ONLY; other GEMM paths that genuinely rank on exact M
    # are untouched. rank_m is the (canonical or exact) M used purely for ranking.
    # `bncap` is passed (not read from env here) so it is part of the lru_cache key
    # -- toggling AITER_A8W4_BNCAP within one process correctly changes the result.
    allowed = _ALLOWED_BN_BY_BM.get(block_m) if bncap else None
    prob = _pm.GemmProblem(rank_m, n, k, _F8, _F4, _F32, 8, 4, 32)
    cfgs = [_mk(block_m, *c) for c in _GRID
            if (swizzle != "CDNA4_SCALE" or c[1] >= 256)
            and (allowed is None or c[0] in allowed)]
    ranked = _pm.rank_configs(prob, cfgs, _hw(arch), 1)
    t = ranked[0]
    return (t.block_n, t.block_k, t.num_stages, t.num_warps)


def pick_a8w4(m, n, k, routing_data, swizzle_mx_scale, arch):
    import os as _os
    block_m = routing_data.block_m
    # Guarded m-bucketing by default (C-class fix); exact-m escape hatch overrides.
    rank_m = int(m) if _os.environ.get("AITER_A8W4_EXACT_M") == "1" \
             else _rank_m(block_m, m)
    # Read the env ONCE and pass it in so it is part of _pick's lru_cache key.
    bncap = _os.environ.get("AITER_A8W4_BNCAP") != "0"
    bn, bk, ns, nw = _pick(arch, block_m, n, k, swizzle_mx_scale, rank_m, bncap)
    if _os.environ.get("AITER_A8W4_AB"):
        bn, bk, ns, nw = _ab_override(block_m, n, k, (bn, bk, ns, nw))
    import os as _os
    _logf = _os.environ.get("AITER_A8W4_PM_LOG")
    if _logf:
        try:
            with open(_logf, "a") as _fh:
                _fh.write(f"{int(m)} {n} {k} {block_m} {bn} {bk} {ns} {nw} {swizzle_mx_scale}\n")
        except Exception:
            pass
    # Routing-skew capture (guarded, sampled): dump the real per-expert token
    # histogram for target shapes so the isolated harness can REPLAY real skew
    # instead of uniform random routing. Cheap: capped samples per (n,k,block_m).
    _histf = _os.environ.get("AITER_A8W4_HIST_LOG")
    if _histf:
        try:
            _key = (int(n), int(k), int(block_m))
            _seen = _HIST_SEEN.setdefault(_key, 0)
            if _seen < _HIST_CAP:
                import json as _json
                _rec = {"m": int(m), "n": int(n), "k": int(k),
                        "block_m": int(block_m), "swizzle": str(swizzle_mx_scale)}
                # per-expert histogram: try routing_data.hist, then .expt_data.hist
                _h = getattr(routing_data, "hist", None)
                if _h is None:
                    _ed = getattr(routing_data, "expt_data", None)
                    _h = getattr(_ed, "hist", None) if _ed is not None else None
                if _h is not None:
                    _hl = (_h.detach().to("cpu").tolist()
                           if hasattr(_h, "detach") else list(_h))
                    _nz = sum(1 for x in _hl if x and x > 0)
                    # Skip warmup/profile-run degenerate routing (routes to a
                    # handful of experts); real gpt-oss routing spreads across
                    # most of the 128 experts. Only real routings count to cap.
                    if _nz >= 32:
                        _rec["hist"] = _hl; _rec["nonzero"] = _nz
                        with open(_histf, "a") as _fh:
                            _fh.write(_json.dumps(_rec) + "\n")
                        _HIST_SEEN[_key] = _seen + 1
                else:
                    _rec["attrs"] = [a for a in dir(routing_data) if not a.startswith("__")]
                    with open(_histf, "a") as _fh:
                        _fh.write(_json.dumps(_rec) + "\n")
                    _HIST_SEEN[_key] = _seen + 1
        except Exception as _e:
            try:
                with open(_histf + ".err", "a") as _fh:
                    _fh.write(repr(_e) + "\n")
            except Exception:
                pass
    return {
        "block_m": block_m, "block_n": bn, "block_k": bk,
        "num_warps": nw, "num_stages": ns,
        "group_m": 4, "xcd_swizzle": 8,
        "w_cache_modifier": ".cg" if block_m <= 32 else None,
        "split_k": 1, "waves_per_eu": 0,
        "matrix_instr_nonkdim": _MFMA, "kpack": 1,
    }
