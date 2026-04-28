"""
AMD analytical GEMM config selector — Origami-style predict-and-use.

Replaces @triton.autotune benchmarking with a pure analytical selection:
  1. generate_candidates()  -- wave-based tile enumeration, filtered by
                               LDS/VGPR/kDim constraints
  2. rank_configs(top_k=1) -- roofline + wave-quantisation ranking
  3. Use the top-1 config directly, no GPU benchmarking

The top_k backdoor lets callers retrieve multiple ranked configs for
experimentation or misprediction diagnosis:
  configs = pick_gemm_config(..., top_k=5)
  # configs[0] is the model's best prediction
  # configs[1:] can be benchmarked to detect mispredictions

Usage
-----
  from triton._C.libtriton.amd import perf_model as pm
  from triton.backends.amd.amd_gemm_selector import pick_gemm_config

  cfg = pick_gemm_config(M, N, K, "fp16", arch)[0]
  grid = (triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n))
  matmul_kernel[grid](
      A, B, C, M, N, K,
      BLOCK_SIZE_M=cfg.block_m,
      BLOCK_SIZE_N=cfg.block_n,
      BLOCK_SIZE_K=cfg.block_k,
      matrix_instr_nonkdim=cfg.mfma_non_k_dim,
      num_warps=cfg.num_warps,
      num_stages=cfg.num_stages,
  )
"""

from triton._C.libtriton import amd

_pm = amd.perf_model

# ---------------------------------------------------------------------------
# Dtype string → ElemKind mapping
# ---------------------------------------------------------------------------

_DTYPE_TO_ELEM_KIND = {
    "fp64":  _pm.ElemKind.FP64,
    "fp32":  _pm.ElemKind.FP32,
    "tf32":  _pm.ElemKind.TF32,
    "xf32":  _pm.ElemKind.TF32,
    "fp16":  _pm.ElemKind.FP16,
    "bf16":  _pm.ElemKind.BF16,
    "fp8":   _pm.ElemKind.FP8,
    "fp6":   _pm.ElemKind.FP6,
    "fp4":   _pm.ElemKind.FP4,
    "int8":  _pm.ElemKind.I8,
    "i8":    _pm.ElemKind.I8,
}

# Bits per element for each dtype string — used to populate GemmProblem.
_DTYPE_BITS = {
    "fp64": 64, "fp32": 32, "tf32": 32, "xf32": 32,
    "fp16": 16, "bf16": 16,
    "fp8":   8, "fp6":   6, "fp4":   4,
    "int8":  8, "i8":    8,
}


def _elem_kind(dtype: str) -> _pm.ElemKind:
    """Convert a dtype string to ElemKind. Raises ValueError for unknown dtypes."""
    kind = _DTYPE_TO_ELEM_KIND.get(dtype.lower())
    if kind is None:
        raise ValueError(
            f"Unknown dtype '{dtype}'. "
            f"Supported: {list(_DTYPE_TO_ELEM_KIND.keys())}"
        )
    return kind


def _dtype_bits(dtype: str) -> int:
    bits = _DTYPE_BITS.get(dtype.lower())
    if bits is None:
        raise ValueError(f"Unknown dtype '{dtype}'.")
    return bits


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

def pick_gemm_config(
    M: int,
    N: int,
    K: int,
    a_dtype: str,
    arch: str,
    *,
    b_dtype: str = None,
    c_dtype: str = "fp32",
    top_k: int = 1,
    kernel_type: str = "standard",
):
    """
    Analytically select the best GEMM config(s) for the given problem.

    Parameters
    ----------
    M, N, K     : Problem dimensions.
    a_dtype     : Input A element dtype string (e.g. "fp16", "bf16", "fp8").
    arch        : GPU architecture string (e.g. "gfx942", "gfx90a").
    b_dtype     : Input B element dtype. Defaults to a_dtype.
    c_dtype     : Accumulator dtype. Defaults to "fp32".
    top_k       : Number of configs to return, ranked best-first.
    kernel_type : "standard" (default) for the compiler-pipelined triton matmul;
                  "gluon" for v9-style hand-tuned 4-quadrant kernels.
                  Gluon constraints: numWarps=4, numStages=2,
                  blockM/blockN multiples of 128, K%(2*blockK)==0.

    Returns
    -------
    List[TritonGemmConfig] of length <= top_k, best config first.
    Returns an empty list if no feasible config exists for this hardware.
    """
    if b_dtype is None:
        b_dtype = a_dtype

    a_kind = _elem_kind(a_dtype)
    b_kind = _elem_kind(b_dtype)
    c_kind = _elem_kind(c_dtype)

    hw   = _pm.HardwareInfo.get(arch)
    prob = _pm.GemmProblem(
        M, N, K,
        a_kind, b_kind, c_kind,
        _dtype_bits(a_dtype),
        _dtype_bits(b_dtype),
        _dtype_bits(c_dtype),
    )

    kt_str = (kernel_type or "standard").lower()
    if kt_str == "gluon":
        kt = _pm.KernelType.Gluon
    elif kt_str in ("standard", "std"):
        kt = _pm.KernelType.Standard
    else:
        raise ValueError(f"unknown kernel_type {kernel_type!r} (use 'standard' or 'gluon')")

    candidates = _pm.generate_candidates(prob, hw, kernel_type=kt)
    if not candidates:
        return []

    return _pm.rank_configs(prob, candidates, hw, top_k=top_k)


# ---------------------------------------------------------------------------
# Config → kernel kwargs conversion
# ---------------------------------------------------------------------------

def config_to_kernel_kwargs(cfg) -> dict:
    """
    Convert a TritonGemmConfig to the kwargs dict for a @triton.jit GEMM kernel.

    Returns a flat dict suitable for unpacking directly into the kernel call:
      matmul_kernel[grid](..., **config_to_kernel_kwargs(cfg))

    group_size_m is set by selectGroupSizeM() using Origami's WGM prediction.
    """
    return {
        "BLOCK_SIZE_M":         cfg.block_m,
        "BLOCK_SIZE_N":         cfg.block_n,
        "BLOCK_SIZE_K":         cfg.block_k,
        "matrix_instr_nonkdim": cfg.mfma_non_k_dim,
        "GROUP_SIZE_M":         cfg.group_size_m,
        "num_warps":            cfg.num_warps,
        "num_stages":           cfg.num_stages,
    }


# ---------------------------------------------------------------------------
# Arch helper
# ---------------------------------------------------------------------------

def current_amd_arch() -> str:
    """
    Return the current AMD GPU arch string (e.g. 'gfx942') from Triton's driver.
    Raises RuntimeError if the active backend is not AMD.
    """
    import triton
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        raise RuntimeError(
            f"current_amd_arch() called on non-AMD backend '{target.backend}'"
        )
    return target.arch
