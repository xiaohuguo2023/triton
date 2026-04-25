"""
Tests for amd_gemm_selector.py.

Run after building Triton with TRITON_BUILD_PYTHON_MODULE=ON:
  pip install -e . --no-build-isolation
  pytest third_party/amd/python/test/test_amd_gemm_selector.py -v
"""
import pytest

try:
    from triton._C.libtriton import amd  # noqa: F401
    from third_party.amd.backend.amd_gemm_selector import (
        pick_gemm_config, config_to_kernel_kwargs, _elem_kind, _dtype_bits,
    )
    from triton._C.libtriton.amd import perf_model as pm
except ImportError:
    pytest.skip("triton AMD module not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# _elem_kind and _dtype_bits helpers
# ---------------------------------------------------------------------------

class TestDtypeHelpers:
    def test_fp16_kind(self):
        assert _elem_kind("fp16") == pm.ElemKind.FP16

    def test_bf16_kind(self):
        assert _elem_kind("bf16") == pm.ElemKind.BF16

    def test_fp32_kind(self):
        assert _elem_kind("fp32") == pm.ElemKind.FP32

    def test_case_insensitive(self):
        assert _elem_kind("FP16") == _elem_kind("fp16")

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _elem_kind("float16")

    def test_bits(self):
        assert _dtype_bits("fp16") == 16
        assert _dtype_bits("bf16") == 16
        assert _dtype_bits("fp32") == 32
        assert _dtype_bits("fp8")  == 8


# ---------------------------------------------------------------------------
# pick_gemm_config
# ---------------------------------------------------------------------------

class TestPickGemmConfig:
    def test_returns_list(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")
        assert isinstance(result, list)

    def test_top1_default_returns_single_config(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")
        assert len(result) == 1

    def test_top_k_returns_k_configs(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942", top_k=5)
        assert len(result) == 5

    def test_config_has_correct_mfma_dim(self):
        # gfx942 FP16 → throughput tie → 16x16 wins
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")
        assert result[0].mfma_non_k_dim == 16

    def test_block_sizes_are_positive(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")
        cfg = result[0]
        assert cfg.block_m > 0
        assert cfg.block_n > 0
        assert cfg.block_k > 0

    def test_num_warps_is_power_of_two(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")
        nw = result[0].num_warps
        assert nw > 0 and (nw & (nw - 1)) == 0

    def test_bf16_works(self):
        result = pick_gemm_config(4096, 4096, 4096, "bf16", "gfx942")
        assert len(result) == 1
        assert result[0].mfma_non_k_dim == 16

    def test_gfx90a_works(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx90a")
        assert len(result) == 1

    def test_gfx950_works(self):
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx950")
        assert len(result) == 1

    def test_ranked_order_best_first(self):
        configs = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942", top_k=3)
        hw   = pm.HardwareInfo.get("gfx942")
        prob = pm.GemmProblem(4096, 4096, 4096,
                              pm.ElemKind.FP16, pm.ElemKind.FP16, pm.ElemKind.FP32,
                              16, 16, 32)
        est0 = pm.estimate_perf(prob, configs[0], hw)
        est1 = pm.estimate_perf(prob, configs[1], hw)
        assert est0.predicted_tflops >= est1.predicted_tflops

    def test_unknown_arch_returns_empty(self):
        # Unknown arch → HardwareInfo has numCUs=0 → no valid candidates
        result = pick_gemm_config(4096, 4096, 4096, "fp16", "gfxUnknown")
        assert result == []


# ---------------------------------------------------------------------------
# config_to_kernel_kwargs
# ---------------------------------------------------------------------------

class TestConfigToKernelKwargs:
    def test_all_keys_present(self):
        cfg = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")[0]
        kwargs = config_to_kernel_kwargs(cfg)
        assert "BLOCK_SIZE_M"         in kwargs
        assert "BLOCK_SIZE_N"         in kwargs
        assert "BLOCK_SIZE_K"         in kwargs
        assert "matrix_instr_nonkdim" in kwargs
        assert "num_warps"            in kwargs
        assert "num_stages"           in kwargs

    def test_values_match_config(self):
        cfg = pick_gemm_config(4096, 4096, 4096, "fp16", "gfx942")[0]
        kwargs = config_to_kernel_kwargs(cfg)
        assert kwargs["BLOCK_SIZE_M"]         == cfg.block_m
        assert kwargs["BLOCK_SIZE_N"]         == cfg.block_n
        assert kwargs["BLOCK_SIZE_K"]         == cfg.block_k
        assert kwargs["matrix_instr_nonkdim"] == cfg.mfma_non_k_dim
        assert kwargs["num_warps"]            == cfg.num_warps
        assert kwargs["num_stages"]           == cfg.num_stages
