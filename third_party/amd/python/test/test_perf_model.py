"""
Unit tests for amd.perf_model Python bindings.

Run after building Triton with TRITON_BUILD_PYTHON_MODULE=ON:
  pip install -e . --no-build-isolation
  pytest third_party/amd/python/test/test_perf_model.py -v
"""
import pytest

try:
    from triton._C.libtriton import amd
    perf_model = amd.perf_model
except ImportError:
    pytest.skip("triton AMD module not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fp16_problem(M=4096, N=4096, K=4096):
    return perf_model.GemmProblem(
        M, N, K,
        perf_model.ElemKind.FP16,
        perf_model.ElemKind.FP16,
        perf_model.ElemKind.FP32,
        16, 16, 32,
    )


def gfx942():
    return perf_model.HardwareInfo.get("gfx942")


def gfx90a():
    return perf_model.HardwareInfo.get("gfx90a")


# ---------------------------------------------------------------------------
# HardwareInfo tests
# ---------------------------------------------------------------------------

class TestHardwareInfo:
    def test_get_gfx942(self):
        hw = gfx942()
        assert hw.num_cus > 0
        assert hw.num_simd_per_cu == 4
        assert hw.wave_size == 64
        assert hw.vgpr_per_simd == 256
        assert hw.lds_per_cu == 65536

    def test_get_gfx90a(self):
        hw = gfx90a()
        assert hw.num_simd_per_cu == 4
        assert hw.wave_size == 64

    def test_get_unknown_arch_returns_defaults(self):
        hw = perf_model.HardwareInfo.get("gfxUnknown")
        # Unknown arch returns a zero-filled HardwareInfo
        assert hw.num_cus == 0


# ---------------------------------------------------------------------------
# GemmProblem tests
# ---------------------------------------------------------------------------

class TestGemmProblem:
    def test_default_constructor(self):
        p = perf_model.GemmProblem()
        assert p.M == 0 and p.N == 0 and p.K == 0

    def test_full_constructor(self):
        p = fp16_problem()
        assert p.M == 4096
        assert p.a_kind == perf_model.ElemKind.FP16
        assert p.c_kind == perf_model.ElemKind.FP32

    def test_field_mutation(self):
        p = perf_model.GemmProblem()
        p.M = 128
        p.N = 256
        assert p.M == 128 and p.N == 256


# ---------------------------------------------------------------------------
# generate_candidates tests
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    def test_returns_nonempty_list(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        assert len(cands) > 0

    def test_all_have_correct_mfma_dim(self):
        # gfx942 FP16 → 16x16 (throughput tie, tiebreak wins)
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        for c in cands:
            assert c.mfma_non_k_dim == 16

    def test_block_sizes_are_multiples_of_mfma_dim(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        for c in cands:
            assert c.block_m % c.mfma_non_k_dim == 0
            assert c.block_n % c.mfma_non_k_dim == 0

    def test_repr_works(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        r = repr(cands[0])
        assert "TritonGemmConfig" in r
        assert "block_m" in r


# ---------------------------------------------------------------------------
# rank_configs tests
# ---------------------------------------------------------------------------

class TestRankConfigs:
    def test_ranked_order_best_first(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        ranked = perf_model.rank_configs(fp16_problem(), cands, gfx942())
        assert len(ranked) > 1
        est0 = perf_model.estimate_perf(fp16_problem(), ranked[0], gfx942())
        est1 = perf_model.estimate_perf(fp16_problem(), ranked[1], gfx942())
        assert est0.predicted_tflops >= est1.predicted_tflops

    def test_top_k_limits_output(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        top5 = perf_model.rank_configs(fp16_problem(), cands, gfx942(), top_k=5)
        assert len(top5) == 5

    def test_top_k_zero_returns_all(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        ranked = perf_model.rank_configs(fp16_problem(), cands, gfx942(), top_k=0)
        assert len(ranked) == len(cands)

    def test_empty_input(self):
        ranked = perf_model.rank_configs(fp16_problem(), [], gfx942())
        assert ranked == []


# ---------------------------------------------------------------------------
# estimate_perf tests
# ---------------------------------------------------------------------------

class TestEstimatePerf:
    def test_valid_config_positive_tflops(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        est = perf_model.estimate_perf(fp16_problem(), cands[0], gfx942())
        assert est.is_valid
        assert est.predicted_tflops > 0.0

    def test_estimate_fields_accessible(self):
        cands = perf_model.generate_candidates(fp16_problem(), gfx942())
        est = perf_model.estimate_perf(fp16_problem(), cands[0], gfx942())
        # All fields should be accessible without error
        _ = est.vgpr_count
        _ = est.lds_bytes
        _ = est.occupancy
        _ = est.arithmetic_intensity
        _ = est.compute_cycles
        _ = est.memory_cycles
        _ = est.pipeline_overlap
        _ = est.wave_efficiency
        _ = est.is_compute_bound
        _ = est.lds_exceeded
        _ = est.likely_spills
