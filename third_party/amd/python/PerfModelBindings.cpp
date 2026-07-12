// Single source of truth for the AMD perf_model pybind API.
//
// The body below is the canonical binding definition. Both the in-Triton
// submodule (triton_amd.cc) and the standalone perf_model.so
// (perf_model_standalone.cpp) call registerPerfModelBindings() so there is
// exactly one copy — no hand-synced duplication.
//
// Build-time revision stamp: define PERF_MODEL_REVISION (e.g.
//   -DPERF_MODEL_REVISION="\"$(git rev-parse --short HEAD)\"")
// to record which commit a given build came from. Exposed at runtime as
// perf_model.__perf_model_revision__ so a stale .so can be detected.
#include "PerfModelBindings.h"
#include "TritonAMDGPUTransforms/PerfModel.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

#ifndef PERF_MODEL_REVISION
#define PERF_MODEL_REVISION "unknown"
#endif

namespace py = pybind11;

namespace mlir::triton::AMD::perf {

void registerPerfModelBindings(py::module_ &m) {
  using namespace mlir::triton::AMD::perf;

  // ── ElemKind enum ──────────────────────────────────────────────────────────
  py::enum_<ElemKind>(m, "ElemKind")
      .value("FP64", ElemKind::FP64)
      .value("FP32", ElemKind::FP32)
      .value("TF32", ElemKind::TF32)
      .value("FP16", ElemKind::FP16)
      .value("BF16", ElemKind::BF16)
      .value("FP8",  ElemKind::FP8)
      .value("FP6",  ElemKind::FP6)
      .value("FP4",  ElemKind::FP4)
      .value("I8",   ElemKind::I8)
      .value("Unknown", ElemKind::Unknown)
      .export_values();

  // ── KernelType ─────────────────────────────────────────────────────────────
  py::enum_<KernelType>(m, "KernelType")
      .value("Standard", KernelType::Standard)
      .value("Gluon",    KernelType::Gluon)
      .export_values();

  // ── HardwareInfo ───────────────────────────────────────────────────────────
  // All fields are precomputed at construction — expose as read-only
  // properties to avoid copies on access.
  py::class_<HardwareInfo>(m, "HardwareInfo")
      .def_static("get", [](const std::string &archStr) {
        return HardwareInfo::get(archStr);
      }, py::arg("arch_str"),
         "Construct HardwareInfo from an arch string (e.g. 'gfx942').")
      .def_readonly("num_cus",          &HardwareInfo::numCUs)
      .def_readonly("num_simd_per_cu",  &HardwareInfo::numSimdPerCU)
      .def_readonly("wave_size",        &HardwareInfo::waveSize)
      .def_readonly("vgpr_per_simd",    &HardwareInfo::vgprPerSimd)
      .def_readonly("lds_per_cu",       &HardwareInfo::ldsPerCU)
      .def_readonly("peak_mem_bw_bytes_per_cycle",
                    &HardwareInfo::peakMemBwBytesPerCycle)
      .def_readonly("clock_mhz",        &HardwareInfo::clockMHz);

  // ── GemmProblem ────────────────────────────────────────────────────────────
  py::class_<GemmProblem>(m, "GemmProblem")
      .def(py::init<>())
      .def(py::init([](int64_t M, int64_t N, int64_t K,
                       ElemKind aKind, ElemKind bKind, ElemKind cKind,
                       int aBits, int bBits, int cBits) {
        GemmProblem p;
        p.M = M; p.N = N; p.K = K;
        p.aKind = aKind; p.bKind = bKind; p.cKind = cKind;
        p.aBits = aBits; p.bBits = bBits; p.cBits = cBits;
        return p;
      }), py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("a_kind") = ElemKind::FP16,
          py::arg("b_kind") = ElemKind::FP16,
          py::arg("c_kind") = ElemKind::FP32,
          py::arg("a_bits") = 16,
          py::arg("b_bits") = 16,
          py::arg("c_bits") = 32)
      .def_readwrite("M",      &GemmProblem::M)
      .def_readwrite("N",      &GemmProblem::N)
      .def_readwrite("K",      &GemmProblem::K)
      .def_readwrite("a_kind", &GemmProblem::aKind)
      .def_readwrite("b_kind", &GemmProblem::bKind)
      .def_readwrite("c_kind", &GemmProblem::cKind)
      .def_readwrite("a_bits", &GemmProblem::aBits)
      .def_readwrite("b_bits", &GemmProblem::bBits)
      .def_readwrite("c_bits", &GemmProblem::cBits);

  // ── TritonGemmConfig ───────────────────────────────────────────────────────
  py::class_<TritonGemmConfig>(m, "TritonGemmConfig")
      .def(py::init<>())
      .def_readwrite("block_m",        &TritonGemmConfig::blockM)
      .def_readwrite("block_n",        &TritonGemmConfig::blockN)
      .def_readwrite("block_k",        &TritonGemmConfig::blockK)
      .def_readwrite("num_stages",     &TritonGemmConfig::numStages)
      .def_readwrite("num_warps",      &TritonGemmConfig::numWarps)
      .def_readwrite("mfma_non_k_dim", &TritonGemmConfig::mfmaNonKDim)
      .def_readwrite("k_width",        &TritonGemmConfig::kWidth)
      .def_readwrite("bypass_lds",     &TritonGemmConfig::bypassLds)
      .def_readwrite("use_async_copy", &TritonGemmConfig::useAsyncCopy)
      .def_readwrite("k_pack",         &TritonGemmConfig::kPack)
      .def_readwrite("waves_per_eu",   &TritonGemmConfig::wavesPerEu)
      .def_readwrite("group_size_m",   &TritonGemmConfig::groupSizeM)
      .def("__repr__", [](const TritonGemmConfig &c) {
        return "TritonGemmConfig(block_m=" + std::to_string(c.blockM) +
               ", block_n=" + std::to_string(c.blockN) +
               ", block_k=" + std::to_string(c.blockK) +
               ", num_warps=" + std::to_string(c.numWarps) +
               ", num_stages=" + std::to_string(c.numStages) +
               ", mfma_non_k_dim=" + std::to_string(c.mfmaNonKDim) + ")";
      });

  // ── PerfEstimate ───────────────────────────────────────────────────────────
  // Read-only — it's a result struct returned by estimate_perf().
  py::class_<PerfEstimate>(m, "PerfEstimate")
      .def_readonly("predicted_tflops",    &PerfEstimate::predictedTflops)
      .def_readonly("is_valid",            &PerfEstimate::isValid)
      .def_readonly("is_compute_bound",    &PerfEstimate::isComputeBound)
      .def_readonly("vgpr_count",          &PerfEstimate::vgprCount)
      .def_readonly("waves_per_eu",        &PerfEstimate::wavesPerSimd)
      .def_readonly("lds_bytes",           &PerfEstimate::ldsBytes)
      .def_readonly("occupancy",           &PerfEstimate::occupancy)
      .def_readonly("arithmetic_intensity",&PerfEstimate::arithmeticIntensity)
      .def_readonly("compute_cycles",      &PerfEstimate::computeCycles)
      .def_readonly("memory_cycles",       &PerfEstimate::memoryCycles)
      .def_readonly("lds_cycles",          &PerfEstimate::ldsCycles)
      .def_readonly("pipeline_overlap",    &PerfEstimate::pipelineOverlap)
      .def_readonly("wave_efficiency",     &PerfEstimate::waveEfficiency)
      .def_readonly("lds_exceeded",        &PerfEstimate::ldsExceeded)
      .def_readonly("likely_spills",       &PerfEstimate::likelySpills)
      .def_readonly("vgpr_count",          &PerfEstimate::vgprCount)
      .def_readonly("num_buffers",          &PerfEstimate::numBuffers)
      .def_readonly("waves_per_simd",       &PerfEstimate::wavesPerSimd)
      .def_readonly("ctas_per_cu",          &PerfEstimate::ctasPerCU)
      .def_readonly("total_output_tiles",   &PerfEstimate::totalOutputTiles)
      .def_readonly("num_waves",            &PerfEstimate::numWaves)
      .def_readonly("effective_tile_cycles",&PerfEstimate::effectiveTileCycles);

  // ── Free functions ─────────────────────────────────────────────────────────
  // generate_candidates: returns vector by value; pybind11 converts to list.
  // topK=0 means return all ranked configs (uses stable_sort).
  // topK>0 uses partial_sort — O(N log K) instead of O(N log N).
  m.def("generate_candidates",
        [](const GemmProblem &prob, const HardwareInfo &hw,
           KernelType kernelType) {
          return generateCandidates(prob, hw, kernelType);
        },
        py::arg("prob"), py::arg("hw"),
        py::arg("kernel_type") = KernelType::Standard,
        "Generate feasible TritonGemmConfig candidates. "
        "kernel_type='Standard' sweeps full numWarps/numStages range; "
        "'Gluon' constrains to numWarps=4, numStages=2, BM/BN multiples of 128, "
        "and K%(2*BK)==0 for v9-style 4-quadrant pipelined kernels.");

  m.def("rank_configs",
        [](const GemmProblem &prob,
           const std::vector<TritonGemmConfig> &configs,
           const HardwareInfo &hw,
           size_t topK) {
          return rankConfigs(prob, configs, hw, topK);
        },
        py::arg("prob"), py::arg("configs"), py::arg("hw"),
        py::arg("top_k") = 0,
        "Sort configs by predicted TFLOPS (best first). "
        "LDS-overflowing configs are excluded. "
        "top_k=0 returns all; top_k>0 uses partial_sort for O(N log K).");

  m.def("estimate_perf",
        [](const GemmProblem &prob, const TritonGemmConfig &cfg,
           const HardwareInfo &hw) {
          return estimatePerf(prob, cfg, hw);
        },
        py::arg("prob"), py::arg("cfg"), py::arg("hw"),
        "Full analytical performance estimate (roofline + wave quantisation).");

  m.def("select_group_size_m",
        [](const GemmProblem &prob, const TritonGemmConfig &cfg,
           const HardwareInfo &hw) {
          return selectGroupSizeM(prob, cfg, hw);
        },
        py::arg("prob"), py::arg("cfg"), py::arg("hw"),
        "Select GROUP_SIZE_M using Origami's WGM prediction algorithm.");

  // ── Build provenance ─────────────────────────────────────────────────────────
  m.attr("__perf_model_revision__") = PERF_MODEL_REVISION;
}

} // namespace mlir::triton::AMD::perf
