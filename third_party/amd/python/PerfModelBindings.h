#ifndef TRITON_AMD_PERF_MODEL_BINDINGS_H
#define TRITON_AMD_PERF_MODEL_BINDINGS_H

#include <pybind11/pybind11.h>

namespace mlir::triton::AMD::perf {

// Register the perf-model pybind API onto module `m`:
//   ElemKind, KernelType, HardwareInfo, GemmProblem, TritonGemmConfig,
//   PerfEstimate, generate_candidates, rank_configs, estimate_perf,
//   select_group_size_m, and the __perf_model_revision__ attribute.
//
// This is the single source of truth for the bindings, shared by:
//   - the in-Triton submodule  (triton_amd.cc: init_triton_amd_perf_model)
//   - the standalone .so       (perf_model_standalone.cpp: PYBIND11_MODULE)
//
// It exposes ONLY the C++ API surface from PerfModel.h. It contains no
// deployment, path, or loading logic — the Python selector decides whether to
// import the in-Triton submodule or the standalone .so.
void registerPerfModelBindings(pybind11::module_ &m);

} // namespace mlir::triton::AMD::perf

#endif // TRITON_AMD_PERF_MODEL_BINDINGS_H
