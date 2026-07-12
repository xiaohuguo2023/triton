// Standalone pybind module for the AMD perf_model API.
//
// Links only PerfModelBindings.cpp + PerfModel.cpp (no MLIR / libtriton), so it
// can be built as a small perf_model.so and imported on a stock Triton that has
// no triton._C.libtriton.amd.perf_model submodule (e.g. the serving container).
//
// The bindings themselves live in PerfModelBindings.cpp — the same definition
// the in-Triton submodule (triton_amd.cc) uses — so there is no duplication.
//
// Build with third_party/amd/build_standalone_perf_model.sh.
#include "PerfModelBindings.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(perf_model, m) {
  m.doc() = "Standalone AMD Triton perf-model bindings (no libtriton).";
  mlir::triton::AMD::perf::registerPerfModelBindings(m);
}
