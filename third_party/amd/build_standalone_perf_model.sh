#!/usr/bin/env bash
# Build the standalone perf_model.so from:
#   python/PerfModelBindings.cpp  (single-source pybind API, shared w/ triton_amd.cc)
#   perf_model_standalone.cpp     (thin PYBIND11_MODULE wrapper)
#   lib/TritonAMDGPUTransforms/PerfModel.cpp  (the cost model itself)
#
# Links only LLVMSupport/Demangle (no MLIR, no libtriton), so the resulting .so
# imports on a stock Triton that lacks triton._C.libtriton.amd.perf_model.
#
# IMPORTANT: run this INSIDE the target serving container. The pybind11 and
# CPython ABI are baked into the .so, so it must be built against the same
# Python/pybind11 the server uses. Building elsewhere risks import-time ABI errors.
#
# Overridable via env: LLVM_DIR, PYBIND11_INCLUDE, PYTHON_INCLUDE, OUT, CXX,
#                      PERF_MODEL_REVISION.
set -euo pipefail

AMD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CXX="${CXX:-clang++}"
OUT="${OUT:-$AMD_DIR/perf_model.so}"

# --- LLVM (headers + LLVMSupport/LLVMDemangle) --------------------------------
# Default to the LLVM that Triton downloads under <home>/.triton/llvm/*. Search
# $HOME plus /root and /home/* since the container HOME may differ from where the
# .triton cache actually lives.
if [[ -z "${LLVM_DIR:-}" ]]; then
  for _root in "${HOME}" /root /home/*; do
    _cand="$(ls -d "${_root}"/.triton/llvm/llvm-* 2>/dev/null | head -1 || true)"
    if [[ -n "$_cand" && -d "$_cand/include" && -d "$_cand/lib" ]]; then
      LLVM_DIR="$_cand"; break
    fi
  done
fi
if [[ -z "${LLVM_DIR:-}" || ! -d "$LLVM_DIR/include" ]]; then
  echo "ERROR: LLVM_DIR not found. Set LLVM_DIR to a dir with include/ and lib/." >&2
  exit 1
fi

# --- pybind11 headers ---------------------------------------------------------
if [[ -z "${PYBIND11_INCLUDE:-}" ]]; then
  PYBIND11_INCLUDE="$(python3 -c 'import pybind11; print(pybind11.get_include())' 2>/dev/null || true)"
fi
[[ -n "${PYBIND11_INCLUDE:-}" ]] || { echo "ERROR: set PYBIND11_INCLUDE (pip install pybind11)." >&2; exit 1; }

# --- CPython headers ----------------------------------------------------------
if [[ -z "${PYTHON_INCLUDE:-}" ]]; then
  PYTHON_INCLUDE="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))' 2>/dev/null || true)"
fi
[[ -n "${PYTHON_INCLUDE:-}" ]] || { echo "ERROR: set PYTHON_INCLUDE." >&2; exit 1; }

# --- revision stamp -----------------------------------------------------------
if [[ -z "${PERF_MODEL_REVISION:-}" ]]; then
  PERF_MODEL_REVISION="$(git -C "$AMD_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi

# --- LLVM link flags ----------------------------------------------------------
# Prefer llvm-config: it resolves the exact archive names AND the system libs
# that static LLVMSupport pulls in (pthread, dl, z, tinfo, m, rt, ...). Falling
# back to a hand-list risks undefined symbols on lean containers.
LLVM_CONFIG="${LLVM_CONFIG:-$LLVM_DIR/bin/llvm-config}"
LLVM_LINK_FLAGS=()
RPATH_FLAGS=()
if [[ -x "$LLVM_CONFIG" ]]; then
  # shellcheck disable=SC2207
  LLVM_LINK_FLAGS=( $("$LLVM_CONFIG" --ldflags --libs support demangle --system-libs) )
  # Only need an rpath if LLVM is packaged as shared libs (this dist is static).
  if [[ "$("$LLVM_CONFIG" --shared-mode 2>/dev/null)" == "shared" ]]; then
    RPATH_FLAGS=( -Wl,-rpath,"$("$LLVM_CONFIG" --libdir)" )
  fi
else
  echo "WARN: llvm-config not found at $LLVM_CONFIG; using a best-effort lib list." >&2
  LLVM_LINK_FLAGS=( -L"$LLVM_DIR/lib" -lLLVMSupport -lLLVMDemangle -lpthread -ldl -lm )
fi

echo "CXX               = $CXX"
echo "LLVM_DIR          = $LLVM_DIR"
echo "LLVM_CONFIG       = $LLVM_CONFIG"
echo "LLVM_LINK_FLAGS   = ${LLVM_LINK_FLAGS[*]}"
echo "RPATH_FLAGS       = ${RPATH_FLAGS[*]:-<none (static LLVM)>}"
echo "PYBIND11_INCLUDE  = $PYBIND11_INCLUDE"
echo "PYTHON_INCLUDE    = $PYTHON_INCLUDE"
echo "PERF_MODEL_REVISION = $PERF_MODEL_REVISION"
echo "OUT               = $OUT"

set -x
"$CXX" -O2 -shared -fPIC -std=gnu++17 \
  -DPERF_MODEL_REVISION="\"$PERF_MODEL_REVISION\"" \
  -I"$LLVM_DIR/include" \
  -I"$AMD_DIR/include" \
  -I"$AMD_DIR/python" \
  -I"$PYBIND11_INCLUDE" \
  -I"$PYTHON_INCLUDE" \
  "$AMD_DIR/perf_model_standalone.cpp" \
  "$AMD_DIR/python/PerfModelBindings.cpp" \
  "$AMD_DIR/lib/TritonAMDGPUTransforms/PerfModel.cpp" \
  "${LLVM_LINK_FLAGS[@]}" \
  "${RPATH_FLAGS[@]}" \
  -o "$OUT"
set +x

echo "built $OUT"
