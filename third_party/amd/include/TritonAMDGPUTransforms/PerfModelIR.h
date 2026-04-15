//===-- PerfModelIR.h - IR-aware factory layer for PerfModel ----*- C++ -*-===//
//
// Companion to PerfModel.h.  Provides factory functions that populate
// GemmProblem and TritonGemmConfig directly from Triton MLIR operations and
// the analyses that are already available in the AMD transform passes.
//
// Design
// ──────
// PerfModel.h is deliberately IR-agnostic so that it can be used from Python
// bindings, standalone tests, and future non-GEMM characterisers without
// pulling in the full MLIR/Triton dialect stack.
//
// This file is the thin IR-aware bridge.  It reads encoding attributes and
// calls the existing Triton analyses rather than asking callers to transcribe
// IR values into struct fields by hand.
//
// Usage by pass
// ─────────────
// AccelerateAMDMatmul.cpp  (before MFMA rewrite)
//   auto prob = perf::gemmProblemFromDotOp(dotOp);
//   auto cfg  = perf::tritonConfigFromDotOpPre(dotOp, numStages, kPack);
//   int  dim  = perf::selectMfmaNonKDim(prob, cfg, hw);
//
// LowerLoops.cpp  (after MFMA rewrite, before allocation)
//   auto prob = perf::gemmProblemFromDotOp(dotOp);
//   auto cfg  = perf::tritonConfigFromDotOpPost(dotOp, numStages);
//   if (!perf::isValidConfig(prob, cfg, hw)) { ... }
//
// Any pass after AllocateSharedMemory
//   int actualLds = perf::ldsFromAllocation(funcOp, moduleAlloc);
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODELAIR_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODELAIR_H_

#include "TritonAMDGPUTransforms/PerfModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD::perf {

//===----------------------------------------------------------------------===//
// Element-type conversion
//===----------------------------------------------------------------------===//

/// Convert an MLIR element type (f16, bf16, f32, f8E4M3FN, i8, …) to the
/// coarse ElemKind used by the throughput table.
ElemKind elemKindFromMlirType(mlir::Type type);

//===----------------------------------------------------------------------===//
// GemmProblem factory
//===----------------------------------------------------------------------===//

/// Build a GemmProblem from a tt.dot operation.
///
/// The M / N / K fields are set to the *tile* (BLOCK) sizes taken from the
/// operand tensor shapes — not the full problem dimensions.  Callers that need
/// the true M/N/K (for wave-quantisation accuracy) should multiply by the grid
/// dimensions obtained from program_id range analysis.
///
/// Fields populated:
///   M       ← shape(result)[0]        (= BLOCK_M)
///   N       ← shape(result)[1]        (= BLOCK_N)
///   K       ← shape(A)[1]             (= BLOCK_K, assuming non-transposed A)
///   aKind / aBits ← A element type
///   bKind / bBits ← B element type
///   cKind / cBits ← result element type
///   batchSize     ← 1  (batched GEMM support can be added later)
GemmProblem gemmProblemFromDotOp(mlir::triton::DotOp dotOp);

//===----------------------------------------------------------------------===//
// TritonGemmConfig factories
//===----------------------------------------------------------------------===//

/// Build a TritonGemmConfig from a tt.dot op **before** AccelerateAMDMatmul
/// has run (i.e. the result still has BlockedEncodingAttr).
///
/// Fields populated from the IR:
///   blockM / blockN  ← shape(result)[0..1]
///   blockK           ← shape(A)[1]
///   numWarps         ← ttg::lookupNumWarps(dotOp)   [reads "ttg.num-warps"]
///   useAsyncCopy     ← walk parent scf.for for ttg::AsyncCopyGlobalToLocalOp
///   bypassLds        ← walk parent scf.for for absence of LDS ops
///
/// Fields provided by the caller (not yet known from the IR at this stage):
///   numStages        ← pass option
///   kPack            ← pass option (1 or 2; default 1)
///   mfmaNonKDim      ← 0 (let selectMfmaNonKDim() decide)
///   kWidth           ← 0 (deriveKWidth() will compute from intrinsic table)
TritonGemmConfig tritonConfigFromDotOpPre(mlir::triton::DotOp dotOp,
                                          int numStages, int kPack = 1);

/// Build a TritonGemmConfig from a tt.dot op **after** AccelerateAMDMatmul has
/// run (result has AMDMfmaEncodingAttr, operands have DotOperandEncodingAttr).
///
/// Additional fields populated compared to the pre-pass version:
///   kWidth       ← DotOperandEncodingAttr::getKWidth() on operand A type
///   mfmaNonKDim  ← AMDMfmaEncodingAttr::getInstrShape()[0] on result type
///   numWarps     ← product(AMDMfmaEncodingAttr::getWarpsPerCTA())
///
/// The post-pass version is the most accurate and should be used in passes
/// that run after AccelerateAMDMatmul (e.g. LowerLoops, BlockPingpong).
TritonGemmConfig tritonConfigFromDotOpPost(mlir::triton::DotOp dotOp,
                                           int numStages);

//===----------------------------------------------------------------------===//
// LDS query from actual allocation
//===----------------------------------------------------------------------===//

/// Return the actual shared-memory bytes allocated for funcOp as computed by
/// AllocateSharedMemory.  More accurate than estimateLdsBytes() because it
/// includes architecture-specific padding from composePaddedLayout() and layout-
/// conversion scratch buffers that the formula-based estimate does not model.
///
/// Falls back to estimateLdsBytes(prob, cfg, hw) if the function has no
/// recorded allocation (e.g. when called before AllocateSharedMemory runs).
int ldsFromAllocation(mlir::func::FuncOp funcOp,
                      mlir::ModuleAllocation &allocation,
                      const GemmProblem &prob, const TritonGemmConfig &cfg,
                      const HardwareInfo &hw);

//===----------------------------------------------------------------------===//
// Hardware detection from module
//===----------------------------------------------------------------------===//

/// Read the AMD target architecture string from the module's "ttg.target"
/// attribute and return a populated HardwareInfo.
/// Falls back to HardwareInfo::get(Arch::Unknown) for unrecognised targets.
HardwareInfo hardwareInfoFromModule(mlir::ModuleOp module);

} // namespace mlir::triton::AMD::perf

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PERFMODELAIR_H_
