//===-- PerfModelIR.cpp - IR-aware factory layer for PerfModel ------------===//
//
// Implementation of PerfModelIR.h.
//
// Each factory function reads encoding attributes and module-level attributes
// that are already present in the IR, rather than requiring callers to
// manually transcribe IR values into PerfModel structs.
//
// Key IR sources used
// ───────────────────
//  ttg::lookupNumWarps(op)
//      Reads the "ttg.num-warps" module attribute — the canonical numWarps
//      source, consistent with how AccelerateAMDMatmul.cpp reads it.
//
//  BlockedEncodingAttr  (pre-MFMA-rewrite result encoding)
//      getSizePerThread(), getWarpsPerCTA() — available on the dot result
//      before AccelerateAMDMatmul.cpp has run.
//
//  AMDMfmaEncodingAttr  (post-MFMA-rewrite result encoding)
//      getInstrShape()   → {mDim, nDim, kDim} → mfmaNonKDim = instrShape[0]
//      getWarpsPerCTA()  → product → numWarps
//
//  DotOperandEncodingAttr  (post-MFMA-rewrite operand encoding)
//      getKWidth() → kWidth  (set by AccelerateAMDMatmul.cpp lines 738-755)
//
//  ModuleAllocation::getSharedMemorySize(funcOp)
//      Actual LDS bytes including composePaddedLayout padding and layout-
//      conversion scratch (more accurate than the formula-based estimate).
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/PerfModelIR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // func::FuncOp
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"          // Float6E3M2FNType, Float4E2M1FNType, …
#include "mlir/IR/Visitors.h"               // WalkResult
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Casting.h"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::AMD::perf {

//===----------------------------------------------------------------------===//
// 1. Element-type conversion
//===----------------------------------------------------------------------===//

ElemKind elemKindFromMlirType(mlir::Type type) {
  if (type.isF64())  return ElemKind::FP64;
  if (type.isF32())  return ElemKind::FP32;
  if (type.isBF16()) return ElemKind::BF16;
  if (type.isF16())  return ElemKind::FP16;
  if (type.isInteger(8)) return ElemKind::I8;
  // FP8: method names in this LLVM version are isF8E4M3FN / isF8E5M2.
  // FNUZ variants use isa<> since they may not have dedicated isXxx() methods.
  if (type.isF8E4M3FN() || type.isF8E5M2() ||
      mlir::isa<mlir::Float8E4M3FNUZType, mlir::Float8E5M2FNUZType>(type))
    return ElemKind::FP8;
  // FP6/FP4: custom MLIR builtin types — no isXxx() methods, use isa<>.
  if (mlir::isa<mlir::Float6E3M2FNType, mlir::Float6E2M3FNType>(type))
    return ElemKind::FP6;
  if (mlir::isa<mlir::Float4E2M1FNType>(type))
    return ElemKind::FP4;
  return ElemKind::Unknown;
}

//===----------------------------------------------------------------------===//
// 2. GemmProblem factory
//===----------------------------------------------------------------------===//

GemmProblem gemmProblemFromDotOp(mlir::triton::DotOp dotOp) {
  GemmProblem prob;

  // Result tensor carries BLOCK_M × BLOCK_N.
  auto resultTy = cast<RankedTensorType>(dotOp.getResult().getType());
  auto shape    = resultTy.getShape();
  prob.M = (shape.size() >= 1) ? shape[0] : 0; // BLOCK_M
  prob.N = (shape.size() >= 2) ? shape[1] : 0; // BLOCK_N

  // A operand: [BLOCK_M, BLOCK_K] for non-transposed A.
  auto aTy   = cast<RankedTensorType>(dotOp.getA().getType());
  auto aShape = aTy.getShape();
  prob.K = (aShape.size() >= 2) ? aShape[1] : 0; // BLOCK_K

  // Element types.
  prob.aKind = elemKindFromMlirType(aTy.getElementType());
  prob.aBits = (int)aTy.getElementType().getIntOrFloatBitWidth();

  auto bTy   = cast<RankedTensorType>(dotOp.getB().getType());
  prob.bKind = elemKindFromMlirType(bTy.getElementType());
  prob.bBits = (int)bTy.getElementType().getIntOrFloatBitWidth();

  prob.cKind = elemKindFromMlirType(resultTy.getElementType());
  prob.cBits = (int)resultTy.getElementType().getIntOrFloatBitWidth();

  // batchSize: stay at 1 until batched GEMM detection is added.
  prob.batchSize = 1;

  return prob;
}

//===----------------------------------------------------------------------===//
// 3. TritonGemmConfig factories
//===----------------------------------------------------------------------===//

/// Walk the immediate parent scf.for body and return true if at least one
/// ttg::AsyncCopyGlobalToLocalOp is present.
static bool hasAsyncCopyInLoop(mlir::triton::DotOp dotOp) {
  auto *forOp = dotOp->getParentOp();
  if (!forOp)
    return false;
  bool found = false;
  forOp->walk([&](Operation *op) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Return true if LDS is bypassed: no LocalLoadOp / LocalStoreOp found in the
/// enclosing loop, meaning operands go directly from registers to MFMA.
static bool isBypassLds(mlir::triton::DotOp dotOp) {
  auto *forOp = dotOp->getParentOp();
  if (!forOp)
    return false;
  bool hasLds = false;
  forOp->walk([&](Operation *op) {
    if (isa<ttg::LocalLoadOp, ttg::LocalAllocOp>(op)) {
      hasLds = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasLds;
}

TritonGemmConfig tritonConfigFromDotOpPre(mlir::triton::DotOp dotOp,
                                          int numStages, int kPack) {
  TritonGemmConfig cfg;

  // Block sizes from tensor shapes.
  auto resultTy = cast<RankedTensorType>(dotOp.getResult().getType());
  auto shape    = resultTy.getShape();
  cfg.blockM = (shape.size() >= 1) ? (int)shape[0] : 0;
  cfg.blockN = (shape.size() >= 2) ? (int)shape[1] : 0;

  auto aTy    = cast<RankedTensorType>(dotOp.getA().getType());
  auto aShape = aTy.getShape();
  cfg.blockK = (aShape.size() >= 2) ? (int)aShape[1] : 0;

  // numWarps: read from the "ttg.num-warps" module attribute.
  // This is the canonical source used by AccelerateAMDMatmul.cpp itself
  // (ttg::lookupNumWarps is a free function in mlir::triton::gpu).
  cfg.numWarps = ttg::lookupNumWarps(dotOp);

  cfg.numStages    = numStages;
  cfg.kPack        = kPack;
  cfg.kWidth       = 0; // derived lazily from intrinsic table in deriveKWidth()
  cfg.mfmaNonKDim  = 0; // let selectMfmaNonKDim() decide
  cfg.useAsyncCopy = hasAsyncCopyInLoop(dotOp);
  cfg.bypassLds    = isBypassLds(dotOp);
  cfg.wavesPerEu   = 0;

  return cfg;
}

TritonGemmConfig tritonConfigFromDotOpPost(mlir::triton::DotOp dotOp,
                                           int numStages) {
  // Start from the pre-pass version for the fields that don't change.
  // kPack is already baked into kWidth by AccelerateAMDMatmul.cpp at this
  // point, so we pass kPack=1 and then overwrite kWidth below.
  TritonGemmConfig cfg = tritonConfigFromDotOpPre(dotOp, numStages, /*kPack=*/1);

  auto resultTy = cast<RankedTensorType>(dotOp.getResult().getType());

  // ── mfmaNonKDim and numWarps from AMDMfmaEncodingAttr ─────────────────────
  if (auto mfmaEnc =
          dyn_cast<ttg::AMDMfmaEncodingAttr>(resultTy.getEncoding())) {
    // instrShape = {mDim, nDim, kDim}; mfmaNonKDim = mDim.
    auto instrShape = mfmaEnc.getInstrShape();
    if (!instrShape.empty())
      cfg.mfmaNonKDim = (int)instrShape[0];

    // Recompute numWarps from the encoding in case it differs from the module
    // attribute (e.g. when warpsPerCTA was redistributed by planWarps()).
    auto warps = mfmaEnc.getWarpsPerCTA();
    int numWarps = 1;
    for (unsigned w : warps)
      numWarps *= (int)w;
    if (numWarps > 0)
      cfg.numWarps = numWarps;
  }

  // ── kWidth from DotOperandEncodingAttr on operand A ────────────────────────
  // AccelerateAMDMatmul.cpp stores kWidth directly in the operand encoding
  // (lines 738-755).  This is the most accurate source: it reflects the actual
  // kBase × kPack decision made by the pass, including chain-dot overrides.
  auto aOpTy = cast<RankedTensorType>(dotOp.getA().getType());
  if (auto dotEnc =
          dyn_cast<ttg::DotOperandEncodingAttr>(aOpTy.getEncoding())) {
    unsigned kw = dotEnc.getKWidth();
    if (kw > 0)
      cfg.kWidth = (int)kw;
  }

  return cfg;
}

//===----------------------------------------------------------------------===//
// 4. LDS from actual allocation
//===----------------------------------------------------------------------===//

int ldsFromAllocation(mlir::func::FuncOp funcOp,
                      mlir::ModuleAllocation &allocation,
                      const GemmProblem &prob, const TritonGemmConfig &cfg,
                      const HardwareInfo &hw) {
  // getSharedMemorySize(funcOp) returns 0 when the function has not yet been
  // processed by AllocateSharedMemory (e.g. in passes that run before it).
  // Fall back to the formula-based estimate in that case.
  size_t actual = allocation.getSharedMemorySize(funcOp);
  if (actual > 0)
    return (int)actual;

  return estimateLdsBytes(prob, cfg, hw);
}

//===----------------------------------------------------------------------===//
// 5. Hardware detection from module
//===----------------------------------------------------------------------===//

HardwareInfo hardwareInfoFromModule(mlir::ModuleOp module) {
  // Triton stores the target string as "ttg.target" on the module.
  // Example value: "hip:gfx942"  or  "hip:gfx1100"
  // The arch portion follows the colon.
  if (auto attr = module->getAttrOfType<StringAttr>("ttg.target")) {
    llvm::StringRef target = attr.getValue();
    // Strip "hip:" or "cuda:" prefix if present.
    auto colon = target.find(':');
    llvm::StringRef archStr =
        (colon != llvm::StringRef::npos) ? target.drop_front(colon + 1) : target;
    return HardwareInfo::get(archStr);
  }
  // Fallback: older Triton used "triton_gpu.compute_capability" or
  // "ttg.num-ctas" without an explicit arch string; return Unknown.
  return HardwareInfo::get(Arch::Unknown);
}

} // namespace mlir::triton::AMD::perf
