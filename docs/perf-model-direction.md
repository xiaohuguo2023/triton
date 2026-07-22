# Performance Model: Architectural Direction Discussion

## Background

The current `PerfModel` in `third_party/amd/lib/TritonAMDGPUTransforms/` is a
three-layer analytical model (hardware database → resource accounting →
roofline + wave quantisation) that was introduced to replace hard-coded
empirical thresholds in AMD GPU transform passes.

Today it is used in three places inside the compiler:
- `AccelerateAMDMatmul` — pick 16×16 vs 32×32 MFMA via `selectMfmaNonKDim`
- `BlockPingpong` — classify compute-bound vs memory-bound via `estimatePerf`
- `LowerLoops` — LDS overflow guard via `estimateLdsBytes`

`rankConfigs` (sort a candidate list by predicted TFLOPS) is fully implemented
but has zero callers. That gap is the starting point for this discussion.

---

## The Core Tension

There are two fundamentally different things a performance model can do:

**Compile-time decisions** — given a fixed kernel config (blockM, blockN, …
already decided), make better code-generation choices inside the compiler
(MFMA tile size, scheduling, LDS layout). This is what the current model does.

**Pre-compile config selection** — given a kernel and a hardware target, choose
*which* config to compile in the first place, replacing or pruning the autotuner
search space before any benchmarking happens.

The second is much more valuable but much harder. The rest of this document
discusses the design space for pursuing it.

---

## Option A: Extend the Current Approach (Stay Inside Triton)

### What it means

Add a config-generation step inside Triton's compilation pipeline. Before
lowering a `tt.func`, enumerate candidate `(blockM, blockN, blockK, numStages,
numWarps)` tuples, score each with `estimatePerf`, and compile only the top-K.

```
tt.func (with tile attrs undecided)
  └── PerfModel.generateCandidates(problem, hw)   → [TritonGemmConfig × N]
  └── PerfModel.rankConfigs(problem, configs, hw) → top-K configs
  └── compile each → pick best at runtime or benchmark
```

The autotuner still runs but over a much smaller search space (e.g. top 5
model-predicted configs instead of 50 hand-written ones).

### Strengths
- Stays inside the existing Triton + MLIR build system
- No new library boundary or ABI to maintain
- Config generation is purely analytical — no GPU benchmarking needed
- Natural extension of what `rankConfigs` already does

### Weaknesses
- Model is GEMM-specific today; generalising to arbitrary kernels requires
  significant new modelling work (see "Beyond GEMM" below)
- Triton's tile size is currently user-specified via `tl.constexpr` — the
  compiler doesn't own that decision today; changing this requires a new
  abstraction
- Tight coupling: model changes require rebuilding the compiler

### Verdict
Good for short-term impact (prune autotuner search space), but hits a ceiling
quickly when the kernel is not a GEMM.

---

## Option B: Separate MLIR-Based Library (`libtriton-perf`)

### What it means

Extract `PerfModel.cpp` (and future additions) into a standalone MLIR library
that is independently versioned and linkable. Triton, ROCm libraries (MIOpen,
rocBLAS), and third-party frameworks can all consume it.

```
┌─────────────────────────────────┐
│  libtriton-perf (MLIR library)  │
│  - Hardware database            │
│  - Resource accounting          │
│  - Roofline model               │
│  - Config ranking               │
│  - (future) Kernel cost model   │
└──────────┬──────────────────────┘
           │ links
  ┌────────┴───────┐  ┌──────────────┐  ┌──────────┐
  │ Triton compiler│  │ MIOpen/GEMM  │  │ User app │
  └────────────────┘  └──────────────┘  └──────────┘
```

Analogous to what Origami already is for ROCm libraries, but MLIR-native and
kernel-language-agnostic.

### Strengths
- Single source of truth for AMD GPU performance knowledge, shared across the
  ROCm software stack
- MLIR IR as the common language: any MLIR-based kernel (Triton, Linalg,
  IREE, custom dialects) can be analysed
- Can evolve independently from the Triton compiler release cycle
- Python bindings (`pybind11` / `nanobind`) are a natural addition without
  polluting the compiler

### Weaknesses
- New library to maintain, version, and distribute
- ABI stability burden (especially for C++ MLIR types)
- Requires consensus from the ROCm community (not just the Triton team)
- Duplication risk with Origami unless the relationship is clearly defined

### Verdict
The right long-term architecture. Origami is the closest existing analogue but
is C++-only and GEMM-specific. A MLIR-native version could subsume it.

---

## Option C: Replace Triton Autotuner

### What it means

Instead of benchmarking configs at runtime, use the analytical model to predict
the best config at compile time and compile only that one. Zero autotuning
overhead, deterministic compilation.

```python
# Today
@triton.autotune(configs=[...50 configs...], key=["M", "N", "K"])
def matmul_kernel(...): ...

# With model-driven selection
@triton.modelselect(hardware="gfx942")   # <-- new decorator
def matmul_kernel(...): ...
# compiler picks config via PerfModel, compiles once
```

### Strengths
- Eliminates the JIT compilation latency caused by autotuning (significant for
  small batch or interactive workloads)
- Reproducible: same hardware + same kernel → same config every time
- Necessary for kernels that cannot be autotuned (e.g. online inference,
  safety-critical code)

### Weaknesses
- Model accuracy is the critical path: a wrong prediction is worse than a
  slow autotune (no fallback)
- The model needs to match real hardware throughput very closely, which
  requires calibration data per GPU SKU
- Not all kernels are GEMM-shaped; the model must generalise (see below)
- Autotuning is often the right answer for production code where compile
  latency is paid once; replacing it entirely may not be worth the risk

### Verdict
A long-term goal, not a near-term replacement. More realistic as a
`prune_configs_by` accelerator that reduces the autotuner search space from
50 configs to 5, not as a full replacement.

---

## Beyond GEMM: Generalising to Arbitrary Kernels

This is the hardest open problem. Today `GemmProblem` hardcodes `M, N, K,
aKind, bKind, cKind`. Real workloads include:

| Kernel type | New modelling challenges |
|---|---|
| **Flash Attention** | Two chained GEMMs + softmax; memory hierarchy reuse between Q×K and P×V |
| **Elementwise / pointwise** | No MFMA; dominated by memory bandwidth |
| **Reduction (softmax, layernorm)** | Multi-pass; LDS used for partial sums not operand tiles |
| **Sparse / irregular** | Tile occupancy varies; standard roofline does not apply |
| **Mixed-precision** | Multiple dtypes in one kernel; accumulator reuse patterns change |
| **Multi-kernel fusions** | Need to model register/LDS state *across* kernel boundaries |

### Approaches to generalisation

**1. Kernel cost model from MLIR IR**

Walk the MLIR ops in a Triton function and accumulate:
- MFMA instruction count → compute cycles
- Global load/store count × element size → memory cycles
- LDS allocation → occupancy limit

This is essentially a static op-count analysis. It is kernel-agnostic but
requires the IR to be in a lowered enough form (after tiling, before codegen).

```
tt.func @attention_fwd(...) {
  // PerfModel walks this IR, counts mfma ops, load/store bytes, LDS allocs
  // Returns PerfEstimate
}
```

**2. Parameterised kernel templates**

Define a small vocabulary of kernel shapes (GEMM, batched GEMM, attention,
convolution) and have the model handle each explicitly, as Origami does. More
accurate but requires manual effort per kernel family.

**3. ML-based surrogate model**

Train a small model (gradient-boosted tree or simple MLP) on (kernel IR
features, hardware, config) → measured throughput. The analytical model
provides the feature engineering; the ML model calibrates the predictions.
This approach is used in products like TVM's Ansor/MetaSchedule.

### Recommendation

For Triton's near term, approach 1 (MLIR IR walk) generalises the model to
any Triton kernel without adding per-kernel code. The accuracy will be lower
than the GEMM-specific model for non-GEMM kernels, but it gives a useful
signal for pruning the autotuner search space.

The `PerfModelIR.h` layer (`gemmProblemFromDotOp`, `hardwareInfoFromModule`)
is already the right abstraction point: extend it with a more general
`KernelCostModel` that counts ops from the MLIR module rather than assuming
a GEMM structure.

A fourth approach — DSL-driven templates — is discussed in the Related Work
section below (ThroughputSolver), with concrete ideas worth borrowing.

---

## Related Work: ThroughputSolver

ThroughputSolver (https://github.com/AMD-internal/ThroughputSolver, C++) is a
single-workgroup symbolic cost-function generator. Two text DSLs (arch.txt for
hardware, kernel.txt for the kernel description) feed a dependency-graph engine
that emits a *readable Python cost function*.

This is a different design point from our PerfModel, with both strengths and
gaps worth understanding before we generalise beyond GEMM.

### What it is, concretely

**Architecture DSL** (`test/arch.txt`):
```
MemoryPath hbm lds 512 500      # bw=512 B/cycle, latency=500 cycles
MemoryPath lds vgpr 512 40
Matrix FP16 FP16 FP32 32768     # MFMA throughput
Coexecution Matrix FP16 FP16 FP32 3 4   # 75% vector/matrix overlap
```

**Kernel DSL** (`test/gemm.txt`):
```
For itrs
  A_lds = MatrixMove M K hbm lds FP16 4 <ALoopLoadLds>
  B_lds = MatrixMove K N hbm lds FP16 4 <BLoopLoadLds>
  A = MatrixMove M K lds vgpr FP16 2 <ALoopLoadVgpr>
  B = MatrixMove K N lds vgpr FP16 2 <BLoopLoadVgpr>
  S = Mma A B C FP16 FP16 FP32 <Mma>
EndFor
Wait all
```

**Engine internals**:
- Dependency DAG with explicit `producers/consumers` (data deps) and
  `pipelineProducers` (hardware pipeline deps).
- Symbolic parameter algebra (`Fixed`, `Optimisable`, `{Div, Sub, Min, Max, Sum, Prod}`
  composites).
- Output: a Python function that takes optimisable parameters as arguments
  and returns the cycle count. The function is human-readable — you can see
  exactly which path the critical-path analysis picks for each operation.

Their flash-attention example (`test/fwd_atten.txt`) describes the full forward
pass — Q/K/V loads, QK^T, softmax (Sub/Exp/Mul), PV, write-out — in ~30 lines
of DSL.

### What's worth borrowing

| Idea | Benefit for our PerfModel | Effort |
|---|---|---|
| **Symbolic cost function as output** | Today `estimatePerf()` returns one `PerfEstimate{}`. A parallel symbolic-tree mode would let us debug *why* config A beats B, share readable formulas with users, and enable sensitivity analysis (change one param, see propagation). | Medium — wrap existing C++ formulas in a symbolic builder; keep numeric path for callers that don't need it. |
| **Kernel-description DSL** | Adding attention/conv/MoE today requires C++ code in `generateCandidates` + a new `KernelType`. A DSL would let us describe new kernel patterns in ~30 lines of text. Pairs naturally with generalisation approach (2) above. | Large — parser + DAG builder + integration with our wave-quant layer. |
| **Architecture DSL** | Adding MI400 today means editing `HardwareInfo::get()` in C++. A DSL would let us add an arch by editing a text file. | Medium — `HardwareInfo` is already a struct; needs a parser. |
| **Explicit pipeline DAG with `pipelineProducers`** | Today our model is closed-form (roofline + corrections). A DAG simulation would capture async-copy pipelines + `s_waitcnt` placement more precisely. | Large — would replace formulas with a simulator; not clear it's worth the rewrite given our corrections already capture most of the effect. |
| **`Coexecution` ratios** | Today MFMA-vs-vector overlap is implicit in the roofline. Explicit ratios in `HardwareInfo` would be cleaner and easier to recalibrate per arch. | Small — add a co-execution table. |
| **`Wait`-as-node modelling** | Today `s_waitcnt` is counted *after* compilation, from AMDGCN. ThroughputSolver models `Wait` as a first-class DAG node. Could improve predictions for kernels with explicit synchronization patterns. | Large. |

### What ThroughputSolver doesn't do (and we do)

ThroughputSolver is a **single-workgroup critical-path model**. It does not model:

- Wave-quantization / occupancy correction (our `stallAmp`, `wavesPerSimd` math).
- Bank conflicts and LDS-padding accounting.
- Multi-XCD L2 cache simulation (our `formocastL2HitRate`).
- TCP-cap pipeline modelling (our Gluon-specific single-chain model).
- VGPR pressure estimation — there is no register model at all.
- MFMA-specific quirks (kBase, kWidth, dot-operand layout).

These are precisely the GPU-level effects that determine ranking accuracy in
our domain. ThroughputSolver's strength is per-workgroup latency analysis from
pseudocode; ours is GPU-scale ranking from a fixed kernel family.

### Recommendation: augment, don't replace

The two systems are complementary, not competing. A pragmatic integration path:

1. **Adopt symbolic-output mode first** (smallest change, highest debuggability
   payoff). Wrap our existing `estimatePerf()` formulas so they can emit either
   a `PerfEstimate` struct (today's behaviour) or a symbolic Python function
   (ThroughputSolver-style). Same math; the debug mode just makes the math
   inspectable.

2. **Adopt the kernel + arch DSLs for generalisation** (the larger investment,
   tied to approach (2) "Parameterised kernel templates" above). When we
   extend beyond GEMM, describe each new kernel pattern as a DSL file. Our
   existing C++ engine becomes the GPU-level wrapper (occupancy, multi-XCD
   L2, TCP cap); the DSL handles per-kernel critical-path enumeration.

3. **Borrow `Coexecution` ratios opportunistically** (small change, modest gain).
   Add an explicit co-exec table to `HardwareInfo` to clean up the MFMA-vs-vector
   overlap modelling that today is buried in roofline math.

We should not adopt the dependency-DAG simulator wholesale. Our closed-form
corrections already capture most of what the DAG simulator would express, and
the rewrite cost would exceed the marginal accuracy gain.

### Open questions

- Where does the boundary live between ThroughputSolver's per-workgroup model
  and our GPU-scale layer? Does the DSL output need to expose registers and
  LDS occupancy hints, or can we infer those from the DAG?
- Is the parameter algebra (Div / Sub / Min / Max / Sum / Prod) expressive
  enough to encode wave-quantization corrections, or do we need to extend it?
- Can ThroughputSolver itself be extended to model multi-CU effects, or does
  it inherently assume single-WG? If the former, the integration is much
  cheaper (just port it).

---

## Multiple Triton Kernels and Kernel Fusion

When multiple Triton kernels are fused (e.g. attention = QK matmul + softmax +
PV matmul compiled as one kernel), the performance model must reason about:

1. **Register live-ranges across fused regions**: the accumulator of the first
   GEMM becomes the input of softmax; VGPR pressure is higher.
2. **LDS reuse**: can the softmax intermediate live in LDS and be reused by the
   second GEMM without a round-trip to global memory?
3. **Warp specialisation**: should some warps handle compute while others
   handle memory for the combined kernel?

This is currently out of scope for `PerfModel.cpp` but is a natural extension
if the model operates on the full fused MLIR module rather than a single
`tt.dot` op.

A multi-kernel cost model would look like:

```
PerfEstimate estimateKernelGraph(
  ModuleOp fusedKernel,       // the full fused MLIR module
  KernelConfig config,        // tile sizes for each fused sub-kernel
  HardwareInfo hw
);
```

---

## Proposed Near-Term Roadmap

Given the above, here is a concrete sequence that builds toward the bigger
vision without over-committing:

### Phase 1 (compiler-internal, current branch)
- [x] Replace VGPR-gated MFMA selection with Origami throughput logic
- [x] Unit tests (`AMDPerfModel`) and lit tests
- [x] **Cost-model accuracy (2026-07): saturation-physics rewrite.** The roofline
      now derives block-K, num_warps, num_stages, and occupancy from Little's-law
      bytes-in-flight + clean-wave MFMA efficiency (not tie-breaks). Config-
      selection accuracy on a 471-shape fp16 sweep: **83% → 93% win vs Triton
      autotune, geomean ~1.44×.** See
      [`perf-model-saturation-physics.md`](perf-model-saturation-physics.md).
- [ ] Priority 3: integration regression tests (BlockPingpong, LowerLoops)
- [ ] Wire `rankConfigs` into `AccelerateAMDMatmul` to pick the best config
      from a small enumerated set (e.g. top-3 tile sizes) at compile time

### Phase 2 (autotuner pruning)
- [ ] Add Python bindings for `estimatePerf` and `rankConfigs`
      (thin `pybind11` wrapper, no new library boundary yet)
- [ ] Plug into `prune_configs_by={"perf_model": amd_perf_model, "top_k": 5}`
      in the Triton autotuner — reduces benchmark trials from ~50 to ~5
- [ ] Validate: measure autotune wall-clock time before/after on a standard
      GEMM benchmark suite

### Phase 3 (generalise beyond GEMM)
- [ ] Implement `estimateFromIR(FuncOp, config, hw)` — op-count-based cost
      model that works for any Triton kernel (attention, elementwise, etc.)
- [ ] Keep the GEMM-specific `estimatePerf` as a faster, more accurate
      specialisation

### Phase 4 (separate library, long-term)
- [ ] Extract into `libtriton-perf` with a stable C API
- [ ] Publish as a standalone ROCm component, usable by MIOpen, rocBLAS,
      and third-party MLIR-based frameworks
- [ ] Replace Origami's role in the AMD software stack for MLIR-native kernels

---

## Open Questions

1. **Accuracy target**: what prediction error is acceptable for autotuner
   pruning? ±20% TFLOPS? If the model mispredicts and discards the best
   config, the user sees a regression with no obvious cause.

2. **Calibration**: the hardware constants in `HardwareInfo` are derived from
   ISA docs and AMD published specs. How do we validate and update them as
   new GPU SKUs ship?

3. **Relationship to Origami**: should `libtriton-perf` (Phase 4) absorb
   Origami, or co-exist with it? Origami is C++-native and GEMM-specific;
   a MLIR-native version would be more general but less mature.

4. **Warp specialisation and heterogeneous kernels**: the roofline model
   assumes all warps do the same work. Warp-specialised kernels (e.g.
   producer/consumer ping-pong) require a different cost model. When do we
   need to handle this?

5. **Non-AMD targets**: the hardware database today is AMD-only. Is there
   value in extending to NVIDIA (Hopper, Blackwell)? The MFMA/Tensor Core
   instruction tables have different structures but the roofline layer is
   generic.
