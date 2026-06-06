# Work Log — Triton/Gluon PerfModel + Gluon GEMM Kernel Family

Single-file record of work delivered on the `gluon_perf_model` branch.
Grouped by theme, in roughly chronological order within each group.
Task IDs come from the Claude Code task list (numbering may have minor
gaps from deletes/renames; the ordering is what's authoritative).

**Status legend**: ✅ done · 🔄 in-progress · ⏳ pending

---

## Phase 1 — PerfModel foundation (tasks 1–7)

Build the C++ PerfModel API + Python bindings + a working selector PoC.

- ✅ **#1** Add `generateCandidates()` to PerfModel
  Implements `generateCandidates(GemmProblem, HardwareInfo) → vector<TritonGemmConfig>`. Enumerates tile sizes as multiples of the selected MFMA instr dims (Origami `_generate_mt_pairs` style). Filtered through `isValidConfig`.
- ✅ **#2** Add Python bindings for PerfModel
  pybind11 bindings for `GemmProblem`, `TritonGemmConfig`, `HardwareInfo` (including `HardwareInfo::get(archStr)`), `estimatePerf`, `rankConfigs`, `generateCandidates`.
- ✅ **#3** Write PoC Python selector script
  Initial `amd_gemm_select.py`. Given (M, N, K, dtype, arch) → top-K `TritonGemmConfig` as `triton.Config` objects. Later evolved into `amd_gemm_selector.py::pick_gemm_config`.
- ✅ **#4** Wire selector into GEMM tutorial and validate
  Modified `python/tutorials/03-matrix-multiplication.py` to use model-selected configs in place of hand-written `get_hip_autotune_config()` on AMD. Verified correctness + compile-time reduction on gfx942.
- ✅ **#5** Priority 3: Integration regression tests
  Lit tests for `BlockPingpong` and `LowerLoops` PerfModel integration: pingpong gating + LDS-overflow guard fires correctly.
- ✅ **#6** Cherry-pick `d78665bc2` and `e3042870c` into perf_model branch
  asyncmark fix for `buffer_load_to_lds` + a related fix from triton_gluon/matmul_4waves.
- ✅ **#7** Rebuild gluon_perf_model on clean upstream asyncmark
  Reset to perf_model base, bump LLVM hash to `87717bf9f81f`, cherry-pick upstream asyncmark `df82d9878`, re-add `extract_slice` Python binding.

## Phase 2 — Gluon kernel + PerfModel integration (tasks 8–15)

Add Gluon awareness to the model and build the kernel zoo it will choose between.

- ✅ **#8** Add gluon kernel support to PerfModel
  New `kernel_type` parameter on `GemmProblem`/`generateCandidates` (enum: standard, gluon). Gluon constraints: `num_warps=4`, BM/BN multiples of 128, etc.
- ✅ **#9** Pytest: `v9_any_tile` vs `torch.matmul` (bit-exact)
- ✅ **#10** Build v9_small_tile gluon kernel for 16×16 to 64×64
  `num_warps=1`, `warps_per_cta=[1,1]`, single MFMA accumulator (no 4-quadrant), async LDS loads + simple K loop.
- ✅ **#11** Tutorial + baseline benchmarks for gluon matmul
- ✅ **#12** Extend gluon PerfModel to cover small-tile range — allow BM, BN ∈ {16, 32, 64, 128, 256}.
- ✅ **#13** Build v9_small_tile_v2 — 4-warp + async pipeline
  Multi-warp double-buffered async pipeline to close the ~10× gap vs triton-std at small problem sizes. Targets BM, BN ∈ {32, 64, 128}.
- ✅ **#14** Document v9_small_tile_v2 design rationale (why drop 4-quadrant)
- ✅ **#15** Implement v9_small_tile_v3 — 8 warps, aspect-aware `warps_per_cta`
  Targets LLM tall/skinny shapes where 4 warps were the bottleneck.

## Phase 3 — LLIR scheduler integration (tasks 16–19)

Bring the LLIR scheduler from matmul_4waves into the perfmodel branch.

- ✅ **#16** Inventory LLIR scheduler files in matmul_4waves
- ✅ **#17** Cherry-pick LLIR scheduler onto gluon_perf_model_v2 (minimum commit set + conflict resolution)
- ✅ **#18** Build triton_gluon_perfmodel with LLIR scheduler (in `trusting_mccarthy` docker)
- ✅ **#19** Verify LLIR scheduler works on v2 @ 512³ — re-dumped AMDGCN with `TRITON_ENABLE_LLIR_SCHED=1`, confirmed MFMA interleaving, re-measured TFLOPS with rocprofv3.

## Phase 4 — PerfModel regime audit + dispatcher (tasks 20–23)

Add regimes to PerfModel so it knows when to pick which kernel.

- ✅ **#20** Document PerfModel gluon audit + v_skinny addition
  Captured regime table (A/B/C/D) and proposed v_skinny kernel for decode shapes (BM≤8).
- ✅ **#21** PerfModel: add v2 regime + per-regime BK constraints
  `gluonNumWarps` returns 4 for medium tiles, emits 32×32 twice (1w v1, 4w v2), split per-tile K-constraint into three kernel-specific branches.
- ✅ **#22** Dispatcher: route 4-warp medium tiles to v9_small_tile_v2
  Extended `_pick_gluon_tile` in `tutorial_gluon_matmul.py`.
- ✅ **#23** Build + verify 9-shape PerfModel matches measured best (success = 8/9 matches)

## Phase 5 — LLIR scheduler on v9_any_tile (tasks 24–27)

Get the LLIR scheduler working on the larger v9_any_tile body.

- ✅ **#24** Test LLIR scheduler on v9_any_tile (4 large shapes)
- ✅ **#25** Diagnose LLIR scheduler invalid-IR bug for v9_any_tile — captured pre/post-scheduler LLVM IR, identified the broken transformation.
- ✅ **#26** Re-sweep with LLIR + update audit
  Swept v1, v2, v9_any_tile base+LLIR across 9 shapes; determined new global-best per shape.
- ✅ **#27** Auto-enable LLIR for v9_any_tile dispatch path
  Wrap `v9_any_tile.matmul` to set `TRITON_ENABLE_LLIR_SCHED` around the kernel call (delivers +10–31% without env-var ritual).

## Phase 6 — Memory-bound shape correctness (tasks 28–30)

PerfModel was mispicking memory-bound tiles. Diagnose + fix.

- ✅ **#28** Benchmark gluon on TensorAtlas memory-bound shapes (`llama3_mlp_shapes_varied_m.yaml`)
- ✅ **#29** Diagnose PerfModel mispicks for memory-bound shapes
  Compared model predictions vs measured TFLOPS for picked vs best gluon tile; identified which factor (memCycles, numWaves, occupancy penalty) was off.
- ✅ **#30** Implement PerfModel Bug 1 + Bug 2 fixes
  Occupancy-penalty saturation for memory-bound + `dramBwPerCU` divisor when CTAs < CUs.

## Phase 7 — Persistent kernel deep-dive (tasks 31–46)

Investigate the multi-tile-per-CTA persistent kernel + fix LLIR scheduler bugs it exposed.

- ✅ **#31** Baseline rocprofv3 — confirm multi-tile-per-CTA gap
- ✅ **#32** Diagnose: structural under-subscription vs loop overhead
- ✅ **#33** Try fix candidates for multi-tile overhead
- 🔄 **#34** Document outcome + bench evaluation *(stale, mostly superseded)*
- ✅ **#35** PoC: 3-stage async pipeline variant of v_persistent_v1
- ⏳ **#36** PoC: 3-stage with BK=32 (LDS-budget rebalance)
- ✅ **#37** Analyze 3-stage AMDGCN for obstacles
- ✅ **#38** Investigate LLIR scheduler crash on persistent kernel
- ✅ **#39** PoC: batched-commit 3-stage variant
- ✅ **#40** Diff amdgcn: persistent NCTAS=num_tiles vs non-persistent v2
- ✅ **#41** Add XCD remap to v_persistent_v1
- ✅ **#42** HW counter comparison: v2 vs persistent
- ✅ **#43** Fix LLIR scheduler dominance bug on persistent kernel
- ✅ **#44** Diff amdgcn: v2 with LLIR vs without LLIR
- ✅ **#45** Implement Fix A: auto-tune mfmaPerGR
- ✅ **#46** Guard LLIR for single-chain MFMA patterns

## Phase 8 — Any-tile-size kernel + layout system port (tasks 47–55)

Build the v_persistent_any_tile kernel and port the whole zoo to inline-annotated layouts.

- ✅ **#47** Build v_persistent_any_tile (256×256 support)
- ✅ **#48** Build v9_persistent unified dispatcher
- ✅ **#49** Extend v_persistent_any_tile to BM=BN=64
- ✅ **#50** Port v9_small_tile_v2 to inline-annotated layouts
- ✅ **#51** Port v9_small_tile (v1) to inline-annotated layouts
- ✅ **#52** Port v9_any_tile to inline-annotated layouts
- ✅ **#53** Port v9_persistent_v1 to inline-annotated layouts
- ✅ **#54** Port v9_persistent_any_tile to inline-annotated layouts
- ✅ **#55** Phase 2 step 5: merge v9_any_tile + v9_persistent_any_tile

## Phase 9 — PerfModel calibration round 2 (tasks 56–63)

Sustained calibration loop: audit, fix, re-measure.

- ✅ **#56** Apply + rebuild + test PerfModel SIMD-utilization fix
- ✅ **#57** Correctness audit across baseline shapes + fix gluon kernel bugs at 2304
- ✅ **#58** Fix v_any_tile tail-tile bug (M not divisible by BM)
- 🔄 **#59** Diagnose PerfModel ranking issue at large shapes *(in-progress, partially addressed by later XCD/L2 work)*
- 🔄 **#60** Phase 2: implement generic per-tile cost model *(effectively addressed by GluonUnified work)*
- ✅ **#61** Compare LLIR/asm: 128×128×64 vs 256×256×64 at 3072³
- ✅ **#62** rocprof: measure actual stall cost (128 vs 256 cfg @3072³)
- ✅ **#63** Run baseline rocprofv3 sweep with new PerfModel

## Phase 10 — GluonUnified kernel family (tasks 64–70)

The headline product: one kernel covers v1 + any_tile + small + skinny via constexpr branches.

- ✅ **#64** Design unified gluon matmul kernel (v1 + any_tile merge)
- ✅ **#65** Bind `composePaddedLayoutForAsyncCopyCDNA4` to Python
- ✅ **#66** Port v_any_tile 4-quadrant body with parametric layouts
- ✅ **#67** Mirror gfx1250 parametric layout system for gfx950
- ✅ **#68** Constexpr-branch single-chain + 4-quadrant in one kernel
- ✅ **#69** Install triton_gluon (matmul_4waves) and reproduce v9_beyond_hotloop perf
- ✅ **#70** Add K-major B support to GluonUnified

## Phase 11 — fp8 (a8w8) port — current focus (tasks 71–82)

Bring the a8w8 fp8 kernel into the GluonUnified family with any-tile support and per-token scales.

- ✅ **#71** Port fp8 (a8w8) into GluonUnified kernel
- ⏳ **#72** Port fp4 (a4w4) into GluonUnified kernel
- ✅ **#73** Rebase gluon_perf_model onto gfx950-tutorial-v0.1
- ⏳ **#74** Stage 1: port gfx1250 cross-tile prefetch pattern to v_persistent_v1
- ⏳ **#75** Stage 2: 4-quadrant epilogue-overlap in v_persistent_any_tile_padded
- ✅ **#76** Rebase to upstream/main + cherry-pick LLIR scheduler
- 🔄 **#77** Unified fp8 kernel: SCALE_MODE constexpr (NONE / PER_TOKEN / PER_K_BLOCK / MXFP) *(NONE + PER_TOKEN/single-chain done)*
- ✅ **#78** Draft unified fp8 kernel design doc (SCALE_MODE branches) — `docs/unified-fp8-kernel-design.md`
- ✅ **#79** Investigate 22% perf regression from `mfma_scaled` → unscaled at BK=128
  Root cause: scaled MFMA does 4× the work per instruction (16,384 cycles either way), unscaled needs 4× more instructions which means 50% more `s_waitcnt` + more issue-pressure. Fix: per-BK dispatch keeps scaled path for BK≥128.
- ✅ **#80** Add single-chain body + USE_SINGLE_CHAIN gate to a8w8 v9_persistent (mirror a16w16 pattern)
  Coverage: BM/BN ∈ {16…256}, BK ∈ {32…256}. 19/20 tile configs validated at 2048³.
- 🔄 **#81** Step 2: PER_TOKEN scales — single-chain done, 4-quadrant pending
  Working: 5 single-chain tile configs at max_diff 0.25 vs fp32 ref. Pending: propagate to 4-quadrant scaled + unscaled bodies (8 sub-tile multiplies per tile).
- ✅ **#82** Extend single-chain to BM/BN < 64 (natural-layout fallback debug)
  Fixed: `_make_padded_layout_natural` ported from a16w16 with corrected fp8 padding `[1024, 8]` and basis-order split at `log2(kWidth)`. Bit-identical to upstream API where the latter is supported; extends to tiles upstream asserts on.

---

## Phase 12 — PerfModel baseline + regression diagnosis (tasks 83–87)

Build the ground-truth measurement infrastructure, then use it to find and fix
PerfModel mispicks at large prefill shapes.

- ✅ **#83** T1.1: PerfModel baseline sweep (model vs autotune vs rocBLAS, 24 shapes)
  Harness in `scripts/perf_baseline/{sweep,preflight,shapes}.py`. Output: `docs/perf-baselines/perf-baseline-2026-06-05-prefix.{csv,md}`. PM beats autotune on 17/24 (70%), geomean 1.31×. Identified 4 large-prefill regressions.
- ⏳ **#84** T1.2: Validate PER_TOKEN single-chain vs aiter (bit-equivalence) *(pending)*
- ✅ **#85** T1.3: TensorAtlas pilot tuning on persistent_matmul (4 regression shapes)
  Pruned-mode tuning revealed all 4 winners cluster at BM=256, BN=256, BK=64, nW=8, nS=2. Initial wrong diagnosis (later corrected): "num_warps=8 missing from PM picks" — turned out PM does emit nW=8. Real issue was the cost-model formula.
- ✅ **#86** PerfModel Fix 2: memory-overlap formula + GPU under-fill correction
  Replaced `effectiveTileCycles = max(compute, mem-hidden)` with `compute + max(0, mem-hidden)` (unhidden memory serializes on top of compute, doesn't get max'd out). Added under-fill floor via `max(1.0, totalTiles/numCUs)`. First attempt: universal `compute + unhidden_mem` alone overshot (4 wins → 16% win rate); revealed missing GPU under-fill case. Combined fix: 17→16 wins, geomean 1.31× → 1.30×, **3 of 4 regressions resolved**.
- ✅ **#87** PerfModel Fix 3: remove selectGroupSizeM large-grids shortcut + use est.numWaves (ceildiv)
  Replaced `max(1.0, ratio)` floor with `est.numWaves` directly (ceildiv). Continuous `max(1.0, ratio)` was correct for under-fill but lost the partial-wave wall-clock cost in the 1<ratio<2 regime; ceildiv handles both correctly. Also removed `selectGroupSizeM`'s "large grids insensitive" shortcut that bypassed cost-eval. **All 4 original regressions resolved (8192×2880×4096: 0.57× → 1.02×), only 1 edge loss remaining (8192×5120×2880 at 0.95×), geomean 1.34×**. Output: `docs/perf-baselines/perf-baseline-2026-06-06-fix3.{csv,md}`.

## Status summary

| state | count | notes |
|---|---|---|
| ✅ completed | 73 | |
| 🔄 in progress | 4 | #34 stale; #59, #60 partially addressed; #77, #81 active |
| ⏳ pending | 5 | #36, #72, #74, #75, #84 |

## Areas not yet ticketed

These have come up in discussion but don't have task IDs yet:

- Fix the (128, 128, 32) 4-quadrant case (pre-existing layout-helper limit)
- Side-by-side PER_TOKEN validation against `aiter.gemm_a8w8`
- Skinny / GEMV kernel for BM ∈ {1, 4, 8} (the current kernel floor is BM=16)
- Symbolic-output mode for PerfModel (borrow from ThroughputSolver, see `docs/perf-model-direction.md` § Related Work)
- Kernel/arch DSL for non-GEMM patterns (attention, conv) — also from `perf-model-direction.md`

## Process

- This log is updated when work lands. New tasks go in the appropriate phase section as `⏳`, flip to `🔄` when started, `✅` when done.
- Source of truth for IDs ≥ 31 is the Claude Code task list. IDs 1–30 were recovered from `~/.claude/projects/-home-xiaohugu-openai-triton/793567a3-d47f-46cd-96d4-12065b49a7f6.jsonl` (numbering may have minor gaps from earlier renames).
- When a task lands, append the commit SHA inline: `✅ **#N** subject — landed in \`<sha>\``
- For deeper detail on any task, grep the session transcript jsonl above.
