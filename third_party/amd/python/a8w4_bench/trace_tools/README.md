# trace_tools — CPU+GPU critical-path trace-diff toolkit

Debugging kit for "kernel X looks slower in serving but per-kernel numbers don't
explain it". Separates **GPU kernel busy time** from **CPU-side dispatch / idle /
sync**, so you can classify a slowdown as:

- **A** GPU kernel itself slower  → per-category busy up
- **B** GPU idle/launch gaps      → `gpu_idle` up
- **C** CPU selector/dispatch     → `cpu_launch`/`cpu_pick` up, launches/step up
- **D** lost comm/compute overlap → (only if >1 GPU stream)
- **E** more work / more steps     → kernels/step up, per-call time same

Built for gpt-oss-120b W4A8 on MI355X (single GPU compute stream, so
`gpu_idle = wall − Σ kernel_dur` is exactly the device bubble waiting on the host),
but the scripts are generic torch-profiler trace parsers.

## Prereqs for capturing usable traces

Serve with the torch profiler on and an env-gated per-launch label:

```
--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$DIR \
--profiler-config.torch_profiler_use_gzip=true \
--profiler-config.active_iterations=200 --profiler-config.warmup_iterations=2
# and in env:  AITER_A8W4_PROF_LABEL=1   (wraps each a8w4 launch, moe_op_gemm_a8w4.py:588-654)
```
Call `POST /start_profile` right before the bench and `POST /stop_profile` after.
Traces land as `$DIR/dp0_pp0_tp<rank>_*.pt.trace.json.gz` (one per rank; the tools
read tp0 unless told otherwise). See ../run_ab.sh for a full cold-cache serve+bench.

## Workflow

1. **inspect_trace.py `<dir>`** — always run first on a new trace layout. Dumps
   event categories, how many GPU streams (pid,tid), step markers, kernel-name
   families (to tune `classify()`), launch/sync names, CPU annotation names.

2. **decompose.py `<dir>` [gen_tag]** — the main tool. Per decode step (bucketed
   into each step's own `execute_context_..._generation_16(16)` CPU-annotation
   interval, so interleaved prefill is excluded):
   `step_wall, gpu_busy, gpu_idle, {moe_gemm,comm,attention,dense_gemm,moe_aux}
   busy, kernels/step, launches/step, cpu_launch, cpu_sync, cpu_pick,
   moe_us_per_call, comm_us_per_call`. Change `gen_tag` for other concurrency
   (e.g. `generation_8(8)`).

3. **moe_hist.py `<dir>`** — histogram of `_moe_gemm_a8w4` kernel durations inside
   decode steps. Use when labels don't cover decode (see below): a config change
   shows up as a shift between duration buckets (`<6us` narrow-BN vs `20-30us`
   BN256).

4. **step_compose.py `<dir>`** — checks whether "decode" steps actually have
   prefill tokens chunked in (rules out component E contamination).

5. **parse_ab.py `<dir>` [run1 run2 ...]** — reads the per-config `a8w4cfg` labels
   for the forced-config A/B (below) and prints per-shape A-vs-B avg_us / ratio.

## Forced-config A/B (matched routing in ONE serve)

`AITER_A8W4_AB` in ../../../openai/perfmodel_a8w4_select.py (`_ab_spec`/`_ab_override`)
pins a target shape to TWO configs and alternates them call-by-call in a single
serve, so both are measured under **identical routing / window / thermal**:

```
AITER_A8W4_AB="bm:n:k=bn/bk/ns/nw|bn/bk/ns/nw;..."   # left=A (e.g. JSON), right=B (e.g. PM)
# e.g. 64:1024:3072=128/256/1/4|256/256/2/8
```
Verify it fired via the picklog: `awk '$4==BM && $2==N' $DIR/picklog.txt | sort | uniq -c`
should show ~50/50 between the two configs.

## Gotchas learned the hard way

- **Labels only attach to EAGER launches.** Prefill bm64/bm128 get `gpu_user_annotation`
  labels; tiny / graph-replayed DECODE kernels (bm16/bm32) do NOT. So `parse_ab.py`
  sees only prefill shapes — use `moe_hist.py` (duration) for decode configs.
- **Bucket kernels into each step's own interval**, not the whole `[first,last]` span
  — at conc>1, prefill steps interleave and pollute the span (early bug: pm gpu_wall
  13147us vs step_wall 1462us until fixed).
- **Separate arms have different routing** (no bench `--seed`): never compare
  `moe_us_per_call` across two independent serves — the block_m mix differs. Only the
  single-serve forced A/B (or duration histogram within one serve) is matched-routing.
- **Single stream ⇒ no overlap to lose (D).** If your target has a separate comm
  stream, add per-stream union/intersection to measure overlap.
