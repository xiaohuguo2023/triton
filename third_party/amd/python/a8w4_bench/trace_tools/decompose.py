#!/usr/bin/env python3
"""CPU+GPU critical-path decode-step decomposition from a torch-profiler trace.

Separates GPU kernel busy time from CPU-side scheduling/dispatch/idle so we can
tell WHERE a PM-vs-JSON delta comes from:
  A. GPU kernel itself slower      -> per-category busy up
  B. GPU idle/launch gaps          -> gpu_idle_gap up
  C. CPU selector/dispatch         -> cpu launch/dispatch up, launches/step up
  D. lost comm/compute overlap     -> (single stream here => N/A, reported anyway)
  E. more work / more steps        -> kernels/step up, nsteps up

Single GPU stream (verified: all kernels on one stream/tid) => kernels are
serialized, so gpu_busy = sum(kernel dur) and gpu_idle = wall - busy is exactly
the device bubble waiting on the host. Decode steps = CPU user_annotation
'execute_context_..._generation_16(16)'. Per-step = aggregate / nsteps.

Usage: decompose.py <trace_dir> [gen_tag]   # gen_tag default 'generation_16(16)'
"""
import gzip, json, glob, sys, collections

TRACE_DIR = sys.argv[1]
GEN_TAG = sys.argv[2] if len(sys.argv) > 2 else "generation_16(16)"

LAUNCH = {"hipLaunchKernel", "hipModuleLaunchKernel", "hipExtModuleLaunchKernel",
          "hipGraphLaunch"}
SYNC = {"hipEventSynchronize", "hipStreamSynchronize", "hipDeviceSynchronize",
        "hipStreamWaitEvent"}

def classify(nm):
    if nm.startswith("_moe_gemm_a8w4"): return "moe_gemm"
    if ("cross_device_reduce" in nm or "quickreduce" in nm or "allreduce" in nm
            or "all_reduce" in nm or "nccl" in nm or "reduce_scatter" in nm
            or "all_gather" in nm): return "comm"
    if "unified_attention" in nm or "attention" in nm.lower(): return "attention"
    if ("wvSplitK" in nm or "gemm_a16_w16" in nm or nm.startswith("Cijk")): return "dense_gemm"
    if ("routing" in nm or "_topk" in nm or "rmsnorm" in nm or "rope" in nm
            or "reduce_grouped" in nm or "downcast" in nm or "reduce_segments" in nm
            or "reshape_and_cache" in nm or "slot_mapping" in nm): return "moe_aux"
    return "other"

def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

import bisect

def decompose(fp):
    ev = json.load(gzip.open(fp))["traceEvents"]
    # decode-step intervals (CPU annotations); non-overlapping, sorted by start.
    steps = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in ev
             if e.get("cat") == "user_annotation"
             and str(e.get("name", "")).startswith("execute_context")
             and GEN_TAG in str(e.get("name", ""))]
    steps.sort()
    if not steps:
        return None
    nsteps = len(steps)
    step_wall = sum(b - a for a, b in steps) / nsteps  # mean CPU step wall
    starts = [a for a, _ in steps]

    def in_step(ts):
        # index of the decode step whose interval contains ts, else -1
        i = bisect.bisect_right(starts, ts) - 1
        return i if 0 <= i < nsteps and ts <= steps[i][1] else -1

    # GPU kernels bucketed into their own decode step (excludes interleaved prefill)
    busy = collections.Counter()
    cnt = collections.Counter()
    nkern = 0
    per_step_span = collections.defaultdict(lambda: [float("inf"), float("-inf")])
    for e in ev:
        if e.get("cat") != "kernel":
            continue
        i = in_step(e["ts"])
        if i < 0:
            continue
        cl = classify(e.get("name", ""))
        busy[cl] += e.get("dur", 0.0)
        cnt[cl] += 1
        nkern += 1
        s = per_step_span[i]; en = e["ts"] + e.get("dur", 0)
        if e["ts"] < s[0]: s[0] = e["ts"]
        if en > s[1]: s[1] = en
    gpu_busy = sum(busy.values())
    gpu_wall = sum(b - a for a, b in per_step_span.values() if b > a)
    gpu_idle = gpu_wall - gpu_busy
    # CPU launch/dispatch + sync within decode steps
    launch = sync = 0.0
    nlaunch = 0
    for e in ev:
        if e.get("cat") != "cuda_runtime":
            continue
        if in_step(e["ts"]) < 0:
            continue
        nm = e.get("name", "")
        if nm in LAUNCH:
            launch += e.get("dur", 0.0); nlaunch += 1
        elif nm in SYNC:
            sync += e.get("dur", 0.0)
    # a8w4cfg CPU annotation time (selector+launch on host path, if captured)
    pick = sum(e.get("dur", 0.0) for e in ev
               if e.get("cat") == "user_annotation"
               and str(e.get("name", "")).startswith("a8w4cfg")
               and in_step(e["ts"]) >= 0)
    per = lambda x: x / nsteps
    return {
        "nsteps": nsteps, "step_wall": step_wall,
        "gpu_wall": per(gpu_wall), "gpu_busy": per(gpu_busy),
        "gpu_idle": per(gpu_idle),
        "moe_gemm": per(busy["moe_gemm"]), "comm": per(busy["comm"]),
        "attention": per(busy["attention"]), "dense_gemm": per(busy["dense_gemm"]),
        "moe_aux": per(busy["moe_aux"]), "other": per(busy["other"]),
        "kernels": nkern / nsteps, "launches": nlaunch / nsteps,
        "cpu_launch": per(launch), "cpu_sync": per(sync),
        "cpu_pick": per(pick),
        # per-call normalization (factors out routing volume)
        "moe_calls": cnt["moe_gemm"] / nsteps,
        "moe_us_per_call": busy["moe_gemm"] / max(1, cnt["moe_gemm"]),
        "comm_calls": cnt["comm"] / nsteps,
        "comm_us_per_call": busy["comm"] / max(1, cnt["comm"]),
    }

def load(dirn):
    fps = sorted(glob.glob(f"{dirn}/dp0_pp0_tp0_*.pt.trace.json.gz"))
    return decompose(fps[0]) if fps else None

if __name__ == "__main__":
    r = load(TRACE_DIR)
    if not r:
        print("no trace / no decode steps for", TRACE_DIR, GEN_TAG); sys.exit(1)
    print(f"{TRACE_DIR}  (gen={GEN_TAG})")
    for k, v in r.items():
        print(f"  {k:12} {v:12.2f}")
