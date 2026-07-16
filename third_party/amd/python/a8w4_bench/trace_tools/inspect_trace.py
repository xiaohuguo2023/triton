#!/usr/bin/env python3
"""Inspect a torch-profiler trace's schema before writing analysis against it.
Reports: event categories, GPU stream (pid,tid), per-step markers, kernel-name
families, cuda_runtime (launch/sync) names, CPU user_annotation names.

Usage: inspect_trace.py <trace_dir> [rank_glob]
  rank_glob default 'dp0_pp0_tp0_*'  (one rank; adjust for your filenames)
"""
import gzip, json, glob, collections, sys
d = sys.argv[1]
rg = sys.argv[2] if len(sys.argv) > 2 else "dp0_pp0_tp0_*"
fp = sorted(glob.glob(f"{d}/{rg}.pt.trace.json.gz"))[0]
ev = json.load(gzip.open(fp))["traceEvents"]
print("file:", fp.split("/")[-1][:40], "events:", len(ev))
print("categories:", dict(collections.Counter(e.get("cat") for e in ev)))
tnames = {(e["pid"], e["tid"]): e.get("args", {}).get("name", "")
          for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
kpt = collections.Counter((e["pid"], e["tid"]) for e in ev if e.get("cat") == "kernel")
print("\nkernel (pid,tid) -> count [thread_name]  (how many GPU streams?):")
for (p, t), c in kpt.most_common(8):
    print("  pid%s tid%s: %d  name=%s" % (p, t, c, tnames.get((p, t), "?")))
ps = [e for e in ev if "ProfilerStep" in str(e.get("name", ""))]
print("\nProfilerStep events:", len(ps))
kn = collections.Counter(e.get("name", "")[:46] for e in ev if e.get("cat") == "kernel")
print("\ntop kernel names (for classify() families):")
for n, c in kn.most_common(22): print("  %6d  %s" % (c, n))
cr = collections.Counter(e.get("name", "") for e in ev if e.get("cat") == "cuda_runtime")
print("\ncuda_runtime (launch/sync) names:")
for n, c in cr.most_common(12): print("  %6d  %s" % (c, n))
ua = collections.Counter(e.get("name", "")[:46] for e in ev if e.get("cat") == "user_annotation")
print("\nCPU user_annotation names (step markers / a8w4cfg labels / nccl):")
for n, c in ua.most_common(12): print("  %6d  %s" % (c, n))
