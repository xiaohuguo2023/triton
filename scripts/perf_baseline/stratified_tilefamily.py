"""STEP 1: ground-truth the tile sweet spot across ALL categories.

For a stratified sample (N per category, spanning the size range), MEASURE a tile
family and record the measured-best tile + the features that should gate a
large-tile efficiency credit (tile count / wave fill, wave_efficiency,
compute-bound, N%BN padding), plus what the model currently picks. Output feeds
the gated-credit design (Step 2).

Gated (preflight before start + util re-check per shape), incremental-save, and
RESUMABLE (skips shapes already in the CSV), so an intermittent neighbor can't
waste the run.

Usage:  python3 stratified_tilefamily.py [--per-cat 4] [--gpu 0] [--force]
Output: docs/perf-baselines/tilefamily_ground_truth.csv
"""
import argparse, csv, math, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from shapes_all import ALL_SHAPES  # noqa: E402

# tile family spanning the small<->large decision (S3, W8; BK 32/64)
FAM = [(64, 64, 64), (64, 128, 64), (64, 256, 64), (128, 64, 64),
       (128, 128, 64), (128, 128, 32), (128, 256, 64), (128, 256, 32),
       (256, 128, 64), (256, 256, 32), (256, 256, 64), (256, 64, 64)]
OUT = HERE.parent.parent / "docs" / "perf-baselines" / "tilefamily_ground_truth.csv"
COLS = ["cat", "M", "N", "K", "best_bm", "best_bn", "best_bk", "best_tf",
        "model_bm", "model_bn", "model_bk", "model_tf", "pm_over_best",
        "best_num_waves", "best_wave_eff", "best_tiles", "best_is_cb",
        "best_pad_n_frac", "num_cus"]


def _busy_once():
    try:
        out = subprocess.run(["rocm-smi", "--showuse"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    return [f"GPU{m.group(1)}={m.group(2)}%" for m in
            re.finditer(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", out)
            if int(m.group(2)) > 10]


def gpu_busy():
    b1 = _busy_once()
    if not b1:
        return b1
    time.sleep(3)
    return _busy_once()


def stratified(per_cat):
    groups = {}
    for (M, N, K, r) in ALL_SHAPES:
        groups.setdefault(r.split("-")[0], []).append((M, N, K, r))
    picks = []
    for cat, shs in groups.items():
        shs = sorted(shs, key=lambda s: s[0] * s[1] * s[2])
        n = min(per_cat, len(shs))
        idxs = sorted(set(int(i * (len(shs) - 1) / max(n - 1, 1)) for i in range(n)))
        picks += [shs[i] for i in idxs]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")

    if gpu_busy() and not args.force:
        print(f"ABORT: machine not clean: {_busy_once()}", file=sys.stderr); sys.exit(1)

    import importlib.util, io, contextlib
    import torch, triton
    from triton.backends.amd import amd_gemm_selector as sel
    from triton._C.libtriton import amd
    pm = amd.perf_model
    TUT = "/home/openai/triton_gluon_perfmodel/python/tutorials/03-matrix-multiplication.py"
    spec = importlib.util.spec_from_file_location("tut03", TUT)
    tut = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try: spec.loader.exec_module(tut)
        except BaseException: pass
    arch = sel.current_amd_arch()
    hw = pm.HardwareInfo.get(arch)

    def bench_ms(fn, w=8, r=25):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
        for _ in range(r):
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    def mk(bm, bn, bk):
        c = pm.TritonGemmConfig(); c.block_m, c.block_n, c.block_k = bm, bn, bk
        c.num_warps = 8; c.num_stages = 3; c.group_size_m = 1; c.mfma_non_k_dim = 16; c.waves_per_eu = 0
        return c

    def bench(a, b, M, N, K, bm, bn, bk):
        try:
            c = mk(bm, bn, bk); kw = sel.config_to_kernel_kwargs(c)
            out = torch.empty((M, N), device="cuda", dtype=torch.float16)
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            return bench_ms(lambda: tut.matmul_kernel_amd[grid](a, b, out, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1), ACTIVATION="", **kw))
        except Exception:
            return None

    def tf(M, N, K, ms): return 2 * M * N * K / (ms * 1e-3) / 1e12

    # resume
    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            done.add((int(r["M"]), int(r["N"]), int(r["K"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    fh = open(OUT, "a", newline=""); wr = csv.DictWriter(fh, fieldnames=COLS)
    if new_file: wr.writeheader()

    picks = stratified(args.per_cat)
    todo = [p for p in picks if (p[0], p[1], p[2]) not in done]
    print(f"{len(picks)} sampled, {len(todo)} to run ({len(done)} already done)")
    for i, (M, N, K, r) in enumerate(todo, 1):
        if gpu_busy() and not args.force:
            print(f"neighbor before {M}x{N}x{K}; stopping ({i-1}/{len(todo)} this run). Re-run to resume.")
            break
        cat = r.split("-")[0]
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        best = None
        for (bm, bn, bk) in FAM:
            ms = bench(a, b, M, N, K, bm, bn, bk)
            if ms and (best is None or ms < best[0]):
                best = (ms, bm, bn, bk)
        # model pick (current binary) measured
        mc = sel.pick_gemm_config(M, N, K, "fp16", arch, top_k=1)[0]
        mms = bench(a, b, M, N, K, mc.block_m, mc.block_n, mc.block_k)
        # features for the measured-best tile
        bm, bn, bk = best[1], best[2], best[3]
        prob = pm.GemmProblem(M, N, K, pm.ElemKind.FP16, pm.ElemKind.FP16, pm.ElemKind.FP32, 16, 16, 32)
        est = pm.estimate_perf(prob, mk(bm, bn, bk), hw)
        best_tf = tf(M, N, K, best[0]); model_tf = tf(M, N, K, mms) if mms else 0
        pad = (((N + bn - 1) // bn) * bn - N) / N
        row = {"cat": cat, "M": M, "N": N, "K": K,
               "best_bm": bm, "best_bn": bn, "best_bk": bk, "best_tf": f"{best_tf:.0f}",
               "model_bm": mc.block_m, "model_bn": mc.block_n, "model_bk": mc.block_k,
               "model_tf": f"{model_tf:.0f}",
               "pm_over_best": f"{model_tf/best_tf:.3f}" if best_tf else "",
               "best_num_waves": est.num_waves, "best_wave_eff": f"{est.wave_efficiency:.3f}",
               "best_tiles": est.total_output_tiles, "best_is_cb": est.is_compute_bound,
               "best_pad_n_frac": f"{pad:.3f}", "num_cus": hw.num_cus}
        wr.writerow(row); fh.flush()
        print(f"  [{i:2}/{len(todo)}] {cat:<12} {M}x{N}x{K:<6} best=BM{bm}BN{bn}BK{bk} "
              f"({best_tf:.0f}TF)  model=BM{mc.block_m}BN{mc.block_n} "
              f"(pm/best={model_tf/best_tf:.2f})  waves={est.num_waves} pad={pad:.2f}", flush=True)
        del a, b; torch.cuda.empty_cache()
    fh.close()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
