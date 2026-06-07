"""Compare PerfModel ranking vs TensorAtlas measured ranking on an exhaustive
tuning cache. Diagnoses *which knobs* PerfModel under/over-prioritizes.

Reuses dashboard.loaders to flatten the cache; adds a `pm_tflops` column
per row and prints the side-by-side rank comparison + param diff between
PM's top pick and the measured top pick (same diff pattern as
dashboard/views/best_vs_rest.py).

Usage:
  python3 pm_vs_measured.py <tuning_cache.json>
"""
from __future__ import annotations
import argparse, sys

sys.path.insert(0, "/home/work/TensorAtlas")
from dashboard.loaders import (
    load_cache, config_columns, hw_counter_columns, ir_feature_columns, param_name,
)  # noqa

from triton._C.libtriton import amd
from triton.backends.amd.amd_gemm_selector import current_amd_arch

pm = amd.perf_model


def build_pm_tflops_column(df, M, N, K):
    arch = current_amd_arch()
    hw = pm.HardwareInfo.get(arch)
    prob = pm.GemmProblem()
    prob.M, prob.N, prob.K = M, N, K
    prob.a_kind = pm.ElemKind.FP16
    prob.b_kind = pm.ElemKind.FP16
    prob.c_kind = pm.ElemKind.FP32
    prob.a_bits = prob.b_bits = 16
    prob.c_bits = 32

    def predict(row):
        c = pm.TritonGemmConfig()
        c.block_m = int(row["cfg_BLOCK_SIZE_M"])
        c.block_n = int(row["cfg_BLOCK_SIZE_N"])
        c.block_k = int(row["cfg_BLOCK_SIZE_K"])
        c.num_warps = int(row["cfg_num_warps"])
        c.num_stages = int(row["cfg_num_stages"])
        c.mfma_non_k_dim = int(row.get("cfg_matrix_instr_nonkdim", 16) or 16)
        c.group_size_m = int(row.get("cfg_GROUP_SIZE_M", 0) or 0)
        c.use_async_copy = True
        e = pm.estimate_perf(prob, c, hw)
        return e.predicted_tflops if e.is_valid else 0.0

    df["pm_tflops"] = df.apply(predict, axis=1)
    return df


def param_diff(row_a, row_b, cfg_cols, label_a="A", label_b="B"):
    """Mirrors best_vs_rest.py:render — list cfg_* columns that differ."""
    rows = []
    for c in cfg_cols:
        va, vb = row_a.get(c), row_b.get(c)
        if va != vb:
            rows.append((param_name(c), va, vb))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", help="path to tuning_cache.json")
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    meta, df = load_cache(args.cache)
    M, N, K = meta["M"], meta["N"], meta["K"]
    print(f"=== {M} x {N} x {K} === ({len(df)} configs)")

    ok_mask = df["status"].isin(["ok", "success"])
    df = df[ok_mask].copy()
    print(f"successful: {len(df)}")

    cfg_cols = config_columns(df)
    print(f"cfg columns: {[param_name(c) for c in cfg_cols]}")

    df = build_pm_tflops_column(df, M, N, K)

    # TA rank by measured TFLOPS desc; PM rank by predicted TFLOPS desc
    ta_sorted = df.sort_values("tflops", ascending=False).reset_index(drop=True)
    ta_sorted["ta_rank"] = ta_sorted.index + 1
    pm_sorted = df.sort_values("pm_tflops", ascending=False).reset_index(drop=True)
    pm_sorted["pm_rank"] = pm_sorted.index + 1

    # Join on config_str to get both ranks per config
    joined = ta_sorted[["config_str", "tflops", "pm_tflops", "ta_rank"]].merge(
        pm_sorted[["config_str", "pm_rank"]], on="config_str")

    # ── TOP-N by measured ranking ─────────────────────────────────────────────
    print()
    print(f"TOP {args.topn} by TA-MEASURED TFLOPS:")
    print(f"  {'TA#':>3} {'PM#':>5} {'TA_TF':>7} {'PM_TF':>7}  config")
    for _, r in joined.head(args.topn).iterrows():
        print(f"  {int(r['ta_rank']):>3} {int(r['pm_rank']):>5} "
              f"{r['tflops']:>7.1f} {r['pm_tflops']:>7.1f}  {r['config_str']}")

    # ── TOP-N by PM ranking ───────────────────────────────────────────────────
    pm_top = pm_sorted.head(args.topn).merge(
        ta_sorted[["config_str", "ta_rank"]], on="config_str")
    print()
    print(f"TOP {args.topn} by PM-PREDICTED TFLOPS:")
    print(f"  {'PM#':>3} {'TA#':>5} {'PM_TF':>7} {'TA_TF':>7}  config")
    for _, r in pm_top.iterrows():
        print(f"  {int(r['pm_rank']):>3} {int(r['ta_rank']):>5} "
              f"{r['pm_tflops']:>7.1f} {r['tflops']:>7.1f}  {r['config_str']}")

    # ── Param diff: PM#1 vs TA#1 ─────────────────────────────────────────────
    ta_best_cfg = ta_sorted.iloc[0]["config_str"]
    pm_best_cfg = pm_sorted.iloc[0]["config_str"]
    ta_best_row = ta_sorted.iloc[0]
    pm_best_row = pm_sorted.iloc[0]
    ta_rank_of_pm_pick = int(joined[joined.config_str == pm_best_cfg].ta_rank.iloc[0])
    pm_rank_of_ta_pick = int(joined[joined.config_str == ta_best_cfg].pm_rank.iloc[0])

    print()
    print("Summary:")
    print(f"  TA #1: {ta_best_cfg}  ({ta_best_row['tflops']:.1f} TF measured)")
    print(f"         PM predicts {ta_best_row['pm_tflops']:.1f} TF → PM rank #{pm_rank_of_ta_pick}")
    print(f"  PM #1: {pm_best_cfg}  ({pm_best_row['pm_tflops']:.1f} TF predicted)")
    print(f"         TA measures {pm_best_row['tflops']:.1f} TF → TA rank #{ta_rank_of_pm_pick}")
    print(f"  PM-pick gap to ground truth: {(1 - pm_best_row['tflops']/ta_best_row['tflops'])*100:.1f}%")

    print()
    diffs = param_diff(ta_best_row, pm_best_row, cfg_cols)
    print(f"Parameter diff (TA#1 vs PM#1): {len(diffs)} knob(s) differ")
    print(f"  {'param':<24} {'TA_pick':>10} {'PM_pick':>10}")
    for name, ta_v, pm_v in diffs:
        print(f"  {name:<24} {str(ta_v):>10} {str(pm_v):>10}")

    # ── Tile-level (BM,BN,BK) rank comparison ────────────────────────────────
    # Knobs interact — a tile that's bad as marginal (e.g. BN=32 alone) might
    # be the best (BM,BN,BK) triple when combined with the right BK. Group
    # by (BM,BN,BK) and rank by best TFLOPS within each tile.
    import pandas as pd
    tile_key = ["cfg_BLOCK_SIZE_M", "cfg_BLOCK_SIZE_N", "cfg_BLOCK_SIZE_K"]
    ta_tiles = (df.sort_values("tflops", ascending=False)
                  .groupby(tile_key, as_index=False)
                  .agg(ta_best_tflops=("tflops", "max"),
                       ta_count=("tflops", "size"),
                       pm_tflops_for_tile=("pm_tflops", "first"))
                  .sort_values("ta_best_tflops", ascending=False)
                  .reset_index(drop=True))
    ta_tiles["ta_tile_rank"] = ta_tiles.index + 1
    pm_tiles = (df.sort_values("pm_tflops", ascending=False)
                  .groupby(tile_key, as_index=False)
                  .agg(pm_best_tflops=("pm_tflops", "max"),
                       ta_tflops_for_tile=("tflops", "first"))
                  .sort_values("pm_best_tflops", ascending=False)
                  .reset_index(drop=True))
    pm_tiles["pm_tile_rank"] = pm_tiles.index + 1

    merged = ta_tiles.merge(
        pm_tiles[tile_key + ["pm_best_tflops", "pm_tile_rank"]],
        on=tile_key)
    print()
    print(f"=== TILE (BM,BN,BK) ranking ===  ({len(merged)} unique tiles)")
    print()
    print(f"TOP {args.topn} TILES by TA best TFLOPS:")
    print(f"  {'TA#':>3} {'PM#':>4} {'BM':>3} {'BN':>4} {'BK':>4}  "
          f"{'TA_TF':>7} {'PM_TF':>7}  #cfgs")
    for _, r in merged.head(args.topn).iterrows():
        print(f"  {int(r['ta_tile_rank']):>3} {int(r['pm_tile_rank']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_M']):>3} {int(r['cfg_BLOCK_SIZE_N']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_K']):>4}  {r['ta_best_tflops']:>7.1f} "
              f"{r['pm_best_tflops']:>7.1f}  {int(r['ta_count'])}")

    print()
    print(f"TOP {args.topn} TILES by PM-predicted TFLOPS:")
    pm_view = merged.sort_values("pm_best_tflops", ascending=False).head(args.topn)
    print(f"  {'PM#':>3} {'TA#':>4} {'BM':>3} {'BN':>4} {'BK':>4}  "
          f"{'PM_TF':>7} {'TA_TF':>7}")
    for _, r in pm_view.iterrows():
        print(f"  {int(r['pm_tile_rank']):>3} {int(r['ta_tile_rank']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_M']):>3} {int(r['cfg_BLOCK_SIZE_N']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_K']):>4}  {r['pm_best_tflops']:>7.1f} "
              f"{r['ta_best_tflops']:>7.1f}")

    # ── Where do PM's top-N picks land in TA's measured ranking? ─────────────
    print()
    print(f"PM's TOP-{args.topn} tiles, sorted by their TA-measured rank:")
    print(f"  {'PM#':>3} {'TA#':>4} {'BM':>3} {'BN':>4} {'BK':>4}  "
          f"{'PM_TF':>7} {'TA_TF':>7} {'%best':>6}")
    ta_best_tf_val = merged.iloc[0]["ta_best_tflops"]
    pm_topN = merged.sort_values("pm_best_tflops", ascending=False).head(args.topn)
    for _, r in pm_topN.sort_values("ta_tile_rank").iterrows():
        pct = r["ta_best_tflops"] / ta_best_tf_val * 100
        print(f"  {int(r['pm_tile_rank']):>3} {int(r['ta_tile_rank']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_M']):>3} {int(r['cfg_BLOCK_SIZE_N']):>4} "
              f"{int(r['cfg_BLOCK_SIZE_K']):>4}  {r['pm_best_tflops']:>7.1f} "
              f"{r['ta_best_tflops']:>7.1f} {pct:>5.1f}%")

    # Best-of-PM-top-N: if we picked PM's top-N and ran them all, what's the best?
    best_of_pm = pm_topN["ta_best_tflops"].max()
    print()
    print(f"  best TA TFLOPS among PM's top-{args.topn}: {best_of_pm:.1f} "
          f"({best_of_pm/ta_best_tf_val*100:.1f}% of GT)")

    # Rank correlation
    try:
        from scipy.stats import spearmanr, kendalltau
        rho, _ = spearmanr(merged["ta_tile_rank"], merged["pm_tile_rank"])
        tau, _ = kendalltau(merged["ta_tile_rank"], merged["pm_tile_rank"])
        print(f"  Spearman rho={rho:.3f}  Kendall tau={tau:.3f}  "
              f"(across {len(merged)} tiles)")
    except ImportError:
        pass

    # ── PM's tile-#1 vs TA's tile-#1 ──────────────────────────────────────────
    ta_t1 = merged.iloc[0]
    pm_t1 = merged.sort_values("pm_best_tflops", ascending=False).iloc[0]
    print()
    print(f"Tile summary:")
    print(f"  TA #1 tile: BM={int(ta_t1.cfg_BLOCK_SIZE_M)} BN={int(ta_t1.cfg_BLOCK_SIZE_N)} "
          f"BK={int(ta_t1.cfg_BLOCK_SIZE_K)}  → TA {ta_t1.ta_best_tflops:.1f}, "
          f"PM {ta_t1.pm_best_tflops:.1f} (PM tile rank {int(ta_t1.pm_tile_rank)})")
    print(f"  PM #1 tile: BM={int(pm_t1.cfg_BLOCK_SIZE_M)} BN={int(pm_t1.cfg_BLOCK_SIZE_N)} "
          f"BK={int(pm_t1.cfg_BLOCK_SIZE_K)}  → PM {pm_t1.pm_best_tflops:.1f}, "
          f"TA {pm_t1.ta_best_tflops:.1f} (TA tile rank {int(pm_t1.ta_tile_rank)})")
    print(f"  PM-pick tile achieves {pm_t1.ta_best_tflops/ta_t1.ta_best_tflops*100:.1f}% "
          f"of ground-truth tile TFLOPS")

    # ── HW counter delta: best config of TA-#1 tile vs best of PM-#1 tile ────
    # Pick the row with max measured TFLOPS for each tile (so we compare the
    # tile's best achievable config, not arbitrary cache modifier choices).
    ta_t1_key = tuple(ta_t1[tile_key])
    pm_t1_key = tuple(pm_t1[tile_key])
    def _best_row(key):
        mask = ((df["cfg_BLOCK_SIZE_M"] == key[0]) &
                (df["cfg_BLOCK_SIZE_N"] == key[1]) &
                (df["cfg_BLOCK_SIZE_K"] == key[2]))
        return df[mask].sort_values("tflops", ascending=False).iloc[0]
    ta_best = _best_row(ta_t1_key)
    pm_best = _best_row(pm_t1_key)

    hw_cols = hw_counter_columns(df)
    # Subset to the "summary" counters most relevant for diagnosis
    diagnostic = [
        "hw_MFMA_util_%", "hw_VALU_util_%",
        "hw_L2_hit_rate_%", "hw_L2_cache_bw_GBps", "hw_L2_access_latency_cycles",
        "hw_vL1D_cache_bw_GBps", "hw_mem_unit_busy_%",
        "hw_occupancy_%", "hw_simd_utilization_%",
        "hw_LDS_bank_conflict_rate_%",
        "hw_wave_dep_wait_%", "hw_wave_issue_wait_%", "hw_wave_exec_%",
        "hw_IPC",
    ]
    print()
    print(f"=== HW counter side-by-side ===  "
          f"(best-config-of-tile for each)")
    print(f"  TA winner: BM={ta_t1_key[0]} BN={ta_t1_key[1]} BK={ta_t1_key[2]}  "
          f"({ta_best['config_str']})")
    print(f"  PM winner: BM={pm_t1_key[0]} BN={pm_t1_key[1]} BK={pm_t1_key[2]}  "
          f"({pm_best['config_str']})")
    print()
    print(f"  {'counter':<32} {'TA':>10} {'PM':>10} {'PM-TA':>10}")
    for c in diagnostic:
        if c not in df.columns: continue
        ta_v = ta_best[c]; pm_v = pm_best[c]
        try:
            delta = pm_v - ta_v
            print(f"  {param_name(c):<32} {ta_v:>10.2f} {pm_v:>10.2f} {delta:>+10.2f}")
        except Exception:
            pass
    # Also show timing breakdown
    print()
    print(f"  {'metric':<32} {'TA':>10} {'PM':>10}")
    for k, fmt in [("tflops","{:>10.1f}"),("timing_us","{:>10.3f}"),
                   ("timing_std_ns","{:>10.0f}")]:
        if k in df.columns:
            print(f"  {k:<32} " + fmt.format(ta_best[k]) + " " + fmt.format(pm_best[k]))

    # ── IR feature side-by-side (kernel codegen-level differences) ───────────
    ir_cols = ir_feature_columns(df)
    if ir_cols:
        print()
        print(f"=== IR features (kernel codegen differences) ===")
        print(f"  {'feature':<36} {'TA':>14} {'PM':>14} {'delta':>10}")
        # Pre-defined diagnostic order
        ordered = [
            "ir_num_vgprs", "ir_num_sgprs", "ir_shared", "ir_spills", "ir_sgpr_spills",
            "ir_num_mfma", "ir_mfma_cycles_total",
            "ir_num_async_copy", "ir_num_local_load", "ir_num_local_store",
            "ir_num_global_load", "ir_num_global_store",
            "ir_num_buffer_load", "ir_num_buffer_store",
            "ir_num_ds_read", "ir_num_ds_write",
            "ir_num_waitcnt", "ir_num_waitcnt_vm", "ir_num_waitcnt_lgkm",
            "ir_mean_waitcnt_vm", "ir_mean_waitcnt_lgkm",
            "ir_num_valu", "ir_num_salu", "ir_num_branch",
            "ir_compute_budget_cycles",
            "ir_global_load_bytes_per_lane", "ir_global_store_bytes_per_lane",
            "ir_buffer_load_bytes_per_lane", "ir_buffer_store_bytes_per_lane",
            "ir_ds_read_bytes_per_lane", "ir_ds_write_bytes_per_lane",
        ]
        seen = set()
        def _fmt_val(v):
            if v is None or (isinstance(v, float) and v != v):  # NaN
                return "—"
            if isinstance(v, (int,)):
                return f"{v:>14d}"
            if isinstance(v, float):
                return f"{v:>14.2f}"
            return f"{str(v):>14}"
        for c in ordered:
            if c not in df.columns: continue
            seen.add(c)
            ta_v, pm_v = ta_best.get(c), pm_best.get(c)
            try:
                delta = pm_v - ta_v if (ta_v is not None and pm_v is not None) else None
                if delta is None:
                    drepr = ""
                elif isinstance(delta, float):
                    drepr = f"{delta:>+10.2f}"
                else:
                    drepr = f"{delta:>+10d}"
            except TypeError:
                drepr = ""  # non-numeric (list/dict)
            print(f"  {param_name(c):<36} {_fmt_val(ta_v):>14} {_fmt_val(pm_v):>14} {drepr}")
        # Show any remaining (list-valued / categorical) features
        for c in ir_cols:
            if c in seen: continue
            ta_v, pm_v = ta_best.get(c), pm_best.get(c)
            if ta_v == pm_v: continue
            print(f"  {param_name(c):<36} {str(ta_v)[:14]:>14} {str(pm_v)[:14]:>14}")


if __name__ == "__main__":
    main()
