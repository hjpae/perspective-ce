#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq_v2_full.py
--------------
Unified v2 analysis across 30 seeds, trained + untrained, with warmup
exclusion. Produces the four core results of the paper:

  Part 1  Φ_r decomposition: Φ_r concentrates in g (not z)
  Part 2  Regime sensitivity: Φ_r(g) drops after switch
  Part 3  Shuffled-g ablation: temporal structure drives Φ_r(g)
  Part 4  Learning-attributable component: trained vs untrained

Protocol (v2):
  T = 500, warmup = 80 (excluded), switch at t = 320.
  pre  = (80, 320]   (240 steps)
  post = (320, 500]  (180 steps)

Usage:
  PYTHONPATH=. python rq_v2_full.py
  PYTHONPATH=. python rq_v2_full.py --n_shuffles 3 --n_seeds 30
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, wilcoxon

from cear_pilot.analysis.phi.phi_r import phi_r_from_trajectory, phi_r_zgj

warnings.filterwarnings("ignore")

OUTDIR = Path("outputs")
WARMUP = 80
T_SWITCH = 320


def load_episode_latents(df, z_cols, g_cols, t_lo=None, t_hi=None):
    """Return list of (Z, G) arrays per episode, optionally windowed by t."""
    out = []
    for ep_id, ep in df.groupby("episode"):
        if t_lo is not None:
            ep = ep[ep.t > t_lo]
        if t_hi is not None:
            ep = ep[ep.t <= t_hi]
        if len(ep) < 30:
            continue
        out.append((ep_id, ep[z_cols].values, ep[g_cols].values))
    return out


def get_cols(df):
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda c: int(c.split("_")[1]))
    return z_cols, g_cols


def shuffle_temporal(X, rng):
    return X[rng.permutation(X.shape[0])]


# =====================================================================
#  Data collection helpers
# =====================================================================

def collect_decomp(group_prefix, condition, seeds, t_lo=WARMUP, t_hi=None):
    """Φ_r(z), Φ_r(g), Φ_r([z,g]) per episode, windowed t in (t_lo, t_hi]."""
    rows = []
    for seed in seeds:
        path = OUTDIR / f"{group_prefix}_seed{seed}_v2_{condition}" / "traj.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        z_cols, g_cols = get_cols(df)
        for ep_id, Z, G in load_episode_latents(df, z_cols, g_cols, t_lo, t_hi):
            out = phi_r_zgj(Z, G, force_zg_split_in_joint=True)
            rows.append({"seed": seed, "episode": ep_id, **out})
    return pd.DataFrame(rows)


def collect_prepost(group_prefix, seeds):
    """For p20 condition: Φ_r(g/z) in pre vs post window per episode."""
    rows = []
    for seed in seeds:
        path = OUTDIR / f"{group_prefix}_seed{seed}_v2_p20" / "traj.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        z_cols, g_cols = get_cols(df)
        for ep_id, ep in df.groupby("episode"):
            pre = ep[(ep.t > WARMUP) & (ep.t <= T_SWITCH)]
            post = ep[ep.t > T_SWITCH]
            if len(pre) < 30 or len(post) < 30:
                continue
            pre_out = phi_r_zgj(pre[z_cols].values, pre[g_cols].values)
            post_out = phi_r_zgj(post[z_cols].values, post[g_cols].values)
            rows.append({
                "seed": seed, "episode": ep_id,
                "phi_z_pre": pre_out["phi_r_z"], "phi_z_post": post_out["phi_r_z"],
                "phi_g_pre": pre_out["phi_r_g"], "phi_g_post": post_out["phi_r_g"],
            })
    return pd.DataFrame(rows)


def collect_shuffle(group_prefix, condition, seeds, n_shuffles, rng):
    """Φ_r(g) original vs temporally-shuffled."""
    rows = []
    for seed in seeds:
        path = OUTDIR / f"{group_prefix}_seed{seed}_v2_{condition}" / "traj.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        z_cols, g_cols = get_cols(df)
        for ep_id, Z, G in load_episode_latents(df, z_cols, g_cols, WARMUP, None):
            g_orig = phi_r_from_trajectory(G)
            g_shuf = np.mean([phi_r_from_trajectory(shuffle_temporal(G, rng))
                              for _ in range(n_shuffles)])
            rows.append({"seed": seed, "episode": ep_id,
                         "phi_g_orig": g_orig, "phi_g_shuf": float(g_shuf)})
    return pd.DataFrame(rows)


# =====================================================================
#  Main
# =====================================================================

def main(n_seeds, n_shuffles):
    seeds = list(range(1, n_seeds + 1))
    rng = np.random.default_rng(12345)

    print(f"\n{'='*80}")
    print(f"RQ v2 FULL ANALYSIS — {n_seeds} seeds, warmup={WARMUP}, switch={T_SWITCH}")
    print(f"{'='*80}")

    # =================================================================
    print(f"\nLoading data (this takes a few minutes)...")
    tr_clean = collect_decomp("replay", "clean", seeds)
    un_clean = collect_decomp("replay_untrained", "clean", seeds)
    tr_pp = collect_prepost("replay", seeds)
    un_pp = collect_prepost("replay_untrained", seeds)
    print(f"  trained clean:   {len(tr_clean)} episodes")
    print(f"  untrained clean: {len(un_clean)} episodes")
    print(f"  trained p20:     {len(tr_pp)} episodes")
    print(f"  untrained p20:   {len(un_pp)} episodes")

    # =================================================================
    # PART 1: Φ_r concentrates in g
    # =================================================================
    print(f"\n{'='*80}")
    print("PART 1: Φ_r concentrates in g, not z  (trained, clean)")
    print(f"{'='*80}")
    z, g = tr_clean["phi_r_z"].values, tr_clean["phi_r_g"].values
    diff = g - z
    t1, p1 = ttest_1samp(diff, 0)
    print(f"  n = {len(tr_clean)} episodes")
    print(f"  Φ_r(z) = {z.mean():+.4f} ± {z.std():.4f}")
    print(f"  Φ_r(g) = {g.mean():+.4f} ± {g.std():.4f}")
    print(f"  ratio g/z = {g.mean()/max(z.mean(),1e-6):.1f}x")
    print(f"  paired t = {t1:.2f}, p = {p1:.2e}")
    # all-seeds consistency
    consistent = all(
        tr_clean[tr_clean.seed == s]["phi_r_g"].mean() >
        tr_clean[tr_clean.seed == s]["phi_r_z"].mean()
        for s in seeds if (tr_clean.seed == s).any()
    )
    print(f"  g > z in ALL {n_seeds} seeds: {consistent}")

    # =================================================================
    # PART 2: Regime sensitivity (trained)
    # =================================================================
    print(f"\n{'='*80}")
    print("PART 2: Φ_r(g) regime sensitivity  (trained, p20: pre vs post)")
    print(f"{'='*80}")
    dg_tr = tr_pp["phi_g_post"].values - tr_pp["phi_g_pre"].values
    dz_tr = tr_pp["phi_z_post"].values - tr_pp["phi_z_pre"].values
    t2g, p2g = ttest_1samp(dg_tr, 0)
    t2z, p2z = ttest_1samp(dz_tr, 0)
    print(f"  n = {len(tr_pp)} episodes")
    print(f"  Φ_r(g): pre {tr_pp['phi_g_pre'].mean():+.4f} → post {tr_pp['phi_g_post'].mean():+.4f}"
          f"   Δ = {dg_tr.mean():+.4f}  (t={t2g:.2f}, p={p2g:.2e}, d={dg_tr.mean()/dg_tr.std():.2f})")
    print(f"  Φ_r(z): pre {tr_pp['phi_z_pre'].mean():+.4f} → post {tr_pp['phi_z_post'].mean():+.4f}"
          f"   Δ = {dz_tr.mean():+.4f}  (t={t2z:.2f}, p={p2z:.2e})")
    print(f"  asymmetry |Δg/Δz| = {abs(dg_tr.mean())/max(abs(dz_tr.mean()),1e-6):.1f}x")

    # =================================================================
    # PART 3: Shuffled-g ablation
    # =================================================================
    print(f"\n{'='*80}")
    print("PART 3: Shuffled-g ablation  (trained, clean)")
    print(f"{'='*80}")
    sh = collect_shuffle("replay", "clean", seeds, n_shuffles, rng)
    g_o, g_s = sh["phi_g_orig"].values, sh["phi_g_shuf"].values
    collapse = (1 - g_s.mean() / g_o.mean()) * 100
    t3, p3 = ttest_rel(g_o, g_s)
    print(f"  n = {len(sh)} episodes, {n_shuffles} shuffles each")
    print(f"  Φ_r(g) original: {g_o.mean():+.4f}")
    print(f"  Φ_r(g) shuffled: {g_s.mean():+.4f}")
    print(f"  collapse: {collapse:.1f}%   (t={t3:.2f}, p={p3:.2e}, d={(g_o-g_s).mean()/(g_o-g_s).std():.2f})")

    # =================================================================
    # PART 4: Learning-attributable component
    # =================================================================
    print(f"\n{'='*80}")
    print("PART 4: Learning-attributable component  (trained vs untrained)")
    print(f"{'='*80}")

    # 4a. Φ_r(g) magnitude: clean
    tr_g = tr_clean["phi_r_g"].values
    un_g = un_clean["phi_r_g"].values
    t4, p4 = ttest_ind(tr_g, un_g, equal_var=False)
    print(f"\n  [4a] Φ_r(g) magnitude (clean):")
    print(f"       trained:   {tr_g.mean():+.4f} ± {tr_g.std():.4f}")
    print(f"       untrained: {un_g.mean():+.4f} ± {un_g.std():.4f}")
    print(f"       Δ(learned) = {tr_g.mean()-un_g.mean():+.4f}  (Welch t={t4:.2f}, p={p4:.2e})")

    # 4b. Regime sensitivity: trained vs untrained
    dg_un = un_pp["phi_g_post"].values - un_pp["phi_g_pre"].values
    t4b, p4b = ttest_ind(dg_tr, dg_un, equal_var=False)
    print(f"\n  [4b] Φ_r(g) regime drop (p20):")
    print(f"       trained Δ:   {dg_tr.mean():+.4f}")
    print(f"       untrained Δ: {dg_un.mean():+.4f}")
    print(f"       ΔΔΦ_r (learning-attributable regime sensitivity) = "
          f"{dg_tr.mean()-dg_un.mean():+.4f}")
    print(f"       Welch t = {t4b:.2f}, p = {p4b:.2e}")
    print(f"\n       ★ This is the key result: regime sensitivity is")
    print(f"         {abs(dg_tr.mean())/max(abs(dg_un.mean()),1e-6):.1f}x larger in trained agents.")

    # =================================================================
    # Summary table
    # =================================================================
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n  {'Quantity':<32} | {'Untrained':>11} | {'Trained':>11} | {'Δ(learned)':>11}")
    print("  " + "-"*72)
    print(f"  {'Φ_r(g) clean':<32} | {un_g.mean():>+11.4f} | {tr_g.mean():>+11.4f} | "
          f"{tr_g.mean()-un_g.mean():>+11.4f}")
    print(f"  {'Φ_r(g) regime drop (p20)':<32} | {dg_un.mean():>+11.4f} | {dg_tr.mean():>+11.4f} | "
          f"{dg_tr.mean()-dg_un.mean():>+11.4f}")
    print(f"  {'Φ_r(g) shuffled (clean)':<32} | {'~0':>11} | {g_s.mean():>+11.4f} | "
          f"{'—':>11}")
    print(f"  {'Φ_r(g)/Φ_r(z) ratio':<32} | {'—':>11} | "
          f"{g.mean()/max(z.mean(),1e-6):>10.1f}x | {'—':>11}")

    # =================================================================
    # Save everything
    # =================================================================
    tr_clean["group"], un_clean["group"] = "trained", "untrained"
    tr_pp["group"], un_pp["group"] = "trained", "untrained"
    sh["group"] = "trained"
    tr_clean.to_csv(OUTDIR / "v2_decomp_trained.csv", index=False)
    un_clean.to_csv(OUTDIR / "v2_decomp_untrained.csv", index=False)
    pd.concat([tr_pp, un_pp]).to_csv(OUTDIR / "v2_prepost.csv", index=False)
    sh.to_csv(OUTDIR / "v2_shuffle.csv", index=False)
    print(f"\n  Saved CSVs to {OUTDIR}/v2_*.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--n_shuffles", type=int, default=3)
    args = ap.parse_args()
    main(args.n_seeds, args.n_shuffles)
