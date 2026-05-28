#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq3_trained_vs_untrained.py
----------------------------
The critical test:

  Is Φ_r(g) the product of LEARNING, or just of ARCHITECTURE?

We compare Φ_r over g, z, and joint [z,g] in two model groups
that share architecture exactly:
  - TRAINED:    weights from 48k-step training (outputs/runs/seedN)
  - UNTRAINED:  same architecture, random init (outputs/runs_untrained/seedN)

PREDICTIONS:
  - Framing X (learning instantiates the slow integrative mode):
      Φ_r(g)_trained ≫ Φ_r(g)_untrained  (untrained ≈ noise floor)
  - Framing Y (Φ_r(g) is trivially high due to GRU architecture):
      Φ_r(g)_trained ≈ Φ_r(g)_untrained

Usage:
  PYTHONPATH=. python rq3_trained_vs_untrained.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from cear_pilot.analysis.phi.phi_r import phi_r_zgj

SEEDS = [1, 2, 3, 4, 5]
OUTDIR = Path("outputs")


def collect_phir(prefix: str, condition: str):
    """Returns DataFrame with per-episode Φ_r for each seed."""
    rows = []
    for seed in SEEDS:
        path = OUTDIR / f"{prefix}_seed{seed}_{condition}" / "traj.parquet"
        if not path.exists():
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_parquet(path)
        z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                        key=lambda c: int(c.split("_")[1]))
        g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                        key=lambda c: int(c.split("_")[1]))
        for ep_id, ep in df.groupby("episode"):
            out = phi_r_zgj(ep[z_cols].values, ep[g_cols].values,
                            force_zg_split_in_joint=True)
            rows.append({"seed": seed, "episode": int(ep_id),
                         "condition": condition, **out})
    return pd.DataFrame(rows)


def report(label: str, trained: pd.DataFrame, untrained: pd.DataFrame, metric: str):
    """Side-by-side stats for one Φ_r metric."""
    tr = trained[metric].values
    un = untrained[metric].values
    # Independent two-sample test (different model populations)
    t_stat, p_val = ttest_ind(tr, un, equal_var=False)
    u_stat, u_p = mannwhitneyu(tr, un, alternative="greater")
    pooled_sd = np.sqrt((tr.var(ddof=1) + un.var(ddof=1)) / 2)
    d = (tr.mean() - un.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
    ratio = tr.mean() / un.mean() if abs(un.mean()) > 1e-6 else float("inf")
    print(f"\n  [{label}]  {metric}")
    print(f"     trained:    mean = {tr.mean():+.4f},  std = {tr.std():.4f}  (n={len(tr)})")
    print(f"     untrained:  mean = {un.mean():+.4f},  std = {un.std():.4f}  (n={len(un)})")
    print(f"     ratio:      {ratio:.1f}x")
    print(f"     Welch's t:  t = {t_stat:.2f},  p = {p_val:.2e}")
    print(f"     Mann-Whitney (tr>un):  U = {u_stat:.0f},  p = {u_p:.2e}")
    print(f"     Cohen's d:  {d:.2f}")


def main():
    print(f"\n{'='*78}")
    print("RQ3: Is Φ_r(g) the product of LEARNING or of ARCHITECTURE?")
    print(f"{'='*78}")
    print("  Comparing trained (48k-step) vs untrained (random-init) ckpts")
    print("  Same architecture, same env, same protocol.")
    print(f"{'='*78}\n")

    print("Loading trained replays...")
    tr_clean = collect_phir("replay", "clean")
    tr_p20 = collect_phir("replay", "p20")

    print("Loading untrained replays...")
    un_clean = collect_phir("replay_untrained", "clean")
    un_p20 = collect_phir("replay_untrained", "p20")

    if len(un_clean) == 0:
        print("\n[ERR] No untrained data found. Run:")
        print("       PYTHONPATH=. python make_untrained_ckpts.py")
        print("       bash collect_untrained.sh")
        return

    # ===========================================================
    # MAIN COMPARISON: clean condition
    # ===========================================================
    print(f"\n{'='*78}")
    print("MAIN COMPARISON (clean condition, n=50 vs n=50)")
    print(f"{'='*78}")
    for metric in ["phi_r_z", "phi_r_g", "phi_r_joint"]:
        report("clean", tr_clean, un_clean, metric)

    # ===========================================================
    # Regime sensitivity comparison
    # ===========================================================
    print(f"\n{'='*78}")
    print("REGIME SENSITIVITY: do untrained models also show Φ_r(g) drop?")
    print(f"{'='*78}")
    # In trained: pre/post split; in untrained: full episode (no learned dynamics
    # to be perturbed). So we just compare aggregate Φ_r(g) across conditions.
    if len(un_p20) > 0:
        for metric in ["phi_r_z", "phi_r_g"]:
            print(f"\n  {metric}:")
            print(f"     trained clean:    {tr_clean[metric].mean():+.4f}")
            print(f"     trained p20:      {tr_p20[metric].mean():+.4f}")
            print(f"     untrained clean:  {un_clean[metric].mean():+.4f}")
            print(f"     untrained p20:    {un_p20[metric].mean():+.4f}")
            tr_diff = tr_p20[metric].mean() - tr_clean[metric].mean()
            un_diff = un_p20[metric].mean() - un_clean[metric].mean()
            print(f"     trained delta:    {tr_diff:+.4f}")
            print(f"     untrained delta:  {un_diff:+.4f}")

    # ===========================================================
    # Per-seed breakdown
    # ===========================================================
    print(f"\n{'='*78}")
    print("PER-SEED Φ_r(g) — clean condition")
    print(f"{'='*78}")
    print(f"\n  {'seed':>4} | {'trained':>10} | {'untrained':>10} | "
          f"{'ratio':>8} | {'Δ':>10}")
    print("  " + "-"*58)
    for seed in SEEDS:
        tr_seed = tr_clean[tr_clean.seed == seed]["phi_r_g"].mean()
        un_seed = un_clean[un_clean.seed == seed]["phi_r_g"].mean()
        if abs(un_seed) > 1e-6:
            r = tr_seed / un_seed
            r_str = f"{r:.1f}x"
        else:
            r_str = "∞"
        print(f"  {seed:>4} | {tr_seed:>+10.4f} | {un_seed:>+10.4f} | "
              f"{r_str:>8} | {tr_seed - un_seed:>+10.4f}")

    # ===========================================================
    # Save and interpret
    # ===========================================================
    tr_clean["group"] = "trained"
    un_clean["group"] = "untrained"
    tr_p20["group"] = "trained"
    un_p20["group"] = "untrained"
    all_data = pd.concat([tr_clean, un_clean, tr_p20, un_p20], ignore_index=True)
    out_csv = OUTDIR / "rq3_trained_vs_untrained.csv"
    all_data.to_csv(out_csv, index=False)

    # Interpretation
    tr_mean = tr_clean["phi_r_g"].mean()
    un_mean = un_clean["phi_r_g"].mean()
    ratio = tr_mean / un_mean if abs(un_mean) > 1e-6 else float("inf")

    print(f"\n{'='*78}")
    print("READING THIS")
    print(f"{'='*78}")
    if un_mean < 0.1 and ratio > 10:
        print(f"  ✓ DECISIVE: Φ_r(g) is a PRODUCT OF LEARNING.")
        print(f"    Untrained Φ_r(g) = {un_mean:+.3f} (near noise floor)")
        print(f"    Trained   Φ_r(g) = {tr_mean:+.3f}  ({ratio:.0f}x higher)")
        print(f"    The GRU architecture alone does not produce ΦID-relevant")
        print(f"    organization; learning is required to instantiate it.")
        print(f"    Framing X confirmed. The trivial-architecture concern is rejected.")
    elif ratio > 3:
        print(f"  ~ PARTIAL: Both architecture and learning contribute.")
        print(f"    Untrained Φ_r(g) = {un_mean:+.3f} (non-negligible)")
        print(f"    Trained   Φ_r(g) = {tr_mean:+.3f}  ({ratio:.1f}x higher)")
        print(f"    Learning amplifies architectural baseline. Paper claim needs")
        print(f"    to be framed as 'learning increases the architecturally")
        print(f"    available ΦID-relevant organization'.")
    else:
        print(f"  ✗ CONCERNING: Φ_r(g) similar for trained and untrained.")
        print(f"    Untrained Φ_r(g) = {un_mean:+.3f}")
        print(f"    Trained   Φ_r(g) = {tr_mean:+.3f}  (only {ratio:.1f}x)")
        print(f"    The GRU architecture alone produces most of the signal.")
        print(f"    Framing X is weakened. Major reframing of paper needed.")

    print(f"\n  Saved → {out_csv}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
