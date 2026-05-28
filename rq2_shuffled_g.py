#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq2_shuffled_g.py
------------------
Within-episode temporal-shuffle ablation for Φ_r.

For each episode, we compute Φ_r over the latent trajectories in TWO
versions:
  - original:  g_t in true temporal order
  - shuffled:  g_t with timesteps randomly permuted (within episode)

This preserves the marginal distribution of g exactly while destroying
temporal structure. The same is done for z as a sanity check.

KEY QUESTION:
  Does Φ_r(g) depend on temporal dynamics, or on the marginal shape of g?

PREDICTIONS:
  - If Φ_r(g_shuffled) ≈ 0  → temporal dynamics drive Φ_r(g).  Claim ✓
  - If Φ_r(g_shuffled) ≈ Φ_r(g_orig)  → marginal shape drives Φ_r(g).  Claim ✗

Usage:
  PYTHONPATH=. python rq2_shuffled_g.py
  PYTHONPATH=. python rq2_shuffled_g.py --n_shuffles 5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from cear_pilot.analysis.phi.phi_r import phi_r_from_trajectory

SEEDS = [1, 2, 3, 4, 5]
OUTDIR = Path("outputs")


def shuffle_temporal(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the time axis of X (shape T, d). Preserves marginal exactly."""
    perm = rng.permutation(X.shape[0])
    return X[perm]


def episode_phir_pair(ep_df, z_cols, g_cols, rng, n_shuffles=1):
    """Compute Φ_r for original and shuffled versions of z and g.
       Shuffle multiple times and average (reduces shuffle-noise)."""
    Z = ep_df[z_cols].values
    G = ep_df[g_cols].values

    out = {
        "phi_z_orig":  phi_r_from_trajectory(Z),
        "phi_g_orig":  phi_r_from_trajectory(G),
    }

    # Shuffle z and g independently, average over multiple shuffles
    phi_z_shuf, phi_g_shuf = [], []
    for _ in range(n_shuffles):
        phi_z_shuf.append(phi_r_from_trajectory(shuffle_temporal(Z, rng)))
        phi_g_shuf.append(phi_r_from_trajectory(shuffle_temporal(G, rng)))
    out["phi_z_shuf"] = float(np.mean(phi_z_shuf))
    out["phi_g_shuf"] = float(np.mean(phi_g_shuf))
    return out


def process_condition(condition: str, n_shuffles: int, rng_seed: int = 12345):
    """Run shuffled-g ablation across all seeds for one condition."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for seed in SEEDS:
        path = OUTDIR / f"replay_seed{seed}_{condition}" / "traj.parquet"
        if not path.exists():
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_parquet(path)
        z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                        key=lambda c: int(c.split("_")[1]))
        g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                        key=lambda c: int(c.split("_")[1]))

        for ep_id, ep in df.groupby("episode"):
            out = episode_phir_pair(ep, z_cols, g_cols, rng, n_shuffles)
            rows.append({"seed": seed, "episode": int(ep_id),
                         "condition": condition, **out})
    return pd.DataFrame(rows)


def main(n_shuffles: int):
    print(f"\n{'='*78}")
    print("RQ2: Shuffled-g ablation — does Φ_r(g) need temporal structure?")
    print(f"{'='*78}")
    print(f"  shuffles per episode: {n_shuffles}")
    print(f"  seeds: {SEEDS}")
    print(f"{'='*78}\n")

    # --- collect data ---
    print("Computing Φ_r for original vs shuffled trajectories...")
    print("  [condition: clean]")
    clean = process_condition("clean", n_shuffles)
    print("  [condition: p20]")
    p20 = process_condition("p20", n_shuffles)

    if len(clean) == 0:
        print("\nNo clean data. Run collect_all_seeds.sh first.")
        return

    # ============================================================
    # PRIMARY RESULT: Does temporal shuffle destroy Φ_r(g)?
    # ============================================================
    print(f"\n{'='*78}")
    print("PRIMARY RESULT: Effect of temporal shuffle on Φ_r(g)")
    print(f"{'='*78}")

    for label, data in [("clean", clean), ("p20", p20)]:
        if len(data) == 0:
            continue
        g_orig = data["phi_g_orig"].values
        g_shuf = data["phi_g_shuf"].values
        ratio_collapse = (1 - g_shuf.mean() / g_orig.mean()) * 100  # % drop
        t_stat, p_val = ttest_rel(g_orig, g_shuf)
        cohens_d = (g_orig - g_shuf).mean() / (g_orig - g_shuf).std()

        print(f"\n  [{label}]  n={len(data)} episodes (5 seeds × 10 ep)")
        print(f"     Φ_r(g) original:  {g_orig.mean():+.4f}  (std {g_orig.std():.3f})")
        print(f"     Φ_r(g) shuffled:  {g_shuf.mean():+.4f}  (std {g_shuf.std():.3f})")
        print(f"     Drop:             {(g_orig - g_shuf).mean():+.4f}  "
              f"({ratio_collapse:.1f}% collapse)")
        print(f"     Paired t-test:    t = {t_stat:.2f},  p = {p_val:.2e},  "
              f"Cohen's d = {cohens_d:.2f}")

    # ============================================================
    # CONTROL: z shuffle effect (should be tiny — z is already near noise)
    # ============================================================
    print(f"\n{'='*78}")
    print("CONTROL: Effect of temporal shuffle on Φ_r(z)")
    print(f"{'='*78}")

    for label, data in [("clean", clean), ("p20", p20)]:
        if len(data) == 0:
            continue
        z_orig = data["phi_z_orig"].values
        z_shuf = data["phi_z_shuf"].values
        t_stat, p_val = ttest_rel(z_orig, z_shuf)

        print(f"\n  [{label}]  n={len(data)}")
        print(f"     Φ_r(z) original:  {z_orig.mean():+.4f}")
        print(f"     Φ_r(z) shuffled:  {z_shuf.mean():+.4f}")
        print(f"     Drop:             {(z_orig - z_shuf).mean():+.4f}")
        print(f"     Paired t-test:    t = {t_stat:.2f},  p = {p_val:.2e}")

    # ============================================================
    # ASYMMETRY: does the g-shuffle drop g far more than z-shuffle drops z?
    # ============================================================
    print(f"\n{'='*78}")
    print("ASYMMETRY: g vs z shuffle sensitivity")
    print(f"{'='*78}")
    for label, data in [("clean", clean), ("p20", p20)]:
        if len(data) == 0:
            continue
        drop_g = (data["phi_g_orig"] - data["phi_g_shuf"]).mean()
        drop_z = (data["phi_z_orig"] - data["phi_z_shuf"]).mean()
        ratio = drop_g / max(abs(drop_z), 1e-6)
        print(f"  [{label}]  Δg from shuffle: {drop_g:+.4f}   "
              f"Δz from shuffle: {drop_z:+.4f}   "
              f"|Δg/Δz| = {ratio:.1f}x")

    # ============================================================
    # Per-seed breakdown (for paper figure)
    # ============================================================
    print(f"\n{'='*78}")
    print("PER-SEED BREAKDOWN (clean condition)")
    print(f"{'='*78}")
    print(f"\n  {'seed':>4} | {'Φ_r(g) orig':>12} | {'Φ_r(g) shuf':>12} | "
          f"{'% collapse':>11} | {'p':>8}")
    print("  " + "-"*60)
    for seed, sub in clean.groupby("seed"):
        g_o = sub["phi_g_orig"].mean()
        g_s = sub["phi_g_shuf"].mean()
        collapse = (1 - g_s / g_o) * 100 if g_o > 0 else float("nan")
        try:
            _, p = wilcoxon(sub["phi_g_orig"], sub["phi_g_shuf"],
                            alternative="greater")
        except ValueError:
            p = float("nan")
        print(f"  {seed:>4} | {g_o:>+12.4f} | {g_s:>+12.4f} | "
              f"{collapse:>10.1f}% | {p:>8.4f}")

    # --- save ---
    out_csv = OUTDIR / "rq2_shuffled_g.csv"
    combined = pd.concat([clean, p20], ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}")
    print(f"{'='*78}\n")

    # ============================================================
    # Interpretation
    # ============================================================
    g_collapse_clean = (1 - clean["phi_g_shuf"].mean() /
                        clean["phi_g_orig"].mean()) * 100
    print("READING THIS")
    print("="*78)
    if g_collapse_clean > 80:
        print(f"  → Φ_r(g) collapses by {g_collapse_clean:.0f}% under temporal shuffle.")
        print(f"    This is a STRONG result: Φ_r(g) depends on temporal dynamics,")
        print(f"    NOT on the marginal distribution of g.")
        print(f"    History-sensitivity claim is causally supported.")
    elif g_collapse_clean > 50:
        print(f"  → Φ_r(g) drops by {g_collapse_clean:.0f}% under shuffle (substantial).")
        print(f"    Temporal structure is the primary driver, but some marginal")
        print(f"    contribution remains. Report both effects.")
    else:
        print(f"  → Φ_r(g) drops only {g_collapse_clean:.0f}% under shuffle.")
        print(f"    This WEAKENS the temporal-dynamics claim. Φ_r(g) appears")
        print(f"    partially driven by g's marginal shape. Reframe needed.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shuffles", type=int, default=3,
                    help="Number of independent shuffles per episode (averaged). "
                         "More shuffles → less shuffle-noise, slower. Default 3.")
    args = ap.parse_args()
    main(args.n_shuffles)
