#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq_hysteresis.py
-----------------
Switch-aligned sliding-window Φ_r(g) trajectories — the ΦID analogue of
the g-score hysteresis curves (cf. AAAI Fig. 4).

For each episode (p20 condition), we slide a window of length W along the
g trajectory, computing Φ_r(g) at each position. Windows are aligned to
the regime switch (t=320): tau = (window_center - T_SWITCH).

PREDICTION:
  - trained:   Φ_r(g) moves SLOWLY after switch (hysteresis / slow adapt)
  - untrained: Φ_r(g) drops SHARPLY and immediately (no hysteresis)

Also runs a window-stability check: in the pure-pre region, Φ_r variance
across windows should be modest (confirms W is large enough).

Outputs:
  outputs/hysteresis_curves.csv   (tau, group, seed, phi_g, ...)

Usage:
  PYTHONPATH=. python rq_hysteresis.py
  PYTHONPATH=. python rq_hysteresis.py --window 80 --stride 5 --n_seeds 30
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cear_pilot.analysis.phi.phi_r import phi_r_from_trajectory

warnings.filterwarnings("ignore")

OUTDIR = Path("outputs")
WARMUP = 80
T_SWITCH = 320


def get_g_cols(df):
    return sorted([c for c in df.columns if c.startswith("g_")],
                  key=lambda c: int(c.split("_")[1]))


def sliding_phir(G_full, t_vals, window, stride):
    """
    Compute Φ_r(g) over sliding windows.
    G_full: (T, d_g) for one episode (full, including warmup).
    t_vals: (T,) the actual t index for each row.
    Returns list of (tau, phi_g) where tau = window_center - T_SWITCH.
    """
    out = []
    T = G_full.shape[0]
    start = 0
    while start + window <= T:
        seg = G_full[start:start + window]
        center_t = t_vals[start + window // 2]
        # skip windows whose center is in the warmup region
        if center_t > WARMUP - window // 2:
            phi = phi_r_from_trajectory(seg)
            tau = center_t - T_SWITCH
            out.append((tau, phi))
        start += stride
    return out


def collect_hysteresis(group_prefix, seeds, window, stride):
    rows = []
    for seed in seeds:
        path = OUTDIR / f"{group_prefix}_seed{seed}_v2_p20" / "traj.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        g_cols = get_g_cols(df)
        for ep_id, ep in df.groupby("episode"):
            ep = ep.sort_values("t")
            G = ep[g_cols].values
            t_vals = ep["t"].values
            for tau, phi in sliding_phir(G, t_vals, window, stride):
                rows.append({"group": group_prefix, "seed": seed,
                             "episode": ep_id, "tau": tau, "phi_g": phi})
    return pd.DataFrame(rows)


def bin_curve(df, bin_width=10):
    """Bin tau into bins of width `bin_width`, return median ± IQR per bin."""
    df = df.copy()
    df["tau_bin"] = (df["tau"] / bin_width).round() * bin_width
    agg = df.groupby("tau_bin")["phi_g"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        n="count",
    ).reset_index()
    return agg


def main(window, stride, n_seeds, bin_width):
    seeds = list(range(1, n_seeds + 1))

    print(f"\n{'='*78}")
    print(f"Hysteresis: switch-aligned sliding-window Φ_r(g)")
    print(f"  window={window}, stride={stride}, seeds={n_seeds}, switch={T_SWITCH}")
    print(f"{'='*78}\n")

    print("Computing sliding-window Φ_r (trained)...")
    tr = collect_hysteresis("replay", seeds, window, stride)
    print("Computing sliding-window Φ_r (untrained)...")
    un = collect_hysteresis("replay_untrained", seeds, window, stride)

    if len(tr) == 0:
        print("No data found.")
        return

    # --- window stability check (pure-pre region: tau well below 0) ---
    print(f"\n{'='*78}")
    print("WINDOW STABILITY CHECK (pure-pre region, tau < -60)")
    print(f"{'='*78}")
    for label, d in [("trained", tr), ("untrained", un)]:
        pre_region = d[d.tau < -60]
        if len(pre_region) > 0:
            cv = pre_region["phi_g"].std() / max(abs(pre_region["phi_g"].mean()), 1e-6)
            print(f"  [{label}] pre-region Φ_r(g): mean={pre_region['phi_g'].mean():.3f}, "
                  f"std={pre_region['phi_g'].std():.3f}, CV={cv:.2f}")
    print(f"  (CV < ~0.5 suggests window={window} is large enough for stable estimates)")

    # --- binned curves ---
    tr_curve = bin_curve(tr, bin_width)
    un_curve = bin_curve(un, bin_width)

    # --- quantify adaptation speed ---
    print(f"\n{'='*78}")
    print("ADAPTATION SPEED (how fast Φ_r(g) settles after switch)")
    print(f"{'='*78}")
    for label, curve in [("trained", tr_curve), ("untrained", un_curve)]:
        pre = curve[curve.tau_bin < 0]["median"]
        post_early = curve[(curve.tau_bin >= 0) & (curve.tau_bin <= 20)]["median"]
        post_late = curve[curve.tau_bin > 40]["median"]
        if len(pre) and len(post_late):
            pre_lvl = pre.mean()
            late_lvl = post_late.mean()
            early_lvl = post_early.mean() if len(post_early) else np.nan
            total_change = late_lvl - pre_lvl
            early_change = early_lvl - pre_lvl if not np.isnan(early_lvl) else np.nan
            frac_early = (early_change / total_change) if abs(total_change) > 1e-6 else np.nan
            print(f"\n  [{label}]")
            print(f"     pre-switch level:        {pre_lvl:+.4f}")
            print(f"     early post (tau 0-20):   {early_lvl:+.4f}")
            print(f"     late post (tau>40):      {late_lvl:+.4f}")
            print(f"     total Δ (late-pre):      {total_change:+.4f}")
            print(f"     fraction completed early: {frac_early*100:.0f}%  "
                  f"({'fast/immediate' if frac_early > 0.7 else 'slow/gradual'})")

    print(f"\n  Interpretation:")
    print(f"  - High 'fraction completed early' = immediate response (no hysteresis)")
    print(f"  - Low fraction = slow gradual adaptation (hysteresis present)")

    # --- print compact curve table ---
    print(f"\n{'='*78}")
    print("BINNED CURVES (median Φ_r(g) per tau bin)")
    print(f"{'='*78}")
    print(f"\n  {'tau':>6} | {'trained':>20} | {'untrained':>20}")
    print(f"  {'':>6} | {'med [q25,q75]':>20} | {'med [q25,q75]':>20}")
    print("  " + "-"*52)
    all_taus = sorted(set(tr_curve.tau_bin) | set(un_curve.tau_bin))
    for tau in all_taus:
        tr_row = tr_curve[tr_curve.tau_bin == tau]
        un_row = un_curve[un_curve.tau_bin == tau]
        tr_s = (f"{tr_row['median'].iloc[0]:.2f} [{tr_row['q25'].iloc[0]:.2f},{tr_row['q75'].iloc[0]:.2f}]"
                if len(tr_row) else "—")
        un_s = (f"{un_row['median'].iloc[0]:.2f} [{un_row['q25'].iloc[0]:.2f},{un_row['q75'].iloc[0]:.2f}]"
                if len(un_row) else "—")
        marker = " ←switch" if tau == 0 else ""
        print(f"  {tau:>6.0f} | {tr_s:>20} | {un_s:>20}{marker}")

    # --- save ---
    tr["group"] = "trained"
    un["group"] = "untrained"
    combined = pd.concat([tr, un], ignore_index=True)
    combined.to_csv(OUTDIR / "hysteresis_curves.csv", index=False)
    tr_curve["group"] = "trained"
    un_curve["group"] = "untrained"
    pd.concat([tr_curve, un_curve]).to_csv(OUTDIR / "hysteresis_binned.csv", index=False)
    print(f"\n  Saved → outputs/hysteresis_curves.csv (raw)")
    print(f"  Saved → outputs/hysteresis_binned.csv (binned for plotting)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=80)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--bin_width", type=int, default=10)
    args = ap.parse_args()
    main(args.window, args.stride, args.n_seeds, args.bin_width)
