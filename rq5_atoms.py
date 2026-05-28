#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq_atoms.py
------------
Atom-level Φ_r decomposition across all conditions.

For each episode we compute the 9 ΦID atoms and 3 theory-driven sums
on the g trajectory:
  - decoupling   (WHOLE -> WHOLE)
  - downward     (WHOLE -> PART*)
  - part_driven  (PART* -> WHOLE, PART* -> PART*)

Conditions: trained × untrained × {clean, p20-pre, p20-post}.

PAPER NARRATIVE: only the 3 group sums are reported in the main text.
The 9 individual atom values are saved to CSV for supplementary.

Usage:
  PYTHONPATH=. python rq_atoms.py
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

from cear_pilot.analysis.phi.phi_atoms import atom_decomposition

warnings.filterwarnings("ignore")

OUTDIR = Path("outputs")
WARMUP = 80
T_SWITCH = 320


def get_g_cols(df):
    return sorted([c for c in df.columns if c.startswith("g_")],
                  key=lambda c: int(c.split("_")[1]))


def collect_atoms(group_prefix, condition, seeds, t_lo=None, t_hi=None,
                  pp_phase=None):
    """
    Per-episode atom decomposition.
    If condition == 'p20' and pp_phase is 'pre'/'post', window the episode.
    """
    rows = []
    cond_label = condition if pp_phase is None else f"{condition}_{pp_phase}"
    for seed in seeds:
        path = OUTDIR / f"{group_prefix}_seed{seed}_v2_{condition}" / "traj.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        g_cols = get_g_cols(df)
        for ep_id, ep in df.groupby("episode"):
            if pp_phase == "pre":
                ep_w = ep[(ep.t > WARMUP) & (ep.t <= T_SWITCH)]
            elif pp_phase == "post":
                ep_w = ep[ep.t > T_SWITCH]
            else:
                ep_w = ep[ep.t > WARMUP] if t_lo is None else ep[(ep.t > t_lo) & (ep.t <= (t_hi or ep.t.max()))]
            if len(ep_w) < 30:
                continue
            G = ep_w[g_cols].values
            decomp = atom_decomposition(G)
            rows.append({
                "group": group_prefix.replace("replay_", "").replace("replay", "trained")
                         if group_prefix == "replay" else "untrained",
                "condition": cond_label,
                "seed": seed,
                "episode": ep_id,
                **decomp,
            })
    return pd.DataFrame(rows)


def report_groups(df, condition_label):
    """Print group sums comparison: trained vs untrained for one condition."""
    tr = df[df.group == "trained"]
    un = df[df.group == "untrained"]
    if len(tr) == 0 or len(un) == 0:
        print(f"  [skip] {condition_label}: missing data")
        return None

    print(f"\n  [{condition_label}]  n_tr={len(tr)}, n_un={len(un)}")
    print(f"     {'group':<15} {'untrained':>12} {'trained':>12} {'Δ(learned)':>14} {'Welch p':>10}")
    print(f"     " + "-"*68)
    out = {}
    for grp in ["group_decoupling", "group_downward", "group_part_driven", "phi_r_total"]:
        u = un[grp].values
        t = tr[grp].values
        d = t.mean() - u.mean()
        t_stat, p = ttest_ind(t, u, equal_var=False)
        label = grp.replace("group_", "").replace("_", " ")
        print(f"     {label:<15} {u.mean():>+12.4f} {t.mean():>+12.4f} "
              f"{d:>+14.4f} {p:>10.2e}")
        out[grp] = {"u": u.mean(), "t": t.mean(), "d": d, "p": p}
    return out


def main(n_seeds):
    seeds = list(range(1, n_seeds + 1))

    print(f"\n{'='*78}")
    print(f"ATOM-LEVEL Φ_r decomposition  (n_seeds={n_seeds})")
    print(f"{'='*78}")
    print("\n  3-way grouping (from PID lattice topology):")
    print("    decoupling   = WHOLE -> WHOLE         (pure synergy, 1 atom)")
    print("    downward     = WHOLE -> PART*         (top-down, 3 atoms)")
    print("    part_driven  = PART* -> {WHOLE,PART*} (bottom-up + lateral, 5 atoms)")
    print("\n  Main text reports group sums; individual atoms saved to CSV.")

    print(f"\nLoading and computing atoms (this takes ~5-10 min)...")

    all_dfs = []
    # clean
    print("  trained clean...")
    df = collect_atoms("replay", "clean", seeds)
    all_dfs.append(df)
    print(f"    → {len(df)} episodes")

    print("  untrained clean...")
    df = collect_atoms("replay_untrained", "clean", seeds)
    df["group"] = "untrained"
    all_dfs.append(df)
    print(f"    → {len(df)} episodes")

    # p20 pre/post
    for phase in ["pre", "post"]:
        print(f"  trained p20-{phase}...")
        df = collect_atoms("replay", "p20", seeds, pp_phase=phase)
        all_dfs.append(df)
        print(f"    → {len(df)} episodes")

        print(f"  untrained p20-{phase}...")
        df = collect_atoms("replay_untrained", "p20", seeds, pp_phase=phase)
        df["group"] = "untrained"
        all_dfs.append(df)
        print(f"    → {len(df)} episodes")

    full = pd.concat(all_dfs, ignore_index=True)

    # =====================================================
    # Report: trained vs untrained per condition
    # =====================================================
    print(f"\n{'='*78}")
    print("RESULTS — group sums (paper-facing)")
    print(f"{'='*78}")

    for cond in ["clean", "p20_pre", "p20_post"]:
        sub = full[full.condition == cond]
        report_groups(sub, cond)

    # =====================================================
    # Regime sensitivity per atom group (trained only, then untrained)
    # =====================================================
    print(f"\n{'='*78}")
    print("REGIME SENSITIVITY by group  (p20: post − pre)")
    print(f"{'='*78}")

    for grp_id in ["trained", "untrained"]:
        pre = full[(full.condition == "p20_pre") & (full.group == grp_id)]
        post = full[(full.condition == "p20_post") & (full.group == grp_id)]
        # paired by (seed, episode)
        merged = pre.merge(post, on=["seed", "episode"], suffixes=("_pre", "_post"))
        if len(merged) == 0:
            continue
        print(f"\n  [{grp_id}]  n={len(merged)} paired episodes")
        print(f"     {'group':<15} {'pre':>10} {'post':>10} {'Δ':>10} {'t':>8} {'p':>10}")
        print(f"     " + "-"*58)
        for grp in ["group_decoupling", "group_downward", "group_part_driven"]:
            pre_v = merged[f"{grp}_pre"].values
            post_v = merged[f"{grp}_post"].values
            diff = post_v - pre_v
            t_stat, p = ttest_1samp(diff, 0)
            label = grp.replace("group_", "").replace("_", " ")
            print(f"     {label:<15} {pre_v.mean():>+10.4f} {post_v.mean():>+10.4f} "
                  f"{diff.mean():>+10.4f} {t_stat:>+8.2f} {p:>10.2e}")

    # =====================================================
    # Learning-attributable regime sensitivity by group
    # =====================================================
    print(f"\n{'='*78}")
    print("LEARNING-ATTRIBUTABLE regime sensitivity by group")
    print(f"  ΔΔ = (trained drop) − (untrained drop) per atom group")
    print(f"{'='*78}")

    pre_tr = full[(full.condition == "p20_pre") & (full.group == "trained")]
    post_tr = full[(full.condition == "p20_post") & (full.group == "trained")]
    pre_un = full[(full.condition == "p20_pre") & (full.group == "untrained")]
    post_un = full[(full.condition == "p20_post") & (full.group == "untrained")]

    print(f"\n     {'group':<15} {'tr Δ':>10} {'un Δ':>10} {'ΔΔ':>10} {'Welch p':>10}")
    print(f"     " + "-"*58)
    tr_paired = pre_tr.merge(post_tr, on=["seed", "episode"], suffixes=("_pre", "_post"))
    un_paired = pre_un.merge(post_un, on=["seed", "episode"], suffixes=("_pre", "_post"))
    for grp in ["group_decoupling", "group_downward", "group_part_driven"]:
        tr_d = (tr_paired[f"{grp}_post"] - tr_paired[f"{grp}_pre"]).values
        un_d = (un_paired[f"{grp}_post"] - un_paired[f"{grp}_pre"]).values
        dd = tr_d.mean() - un_d.mean()
        t_stat, p = ttest_ind(tr_d, un_d, equal_var=False)
        label = grp.replace("group_", "").replace("_", " ")
        print(f"     {label:<15} {tr_d.mean():>+10.4f} {un_d.mean():>+10.4f} "
              f"{dd:>+10.4f} {p:>10.2e}")

    # =====================================================
    # Save
    # =====================================================
    out = OUTDIR / "atoms_decomp.csv"
    full.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    print(f"  ({len(full)} rows × {len(full.columns)} cols incl. 9 individual atoms)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=30)
    args = ap.parse_args()
    main(args.n_seeds)
