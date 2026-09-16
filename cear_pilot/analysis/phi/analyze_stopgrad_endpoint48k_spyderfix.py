# cear_pilot/analysis/phi/analyze_stopgrad_endpoint48k.py
# -*- coding: utf-8 -*-
"""
48k endpoint analysis for the journal stop-gradient experiment.

Spyder use
----------
Save this file as:
    cear_pilot/analysis/phi/analyze_stopgrad_endpoint48k.py
and press Run. No command-line arguments are required.

This script intentionally ignores longitudinal training-step trajectories and
analyzes only the pre-specified final endpoint (48,000 steps).

Inputs expected from the previous matched-replay analysis
----------------------------------------------------------
outputs/journal_stopgrad_analysis/
    journal_phi_episode.csv
    journal_phi_seed.csv

Raw matched replay trajectories are also used for an optional shared-partition
robustness analysis:
outputs/journal_replay_matched/{default,coupled}/step48000/seed*/serpentine_dwell4/traj.parquet

Main endpoint questions
-----------------------
1. Paired 48k coupled - default effects for:
   - Phi_r(g)
   - decoupling contribution
   - whole->part/downward contribution
   - part-driven contribution
   - Phi_r(z) control

2. What drives the downward contribution?
   Because Fiedler partition orientation is arbitrary, PART0 and PART1 should
   not be interpreted separately across episodes. We therefore report two
   orientation-invariant components:
       whole_to_parts
       whole_to_single_sum = whole_to_part0 + whole_to_part1

3. Is the decrease in decoupling coupled seed-wise to an increase in
   whole->part contribution?
   - Spearman rho across the 30 paired seed effects
   - seed-bootstrap 95% CI for rho

4. Could the result be an artifact of default/coupled choosing different
   Fiedler partitions?
   - diagnostic partition agreement at 48k
   - optional shared-pair-partition robustness analysis. For each matched
     episode pair, a single symmetric partition is derived from the mean of
     the default and coupled lag-1 MI matrices, then BOTH conditions are
     evaluated under that same partition.

Statistics
----------
Experimental unit = training seed (n=30).
Primary summary = median paired effect + percentile seed-bootstrap 95% CI.
Wilcoxon signed-rank p-values are secondary/descriptive.

Outputs
-------
outputs/journal_stopgrad_endpoint48k/
    endpoint_primary_seed_effects.csv
    endpoint_primary_summary.csv
    endpoint_downward_decomposition_seed_effects.csv
    endpoint_downward_decomposition_summary.csv
    endpoint_reorganization_correlation.csv
    endpoint_partition_episode_diagnostics.csv
    endpoint_partition_seed_diagnostics.csv
    endpoint_partition_summary.csv
    endpoint_shared_partition_episode.csv
    endpoint_shared_partition_seed.csv
    endpoint_shared_partition_summary.csv
    endpoint_shared_vs_free_summary.csv
    endpoint_manifest.json
    figures/*.png, *.pdf
"""

from __future__ import annotations

# =====================================================================
# CONFIG
# =====================================================================

FINAL_STEP = 48000
SEEDS = tuple(range(1, 31))
N_EPISODES = 10
N_STEPS = 800

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260915
ANALYSIS_RANDOM_SEED = 884211

# Shared-partition robustness is endpoint-only and CPU-feasible.
COMPUTE_SHARED_PARTITION_ROBUSTNESS = True
FORCE_SHARED_RECOMPUTE = False
SHARED_CACHE_EVERY_SEEDS = 2

MAKE_FIGURES = True
SHOW_FIGURES = True
STRICT = True

# =====================================================================

import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon


# ---------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

if not (REPO_ROOT / "cear_pilot").exists():
    cwd = Path.cwd().resolve()
    if (cwd / "cear_pilot").exists():
        REPO_ROOT = cwd
    else:
        raise RuntimeError(
            "Could not locate repository root. Save this script under "
            "cear_pilot/analysis/phi/ inside perspective-ce."
        )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Spyder's User Module Reloader (UMR) can occasionally leave a package tree
# in an inconsistent state: a submodule such as cear_pilot.analysis.phi may
# remain in sys.modules after its parent cear_pilot has been removed.  The
# next package import then fails inside importlib with KeyError: 'cear_pilot'.
# Purge the CEAR package tree and import it cleanly.  This affects only the
# current Python process; it does not touch files or analysis results.
import importlib

_stale_cear_modules = [
    name for name in list(sys.modules)
    if name == "cear_pilot" or name.startswith("cear_pilot.")
]
for _name in sorted(
    _stale_cear_modules,
    key=lambda s: s.count("."),
    reverse=True,
):
    sys.modules.pop(_name, None)

importlib.invalidate_caches()

# Import the parent package explicitly before its subpackages.  This is more
# robust under Spyder's runfile/UMR workflow than importing a deep submodule
# directly after a reload.
importlib.import_module("cear_pilot")
importlib.import_module("cear_pilot.analysis")
importlib.import_module("cear_pilot.analysis.phi")

info = importlib.import_module("cear_pilot.analysis.phi.information")
phi_atoms = importlib.import_module("cear_pilot.analysis.phi.phi_atoms")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PREV_ROOT = REPO_ROOT / "outputs" / "journal_stopgrad_analysis"
EPISODE_CSV = PREV_ROOT / "journal_phi_episode.csv"
SEED_CSV = PREV_ROOT / "journal_phi_seed.csv"

RAW_ROOT = REPO_ROOT / "outputs" / "journal_replay_matched"

OUT_ROOT = REPO_ROOT / "outputs" / "journal_stopgrad_endpoint48k"
FIG_ROOT = OUT_ROOT / "figures"

PRIMARY_SEED_CSV = OUT_ROOT / "endpoint_primary_seed_effects.csv"
PRIMARY_SUMMARY_CSV = OUT_ROOT / "endpoint_primary_summary.csv"

DOWN_SEED_CSV = OUT_ROOT / "endpoint_downward_decomposition_seed_effects.csv"
DOWN_SUMMARY_CSV = OUT_ROOT / "endpoint_downward_decomposition_summary.csv"

CORR_CSV = OUT_ROOT / "endpoint_reorganization_correlation.csv"

PART_EP_CSV = OUT_ROOT / "endpoint_partition_episode_diagnostics.csv"
PART_SEED_CSV = OUT_ROOT / "endpoint_partition_seed_diagnostics.csv"
PART_SUMMARY_CSV = OUT_ROOT / "endpoint_partition_summary.csv"

SHARED_EP_CSV = OUT_ROOT / "endpoint_shared_partition_episode.csv"
SHARED_SEED_CSV = OUT_ROOT / "endpoint_shared_partition_seed.csv"
SHARED_SUMMARY_CSV = OUT_ROOT / "endpoint_shared_partition_summary.csv"
SHARED_COMPARE_CSV = OUT_ROOT / "endpoint_shared_vs_free_summary.csv"

MANIFEST_JSON = OUT_ROOT / "endpoint_manifest.json"


PRIMARY_METRICS = (
    "phi_r_g",
    "group_decoupling_g",
    "group_downward_g",
    "group_part_driven_g",
)

PRIMARY_LABELS = {
    "phi_r_g": r"$\Phi_r(g)$",
    "group_decoupling_g": "Decoupling",
    "group_downward_g": "Whole-to-part contribution",
    "group_part_driven_g": "Part-driven contribution",
    "phi_r_z": r"$\Phi_r(z)$ control",
}


# =====================================================================
# Generic stats helpers
# =====================================================================

def stable_seed(*parts) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return zlib.crc32(text) & 0xFFFFFFFF


def bootstrap_median_ci(values, tag):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if len(x) == 1:
        return med, med, med

    rng = np.random.default_rng((BOOTSTRAP_SEED + stable_seed(tag)) & 0xFFFFFFFF)
    idx = rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))
    boot = np.median(x[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return med, float(lo), float(hi)


def safe_wilcoxon(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0.0, atol=1e-15, rtol=0.0):
        return 1.0
    try:
        return float(
            wilcoxon(
                x,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return np.nan


def bootstrap_spearman_ci(x, y, tag):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    rho, p = spearmanr(x, y)
    rng = np.random.default_rng((BOOTSTRAP_SEED + stable_seed(tag)) & 0xFFFFFFFF)

    vals = []
    for _ in range(N_BOOTSTRAP):
        ii = rng.integers(0, len(x), size=len(x))
        xb, yb = x[ii], y[ii]
        # Degenerate bootstrap samples can have zero variance.
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        rb = spearmanr(xb, yb).statistic
        if np.isfinite(rb):
            vals.append(float(rb))

    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(rho), float(lo), float(hi), float(p), len(vals)


def paired_effect_table(seed_df, metrics):
    """48k coupled-default paired effects at the seed level."""
    d = seed_df[seed_df["condition"] == "default"].set_index("seed")
    c = seed_df[seed_df["condition"] == "coupled"].set_index("seed")

    seed_rows = []
    summary_rows = []

    for metric in metrics:
        dv = d.loc[list(SEEDS), metric].to_numpy(dtype=float)
        cv = c.loc[list(SEEDS), metric].to_numpy(dtype=float)
        effect = cv - dv

        for i, seed in enumerate(SEEDS):
            seed_rows.append({
                "seed": seed,
                "metric": metric,
                "default": float(dv[i]),
                "coupled": float(cv[i]),
                "effect_coupled_minus_default": float(effect[i]),
            })

        med, lo, hi = bootstrap_median_ci(effect, f"paired|{metric}")
        summary_rows.append({
            "metric": metric,
            "n_seeds": len(SEEDS),
            "median_effect": med,
            "ci95_low": lo,
            "ci95_high": hi,
            "wilcoxon_p": safe_wilcoxon(effect),
            "n_positive": int(np.sum(effect > 0)),
            "n_negative": int(np.sum(effect < 0)),
            "n_zero": int(np.sum(effect == 0)),
        })

    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


# =====================================================================
# Load and validate previous analysis tables
# =====================================================================

def load_endpoint_tables():
    if not EPISODE_CSV.exists() or not SEED_CSV.exists():
        raise FileNotFoundError(
            "Missing previous analysis tables. Run analyze_stopgrad_matched.py first.\n"
            f"Expected:\n  {EPISODE_CSV}\n  {SEED_CSV}"
        )

    ep = pd.read_csv(EPISODE_CSV)
    seed = pd.read_csv(SEED_CSV)

    ep = ep[ep["checkpoint"] == FINAL_STEP].copy()
    seed = seed[seed["checkpoint"] == FINAL_STEP].copy()

    expected_ep = 2 * len(SEEDS) * N_EPISODES
    expected_seed = 2 * len(SEEDS)

    if STRICT:
        if len(ep) != expected_ep:
            raise RuntimeError(f"48k episode rows: got {len(ep)}, expected {expected_ep}")
        if len(seed) != expected_seed:
            raise RuntimeError(f"48k seed rows: got {len(seed)}, expected {expected_seed}")

    for cond in ("default", "coupled"):
        got = sorted(seed.loc[seed.condition == cond, "seed"].astype(int).tolist())
        if STRICT and got != list(SEEDS):
            raise RuntimeError(f"{cond}: incomplete 48k seeds: {got}")

    return ep, seed


# =====================================================================
# Primary endpoint
# =====================================================================

def primary_endpoint(seed_df):
    metrics = list(PRIMARY_METRICS)
    if "phi_r_z" in seed_df.columns:
        metrics.append("phi_r_z")

    seed_effects, summary = paired_effect_table(seed_df, metrics)
    seed_effects.to_csv(PRIMARY_SEED_CSV, index=False)
    summary.to_csv(PRIMARY_SUMMARY_CSV, index=False)
    return seed_effects, summary


# =====================================================================
# Downward decomposition, orientation-invariant
# =====================================================================

def downward_decomposition(episode_df):
    p0 = "atom_whole_to_part0_g"
    ps = "atom_whole_to_parts_g"
    p1 = "atom_whole_to_part1_g"

    for col in (p0, ps, p1):
        if col not in episode_df.columns:
            raise KeyError(f"Missing atom column: {col}")

    x = episode_df[["condition", "seed", "episode", p0, ps, p1, "group_downward_g"]].copy()

    # PART0/PART1 identities can flip with Fiedler sign. Their sum is invariant.
    x["whole_to_single_sum"] = x[p0] + x[p1]
    x["whole_to_parts"] = x[ps]
    x["downward_reconstructed"] = x["whole_to_single_sum"] + x["whole_to_parts"]
    x["reconstruction_error"] = x["downward_reconstructed"] - x["group_downward_g"]

    max_err = float(np.max(np.abs(x["reconstruction_error"])))
    if STRICT and max_err > 1e-10:
        raise RuntimeError(f"Downward reconstruction mismatch: max error={max_err}")

    seed_comp = (
        x.groupby(["condition", "seed"], as_index=False)[
            ["whole_to_single_sum", "whole_to_parts", "downward_reconstructed"]
        ]
        .median()
    )

    metrics = ["whole_to_single_sum", "whole_to_parts", "downward_reconstructed"]
    seed_effects, summary = paired_effect_table(seed_comp, metrics)
    seed_effects.to_csv(DOWN_SEED_CSV, index=False)
    summary.to_csv(DOWN_SUMMARY_CSV, index=False)

    return seed_effects, summary, max_err


# =====================================================================
# Reorganization relationship
# =====================================================================

def reorganization_correlation(primary_seed_effects):
    wide = primary_seed_effects.pivot(
        index="seed",
        columns="metric",
        values="effect_coupled_minus_default",
    )

    x = wide["group_decoupling_g"].to_numpy(dtype=float)
    y = wide["group_downward_g"].to_numpy(dtype=float)

    rho, lo, hi, p, nboot = bootstrap_spearman_ci(
        x, y, "decoupling-vs-downward"
    )

    out = pd.DataFrame([{
        "x_metric": "group_decoupling_g",
        "y_metric": "group_downward_g",
        "spearman_rho": rho,
        "ci95_low": lo,
        "ci95_high": hi,
        "spearman_p": p,
        "n_seeds": len(SEEDS),
        "valid_bootstrap_samples": nboot,
    }])
    out.to_csv(CORR_CSV, index=False)
    return wide.reset_index(), out


# =====================================================================
# Partition diagnostics from the original free-Fiedler analysis
# =====================================================================

def parse_partition_signature(sig):
    m = re.fullmatch(r"A\[(.*?)\]\|B\[(.*?)\]", str(sig))
    if not m:
        raise ValueError(f"Cannot parse partition signature: {sig}")

    def parse_side(s):
        s = s.strip()
        if not s:
            return set()
        return {int(v) for v in s.split(",")}

    return parse_side(m.group(1)), parse_side(m.group(2))


def partition_distance(sig_a, sig_b, d=12):
    """
    Label-invariant partition disagreement.
    Returns mismatch fraction in [0, 0.5] and agreement = 1-mismatch.
    """
    a1, a2 = parse_partition_signature(sig_a)
    b1, b2 = parse_partition_signature(sig_b)

    if (a1 | a2) != set(range(d)) or (b1 | b2) != set(range(d)):
        raise ValueError("Partition does not cover expected latent dimensions")

    direct = len(a1.symmetric_difference(b1))
    flipped = len(a1.symmetric_difference(b2))
    mismatches = min(direct, flipped)
    mismatch_fraction = mismatches / d
    return mismatch_fraction, 1.0 - mismatch_fraction, mismatches == 0


def partition_diagnostics(episode_df):
    keys = ["seed", "episode"]
    d = episode_df[episode_df.condition == "default"][
        keys + ["g_partition_signature", "g_partition_n1", "g_partition_n2"]
    ].rename(columns={
        "g_partition_signature": "signature_default",
        "g_partition_n1": "n1_default",
        "g_partition_n2": "n2_default",
    })
    c = episode_df[episode_df.condition == "coupled"][
        keys + ["g_partition_signature", "g_partition_n1", "g_partition_n2"]
    ].rename(columns={
        "g_partition_signature": "signature_coupled",
        "g_partition_n1": "n1_coupled",
        "g_partition_n2": "n2_coupled",
    })

    m = d.merge(c, on=keys, how="inner")
    rows = []
    for r in m.itertuples(index=False):
        mismatch, agreement, exact = partition_distance(
            r.signature_default, r.signature_coupled, d=12
        )
        rows.append({
            "seed": int(r.seed),
            "episode": int(r.episode),
            "signature_default": r.signature_default,
            "signature_coupled": r.signature_coupled,
            "partition_mismatch_fraction": mismatch,
            "partition_agreement": agreement,
            "exact_partition_match": bool(exact),
        })

    ep_diag = pd.DataFrame(rows)
    ep_diag.to_csv(PART_EP_CSV, index=False)

    seed_diag = (
        ep_diag.groupby("seed", as_index=False)
        .agg(
            exact_match_fraction=("exact_partition_match", "mean"),
            median_partition_agreement=("partition_agreement", "median"),
            mean_partition_agreement=("partition_agreement", "mean"),
        )
    )
    seed_diag.to_csv(PART_SEED_CSV, index=False)

    em, el, eh = bootstrap_median_ci(seed_diag["exact_match_fraction"], "partition-exact")
    am, al, ah = bootstrap_median_ci(seed_diag["median_partition_agreement"], "partition-agree")

    summary = pd.DataFrame([
        {
            "metric": "seed_exact_partition_match_fraction",
            "median": em,
            "ci95_low": el,
            "ci95_high": eh,
            "overall_episode_fraction": float(ep_diag["exact_partition_match"].mean()),
        },
        {
            "metric": "seed_median_partition_agreement",
            "median": am,
            "ci95_low": al,
            "ci95_high": ah,
            "overall_episode_fraction": np.nan,
        },
    ])
    summary.to_csv(PART_SUMMARY_CSV, index=False)

    return ep_diag, seed_diag, summary


# =====================================================================
# Shared-partition robustness
# =====================================================================

def raw_path(condition, seed):
    return (
        RAW_ROOT / condition / f"step{FINAL_STEP:05d}" / f"seed{seed}"
        / "serpentine_dwell4" / "traj.parquet"
    )


def latent_columns(df, prefix):
    cc = [c for c in df.columns if c.startswith(prefix + "_")]
    return sorted(cc, key=lambda s: int(s.rsplit("_", 1)[1]))


def zscore_for_pair_partition(X, rng_seed):
    np.random.seed(int(rng_seed))
    random.seed(int(rng_seed))
    return info.corrected_zscore(np.asarray(X, float).T.copy(), axis=1)


def shared_partition(Gd, Gc, rng_seed):
    """
    Symmetric pairwise Fiedler partition from the mean of separately estimated
    default/coupled lag-1 MI matrices. No artificial concatenation boundary.
    """
    xd = zscore_for_pair_partition(Gd, rng_seed)
    xc = zscore_for_pair_partition(Gc, rng_seed)

    mi_d = info.mutual_information_matrix_fast(
        xd, alpha=0.05, lag=1, bonferonni=True
    )
    mi_c = info.mutual_information_matrix_fast(
        xc, alpha=0.05, lag=1, bonferonni=True
    )
    mi_shared = 0.5 * (mi_d + mi_c)

    np.random.seed(int(rng_seed))
    random.seed(int(rng_seed))
    a, b = info.minimum_information_bipartition(mi_shared, noise=True)

    if len(a) == 0 or len(b) == 0:
        d = Gd.shape[1]
        a = list(range(d // 2))
        b = list(range(d // 2, d))

    return list(a), list(b)


def decompose_forced_partition(G, a, b, rng_seed):
    np.random.seed(int(rng_seed))
    random.seed(int(rng_seed))
    x = info.corrected_zscore(np.asarray(G, float).T.copy(), axis=1)

    x2 = np.vstack([
        x[a].mean(axis=0, keepdims=True),
        x[b].mean(axis=0, keepdims=True),
    ])
    lattice = info.local_phi_id(0, 1, x2)

    series = {
        atom: np.asarray(lattice.nodes[atom]["pi"], float).copy()
        for atom in phi_atoms.ALL_ATOMS
    }
    phi_local = np.sum(
        np.stack([series[atom] for atom in phi_atoms.ALL_ATOMS]), axis=0
    )

    return {
        "phi_r_g": float(np.median(phi_local)),
        "group_decoupling_g": float(sum(
            np.median(series[a0]) for a0 in phi_atoms.DECOUPLING_ATOMS
        )),
        "group_downward_g": float(sum(
            np.median(series[a0]) for a0 in phi_atoms.DOWNWARD_ATOMS
        )),
        "group_part_driven_g": float(sum(
            np.median(series[a0]) for a0 in phi_atoms.PART_DRIVEN_ATOMS
        )),
    }


def shared_partition_robustness():
    if not COMPUTE_SHARED_PARTITION_ROBUSTNESS:
        return None, None, None

    if SHARED_EP_CSV.exists() and not FORCE_SHARED_RECOMPUTE:
        cache = pd.read_csv(SHARED_EP_CSV)
        print(f"Loaded shared-partition cache: {len(cache)} rows")
    else:
        cache = pd.DataFrame()

    def seed_complete(seed):
        if cache.empty:
            return False
        s = cache[cache.seed == seed]
        return len(s) == 2 * N_EPISODES and set(s.condition) == {"default", "coupled"}

    done_since_save = 0

    for seed in SEEDS:
        if seed_complete(seed) and not FORCE_SHARED_RECOMPUTE:
            print(f"[shared] cache seed={seed:02d}")
            continue

        print(f"[shared] calc  seed={seed:02d}")
        if not cache.empty:
            cache = cache[cache.seed != seed].copy()

        pd_path = raw_path("default", seed)
        pc_path = raw_path("coupled", seed)
        if not pd_path.exists() or not pc_path.exists():
            raise FileNotFoundError(f"Missing raw 48k trajectory for seed {seed}")

        dd = pd.read_parquet(pd_path)
        cc = pd.read_parquet(pc_path)
        gd_cols = latent_columns(dd, "g")
        gc_cols = latent_columns(cc, "g")
        if gd_cols != gc_cols:
            raise RuntimeError(f"g columns differ for seed {seed}")

        rows = []
        for ep in range(N_EPISODES):
            de = dd[dd.episode == ep].sort_values("t").reset_index(drop=True)
            ce = cc[cc.episode == ep].sort_values("t").reset_index(drop=True)
            if len(de) != N_STEPS or len(ce) != N_STEPS:
                raise RuntimeError(f"seed {seed} ep {ep}: wrong row count")

            Gd = de[gd_cols].to_numpy(float)
            Gc = ce[gc_cols].to_numpy(float)

            rng_seed = (
                ANALYSIS_RANDOM_SEED + stable_seed("shared", seed, ep)
            ) & 0xFFFFFFFF
            a, b = shared_partition(Gd, Gc, rng_seed)

            md = decompose_forced_partition(Gd, a, b, rng_seed)
            mc = decompose_forced_partition(Gc, a, b, rng_seed)

            sig = (
                "A[" + ",".join(map(str, sorted(a))) + "]|B["
                + ",".join(map(str, sorted(b))) + "]"
            )

            for cond, vals in (("default", md), ("coupled", mc)):
                rows.append({
                    "seed": seed,
                    "episode": ep,
                    "condition": cond,
                    "shared_partition_signature": sig,
                    "shared_partition_n1": len(a),
                    "shared_partition_n2": len(b),
                    **vals,
                })

        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        done_since_save += 1
        if done_since_save >= SHARED_CACHE_EVERY_SEEDS:
            cache.sort_values(["seed", "episode", "condition"]).to_csv(
                SHARED_EP_CSV, index=False
            )
            done_since_save = 0

    cache = cache.sort_values(["seed", "episode", "condition"]).reset_index(drop=True)
    cache.to_csv(SHARED_EP_CSV, index=False)

    seed_df = (
        cache.groupby(["condition", "seed"], as_index=False)[list(PRIMARY_METRICS)]
        .median()
    )
    seed_df.to_csv(SHARED_SEED_CSV, index=False)

    seed_effects, summary = paired_effect_table(seed_df, list(PRIMARY_METRICS))
    # For this file the seed-effect rows are useful, but keep one canonical seed table.
    summary.to_csv(SHARED_SUMMARY_CSV, index=False)

    return cache, seed_effects, summary


# =====================================================================
# Compare free-Fiedler vs shared-partition effects
# =====================================================================

def shared_vs_free(primary_summary, shared_summary):
    if shared_summary is None:
        return None

    free = primary_summary[primary_summary.metric.isin(PRIMARY_METRICS)][
        ["metric", "median_effect", "ci95_low", "ci95_high"]
    ].copy()
    free["partition_mode"] = "free_fiedler"

    shared = shared_summary[
        ["metric", "median_effect", "ci95_low", "ci95_high"]
    ].copy()
    shared["partition_mode"] = "shared_pairwise_fiedler"

    out = pd.concat([free, shared], ignore_index=True)
    out.to_csv(SHARED_COMPARE_CSV, index=False)
    return out


# =====================================================================
# Figures
# =====================================================================

def savefig(fig, name):
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{name}.pdf", bbox_inches="tight")


def plot_endpoint_primary(seed_effects, summary):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    rng = np.random.default_rng(1209)

    for ax, metric in zip(axes, PRIMARY_METRICS):
        vals = seed_effects[seed_effects.metric == metric][
            "effect_coupled_minus_default"
        ].to_numpy(float)
        s = summary[summary.metric == metric].iloc[0]

        jitter = rng.normal(0.0, 0.035, size=len(vals))
        ax.scatter(jitter, vals, alpha=0.75)
        ax.axhline(0.0, linewidth=1, alpha=0.6)
        ax.plot([-0.18, 0.18], [s.median_effect, s.median_effect], linewidth=2.5)
        ax.vlines(0.0, s.ci95_low, s.ci95_high, linewidth=2.0)
        ax.set_xlim(-0.35, 0.35)
        ax.set_xticks([])
        ax.set_title(PRIMARY_LABELS[metric])
        ax.set_ylabel("Coupled - default at 48k")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("48k endpoint: paired stop-gradient effects (n=30 seeds)")
    fig.tight_layout()
    savefig(fig, "fig_A_endpoint_primary")


def plot_downward_decomposition(seed_effects, summary):
    metrics = ["whole_to_single_sum", "whole_to_parts", "downward_reconstructed"]
    labels = {
        "whole_to_single_sum": "Whole to single-part sum",
        "whole_to_parts": "Whole to parts",
        "downward_reconstructed": "Total whole-to-part",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    rng = np.random.default_rng(771)

    for ax, metric in zip(axes, metrics):
        vals = seed_effects[seed_effects.metric == metric][
            "effect_coupled_minus_default"
        ].to_numpy(float)
        s = summary[summary.metric == metric].iloc[0]
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        ax.scatter(jitter, vals, alpha=0.75)
        ax.axhline(0.0, linewidth=1, alpha=0.6)
        ax.plot([-0.18, 0.18], [s.median_effect, s.median_effect], linewidth=2.5)
        ax.vlines(0.0, s.ci95_low, s.ci95_high, linewidth=2.0)
        ax.set_xlim(-0.35, 0.35)
        ax.set_xticks([])
        ax.set_title(labels[metric])
        ax.set_ylabel("Coupled - default at 48k")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("48k endpoint: orientation-invariant decomposition of whole-to-part effect")
    fig.tight_layout()
    savefig(fig, "fig_B_downward_decomposition")


def plot_reorganization(wide, corr):
    x = wide["group_decoupling_g"].to_numpy(float)
    y = wide["group_downward_g"].to_numpy(float)
    r = corr.iloc[0]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, alpha=0.8)
    ax.axhline(0.0, linewidth=1, alpha=0.5)
    ax.axvline(0.0, linewidth=1, alpha=0.5)
    ax.set_xlabel("Decoupling effect: coupled - default")
    ax.set_ylabel("Whole-to-part effect: coupled - default")
    ax.set_title(
        "Seed-wise reorganization at 48k\n"
        f"Spearman rho={r.spearman_rho:.3f} "
        f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}]"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "fig_C_decoupling_vs_downward")


def plot_shared_robustness(compare_df):
    if compare_df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    modes = ["free_fiedler", "shared_pairwise_fiedler"]
    x = np.arange(len(modes), dtype=float)

    for ax, metric in zip(axes, PRIMARY_METRICS):
        s = compare_df[compare_df.metric == metric].set_index("partition_mode").loc[modes]
        y = s["median_effect"].to_numpy(float)
        lo = s["ci95_low"].to_numpy(float)
        hi = s["ci95_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
        ax.axhline(0.0, linewidth=1, alpha=0.6)
        ax.set_xticks(x, ["Free Fiedler", "Shared partition"])
        ax.set_title(PRIMARY_LABELS[metric])
        ax.set_ylabel("Median paired effect at 48k")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Partition-robustness check at the 48k endpoint")
    fig.tight_layout()
    savefig(fig, "fig_D_shared_partition_robustness")


def plot_partition_diagnostics(seed_diag, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(seed_diag.seed, seed_diag.exact_match_fraction, alpha=0.8)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Exact partition match fraction")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(seed_diag.seed, seed_diag.median_partition_agreement, alpha=0.8)
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Median partition agreement")
    axes[1].set_ylim(0.45, 1.05)
    axes[1].grid(alpha=0.2)

    fig.suptitle("48k default-coupled Fiedler partition diagnostics")
    fig.tight_layout()
    savefig(fig, "fig_S_partition_diagnostics")


def plot_z_control(primary_seed_effects, primary_summary):
    if "phi_r_z" not in primary_summary.metric.values:
        return

    vals = primary_seed_effects[primary_seed_effects.metric == "phi_r_z"][
        "effect_coupled_minus_default"
    ].to_numpy(float)
    s = primary_summary[primary_summary.metric == "phi_r_z"].iloc[0]
    rng = np.random.default_rng(99)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    jitter = rng.normal(0.0, 0.035, size=len(vals))
    ax.scatter(jitter, vals, alpha=0.8)
    ax.axhline(0.0, linewidth=1, alpha=0.6)
    ax.plot([-0.18, 0.18], [s.median_effect, s.median_effect], linewidth=2.5)
    ax.vlines(0.0, s.ci95_low, s.ci95_high, linewidth=2.0)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([])
    ax.set_ylabel("Coupled - default at 48k")
    ax.set_title(r"Specificity control: $\Phi_r(z)$")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    savefig(fig, "fig_S_phi_r_z_control")


# =====================================================================
# Reporting
# =====================================================================

def print_summary(primary_summary, down_summary, corr, part_summary, shared_summary):
    print("\n" + "=" * 82)
    print("48k PRIMARY ENDPOINT: coupled - default")
    print("=" * 82)

    for metric in list(PRIMARY_METRICS) + (["phi_r_z"] if "phi_r_z" in primary_summary.metric.values else []):
        r = primary_summary[primary_summary.metric == metric].iloc[0]
        print(
            f"{metric:30s} "
            f"{r.median_effect:+.6f} "
            f"[{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] "
            f"p={r.wilcoxon_p:.4g} "
            f"(+/-={int(r.n_positive)}/{int(r.n_negative)})"
        )

    print("\n" + "=" * 82)
    print("WHOLE-TO-PART DECOMPOSITION")
    print("=" * 82)
    for metric in ["whole_to_single_sum", "whole_to_parts", "downward_reconstructed"]:
        r = down_summary[down_summary.metric == metric].iloc[0]
        print(
            f"{metric:30s} "
            f"{r.median_effect:+.6f} "
            f"[{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] "
            f"p={r.wilcoxon_p:.4g}"
        )

    r = corr.iloc[0]
    print("\n" + "=" * 82)
    print("SEED-WISE DECOUPLING vs WHOLE-TO-PART")
    print("=" * 82)
    print(
        f"Spearman rho = {r.spearman_rho:+.4f} "
        f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}], p={r.spearman_p:.4g}"
    )

    print("\n" + "=" * 82)
    print("PARTITION DIAGNOSTICS")
    print("=" * 82)
    for rr in part_summary.itertuples(index=False):
        print(
            f"{rr.metric:40s} "
            f"{rr.median:.4f} [{rr.ci95_low:.4f}, {rr.ci95_high:.4f}]"
        )

    if shared_summary is not None:
        print("\n" + "=" * 82)
        print("SHARED-PARTITION ROBUSTNESS: coupled - default")
        print("=" * 82)
        for metric in PRIMARY_METRICS:
            rr = shared_summary[shared_summary.metric == metric].iloc[0]
            print(
                f"{metric:30s} "
                f"{rr.median_effect:+.6f} "
                f"[{rr.ci95_low:+.6f}, {rr.ci95_high:+.6f}] "
                f"p={rr.wilcoxon_p:.4g}"
            )


# =====================================================================
# Main
# =====================================================================

def main():
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("CEAR 48k ENDPOINT STOP-GRADIENT ANALYSIS")
    print("=" * 82)
    print("Repo  :", REPO_ROOT)
    print("Input :", PREV_ROOT)
    print("Output:", OUT_ROOT)
    print()

    episode_df, seed_df = load_endpoint_tables()

    primary_seed, primary_summary = primary_endpoint(seed_df)
    down_seed, down_summary, recon_err = downward_decomposition(episode_df)
    corr_wide, corr = reorganization_correlation(primary_seed)
    part_ep, part_seed, part_summary = partition_diagnostics(episode_df)

    shared_ep = shared_seed_effects = shared_summary = None
    if COMPUTE_SHARED_PARTITION_ROBUSTNESS:
        shared_ep, shared_seed_effects, shared_summary = shared_partition_robustness()

    compare = shared_vs_free(primary_summary, shared_summary)

    if MAKE_FIGURES:
        plot_endpoint_primary(primary_seed, primary_summary)
        plot_downward_decomposition(down_seed, down_summary)
        plot_reorganization(corr_wide, corr)
        plot_partition_diagnostics(part_seed, part_summary)
        plot_z_control(primary_seed, primary_summary)
        plot_shared_robustness(compare)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close("all")

    manifest = {
        "endpoint_step": FINAL_STEP,
        "n_training_seeds": len(SEEDS),
        "episodes_per_seed": N_EPISODES,
        "steps_per_episode": N_STEPS,
        "primary_effect": "paired coupled - default at 48k",
        "bootstrap_unit": "training seed",
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_ci": "percentile 95%",
        "downward_decomposition": {
            "whole_to_single_sum": "whole_to_part0 + whole_to_part1; orientation invariant",
            "whole_to_parts": "whole_to_parts atom",
            "max_reconstruction_error": recon_err,
        },
        "shared_partition_robustness": COMPUTE_SHARED_PARTITION_ROBUSTNESS,
        "shared_partition_definition": (
            "pair-specific Fiedler partition from mean(default MI, coupled MI), "
            "then held fixed across the matched pair"
            if COMPUTE_SHARED_PARTITION_ROBUSTNESS else None
        ),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_summary(
        primary_summary,
        down_summary,
        corr,
        part_summary,
        shared_summary,
    )

    print("\n" + "=" * 82)
    print("ENDPOINT ANALYSIS COMPLETE")
    print("=" * 82)
    print(f"Elapsed: {(time.time() - t0) / 60:.2f} min")
    print("Outputs:", OUT_ROOT)
    print("Figures:", FIG_ROOT)
    print()


if __name__ == "__main__":
    main()
