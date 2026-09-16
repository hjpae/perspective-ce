# -*- coding: utf-8 -*-
"""
CEAR P1 journal rebuild: matched-probe analysis + Figures 2, 3, and 4.

Designed to be run directly in Spyder (no CLI arguments).

What this script does
---------------------
Figure 2: measurement qualification at 48k default
    A. Phi_r(g) vs Phi_r(z)
    B. intact g vs temporally shuffled g
    C. intact g vs independent AR(1)-matched surrogates

Figure 3: learning effect under the same matched serpentine probe
    A. Phi_r(g): default step0 vs default step48k
    B. paired learning effects on PhiID groups
       (decoupling, whole->part, part-driven)
    C. orientation-invariant decomposition of whole->part
       (whole->single-part sum, whole->parts, total)

Figure 4: learning vs policy-side gradient coupling
    Paired seed-level effects shown side-by-side for:
        Phi_r(g), decoupling, whole->part, part-driven

Primary analysis unit
---------------------
    episode metric = temporal median from the full 800-step probe
    seed metric    = median across 10 episodes
    group summary  = median across 30 training seeds
    uncertainty    = percentile bootstrap 95% CI over seeds
    Wilcoxon       = secondary paired test

IMPORTANT
---------
This script intentionally uses the FULL 800-step trajectory. It does NOT apply
an initialization cutoff. The matched probe is defined as organization emerging
from the common reset state g0=0 as sensorimotor history accumulates.

AR(1) surrogate null
--------------------
For each 48k-default episode and each g dimension independently:
    - fit empirical mean, SD, and lag-1 autocorrelation rho;
    - simulate a stationary Gaussian AR(1) with that rho;
    - rescale the simulated series to the empirical mean and SD;
    - dimensions are simulated independently, so cross-dimensional coupling is
      removed while first-order within-dimension persistence is retained.

The AR(1) result therefore asks whether observed Phi_r(g) exceeds what is
expected from independent first-order persistence alone.

Data sources
------------
The script prefers existing validated analysis caches if available:
    outputs/journal_learning_matched/learning_episode.csv
    outputs/journal_stopgrad_analysis/journal_phi_episode.csv

If those are absent/incomplete, it falls back to the raw matched replay files:
    outputs/journal_replay_matched/

Outputs
-------
outputs/journal_figures_2_4/
    unified_episode_metrics.csv
    unified_seed_metrics.csv
    ar1_surrogate_replicates.csv
    ar1_episode_summary.csv
    ar1_seed_summary.csv
    ar1_diagnostics.csv
    fig2_stats.csv
    fig3_stats.csv
    fig4_stats.csv
    figure2_temporal_substrate.png/.pdf
    figure3_learning_reorganization.png/.pdf
    figure4_learning_vs_coupling.png/.pdf

Recommended placement
---------------------
Save this file under:
    cear_pilot/analysis/phi/journal_figures_2_4_matched.py
then press Run in Spyder.
"""

from __future__ import annotations

# ============================================================================
# CONFIG
# ============================================================================

SEEDS = tuple(range(1, 31))
N_EPISODES = 10
N_STEPS = 800
STEP0 = 0
STEP48 = 48000

# AR(1): median across surrogate replicates is taken within each empirical episode.
# 5 is a good default. Increase to 10 only if you want a lower Monte-Carlo-noise pass.
N_AR1_SURROGATES = 5
AR1_BURNIN = 1000
AR1_RHO_CLIP = 0.995
AR1_RANDOM_SEED = 20260917

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260917
SHUFFLE_RANDOM_SEED = 20260917

FORCE_BASE_RECOMPUTE = False
FORCE_AR1_RECOMPUTE = False
STRICT = True
SHOW_FIGURES = True
SAVE_PDF = True
DPI = 240

# ============================================================================
# IMPORTS / REPO PATH
# ============================================================================

import importlib
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

if not (REPO_ROOT / "cear_pilot").exists():
    cwd = Path.cwd().resolve()
    if (cwd / "cear_pilot").exists():
        REPO_ROOT = cwd
    else:
        raise RuntimeError(
            "Could not locate the perspective-ce repo root.\n"
            "Save this script under cear_pilot/analysis/phi/ or run Spyder "
            "with the repository root as the working directory."
        )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Spyder/UMR-safe import of the current repo code.
_stale = [
    name for name in list(sys.modules)
    if name == "cear_pilot" or name.startswith("cear_pilot.")
]
for _name in sorted(_stale, key=lambda s: s.count("."), reverse=True):
    sys.modules.pop(_name, None)

importlib.invalidate_caches()
importlib.import_module("cear_pilot")
importlib.import_module("cear_pilot.analysis")
importlib.import_module("cear_pilot.analysis.phi")

sg = importlib.import_module("cear_pilot.analysis.phi.analyze_stopgrad_matched")
phi_atoms = importlib.import_module("cear_pilot.analysis.phi.phi_atoms")

# ============================================================================
# PATHS
# ============================================================================

RAW_ROOT = REPO_ROOT / "outputs" / "journal_replay_matched"
LEARNING_EP_CACHE = REPO_ROOT / "outputs" / "journal_learning_matched" / "learning_episode.csv"
STOPGRAD_EP_CACHE = REPO_ROOT / "outputs" / "journal_stopgrad_analysis" / "journal_phi_episode.csv"

OUT_ROOT = REPO_ROOT / "outputs" / "journal_figures_2_4"
FIG_ROOT = OUT_ROOT / "figures"

UNIFIED_EP_CSV = OUT_ROOT / "unified_episode_metrics.csv"
UNIFIED_SEED_CSV = OUT_ROOT / "unified_seed_metrics.csv"
AR1_REP_CSV = OUT_ROOT / "ar1_surrogate_replicates.csv"
AR1_EP_CSV = OUT_ROOT / "ar1_episode_summary.csv"
AR1_SEED_CSV = OUT_ROOT / "ar1_seed_summary.csv"
AR1_DIAG_CSV = OUT_ROOT / "ar1_diagnostics.csv"
FIG2_STATS_CSV = OUT_ROOT / "fig2_stats.csv"
FIG3_STATS_CSV = OUT_ROOT / "fig3_stats.csv"
FIG4_STATS_CSV = OUT_ROOT / "fig4_stats.csv"
MANIFEST_JSON = OUT_ROOT / "manifest.json"

# ============================================================================
# LABELS
# ============================================================================

METRICS = {
    "phi_r_g": r"$\Phi_r(g)$",
    "phi_r_z": r"$\Phi_r(z)$",
    "group_decoupling_g": "Decoupling",
    "group_downward_g": "Whole-to-part",
    "group_part_driven_g": "Part-driven",
    "whole_to_single_sum": "Whole to\nsingle-part sum",
    "whole_to_parts": "Whole to parts",
    "downward_reconstructed": "Total\nwhole-to-part",
}

PRIMARY_GROUP_METRICS = (
    "group_decoupling_g",
    "group_downward_g",
    "group_part_driven_g",
)

# ============================================================================
# GENERIC HELPERS
# ============================================================================

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

    rng = np.random.default_rng(
        (BOOTSTRAP_SEED + stable_seed(tag)) & 0xFFFFFFFF
    )
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


def paired_stats(a, b, label, effect_name="A_minus_B"):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays differ in shape: {a.shape} vs {b.shape}")
    effect = a - b
    med, lo, hi = bootstrap_median_ci(effect, label)
    return {
        "contrast": label,
        "effect_definition": effect_name,
        "n_seeds": int(len(effect)),
        "median_A": float(np.median(a)),
        "median_B": float(np.median(b)),
        "median_effect": med,
        "ci95_low": lo,
        "ci95_high": hi,
        "wilcoxon_p": safe_wilcoxon(effect),
        "n_positive": int(np.sum(effect > 0)),
        "n_negative": int(np.sum(effect < 0)),
        "n_zero": int(np.sum(effect == 0)),
    }


def effect_stats(effect, label):
    effect = np.asarray(effect, dtype=np.float64)
    med, lo, hi = bootstrap_median_ci(effect, label)
    return {
        "contrast": label,
        "n_seeds": int(len(effect)),
        "median_effect": med,
        "ci95_low": lo,
        "ci95_high": hi,
        "wilcoxon_p": safe_wilcoxon(effect),
        "n_positive": int(np.sum(effect > 0)),
        "n_negative": int(np.sum(effect < 0)),
        "n_zero": int(np.sum(effect == 0)),
    }


def raw_path(condition: str, checkpoint: int, seed: int) -> Path:
    return (
        RAW_ROOT
        / condition
        / f"step{checkpoint:05d}"
        / f"seed{seed}"
        / "serpentine_dwell4"
        / "traj.parquet"
    )


def temporal_shuffle(X, seed):
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return X[rng.permutation(X.shape[0])]

# ============================================================================
# BASE EPISODE METRICS
# ============================================================================

ATOM_COLS = tuple(
    f"atom_{phi_atoms.ATOM_LABEL[a]}_g"
    for a in phi_atoms.ALL_ATOMS
)

REQUIRED_BASE_COLS = {
    "condition",
    "checkpoint",
    "seed",
    "episode",
    "phi_r_g",
    "phi_r_z",
    "group_decoupling_g",
    "group_downward_g",
    "group_part_driven_g",
    *ATOM_COLS,
}


def _normalize_cached_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed by this paper-facing analysis."""
    keep = [
        c for c in [
            "condition", "checkpoint", "seed", "episode",
            "external_sha256",
            "phi_r_g", "phi_r_z", "phi_r_g_shuffled",
            "group_decoupling_g", "group_downward_g", "group_part_driven_g",
            *ATOM_COLS,
        ] if c in df.columns
    ]
    return df[keep].copy()


def _cache_has_state(df, condition, checkpoint):
    if df is None or df.empty:
        return False
    if not REQUIRED_BASE_COLS.issubset(df.columns):
        return False
    sub = df[(df.condition == condition) & (df.checkpoint == checkpoint)]
    return len(sub) == len(SEEDS) * N_EPISODES


def compute_state_from_raw(condition: str, checkpoint: int) -> pd.DataFrame:
    """Compute full-800-step episode metrics for one condition/checkpoint."""
    rows = []
    total = len(SEEDS)

    for k, seed in enumerate(SEEDS, start=1):
        path = raw_path(condition, checkpoint, seed)
        if not path.exists():
            raise FileNotFoundError(path)

        print(
            f"[base {condition} step={checkpoint:05d}] "
            f"seed {seed:02d} ({k}/{total})"
        )
        df = pd.read_parquet(path)
        g_cols = sg.latent_columns(df, "g")
        z_cols = sg.latent_columns(df, "z")
        if not g_cols or not z_cols:
            raise RuntimeError(f"Missing g/z columns in {path}")

        for ep in range(N_EPISODES):
            e = df[df.episode == ep].sort_values("t").reset_index(drop=True)
            if len(e) != N_STEPS:
                raise RuntimeError(
                    f"{condition} step={checkpoint} seed={seed} ep={ep}: "
                    f"got {len(e)} rows, expected {N_STEPS}"
                )

            G = e[g_cols].to_numpy(np.float64)
            Z = e[z_cols].to_numpy(np.float64)

            gm = sg.decompose_trajectory(
                G,
                sg.episode_analysis_seed(seed, checkpoint, ep, "g"),
                include_atoms=True,
            )
            zm = sg.decompose_trajectory(
                Z,
                sg.episode_analysis_seed(seed, checkpoint, ep, "z"),
                include_atoms=False,
            )

            row = {
                "condition": condition,
                "checkpoint": checkpoint,
                "seed": seed,
                "episode": ep,
                "external_sha256": sg.external_episode_hash(e),
                "phi_r_g": gm["phi_r"],
                "phi_r_z": zm["phi_r"],
                "group_decoupling_g": gm["group_decoupling"],
                "group_downward_g": gm["group_downward"],
                "group_part_driven_g": gm["group_part_driven"],
            }

            for atom in phi_atoms.ALL_ATOMS:
                lab = phi_atoms.ATOM_LABEL[atom]
                row[f"atom_{lab}_g"] = gm[lab]

            if condition == "default" and checkpoint == STEP48:
                shuf_seed = (
                    SHUFFLE_RANDOM_SEED
                    + stable_seed("shuffle", seed, checkpoint, ep)
                ) & 0xFFFFFFFF
                Gs = temporal_shuffle(G, shuf_seed)
                gsm = sg.decompose_trajectory(
                    Gs,
                    sg.episode_analysis_seed(seed, checkpoint, ep, "g_shuffled"),
                    include_atoms=False,
                )
                row["phi_r_g_shuffled"] = gsm["phi_r"]

            rows.append(row)

    return pd.DataFrame(rows)


def load_or_compute_unified_episode_table() -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if UNIFIED_EP_CSV.exists() and not FORCE_BASE_RECOMPUTE:
        df = pd.read_csv(UNIFIED_EP_CSV)
        if (
            _cache_has_state(df, "default", STEP0)
            and _cache_has_state(df, "default", STEP48)
            and _cache_has_state(df, "coupled", STEP48)
            and "phi_r_g_shuffled" in df.columns
        ):
            print(f"Loaded unified episode cache: {UNIFIED_EP_CSV}")
            return df

    learning = None
    stopgrad = None
    if LEARNING_EP_CACHE.exists() and not FORCE_BASE_RECOMPUTE:
        learning = pd.read_csv(LEARNING_EP_CACHE)
        learning = _normalize_cached_rows(learning)
        print(f"Loaded learning cache: {LEARNING_EP_CACHE}")
    if STOPGRAD_EP_CACHE.exists() and not FORCE_BASE_RECOMPUTE:
        stopgrad = pd.read_csv(STOPGRAD_EP_CACHE)
        stopgrad = _normalize_cached_rows(stopgrad)
        print(f"Loaded stop-gradient cache: {STOPGRAD_EP_CACHE}")

    chunks = []

    # default step0
    if _cache_has_state(learning, "default", STEP0):
        chunks.append(learning[(learning.condition == "default") & (learning.checkpoint == STEP0)].copy())
    else:
        chunks.append(compute_state_from_raw("default", STEP0))

    # default step48: prefer learning cache because it already contains shuffle.
    if (
        _cache_has_state(learning, "default", STEP48)
        and "phi_r_g_shuffled" in learning.columns
        and learning.loc[
            (learning.condition == "default") & (learning.checkpoint == STEP48),
            "phi_r_g_shuffled",
        ].notna().all()
    ):
        chunks.append(learning[(learning.condition == "default") & (learning.checkpoint == STEP48)].copy())
    else:
        chunks.append(compute_state_from_raw("default", STEP48))

    # coupled step48
    if _cache_has_state(stopgrad, "coupled", STEP48):
        chunks.append(stopgrad[(stopgrad.condition == "coupled") & (stopgrad.checkpoint == STEP48)].copy())
    else:
        chunks.append(compute_state_from_raw("coupled", STEP48))

    df = pd.concat(chunks, ignore_index=True, sort=False)
    df = df.sort_values(["condition", "checkpoint", "seed", "episode"]).reset_index(drop=True)

    # Add orientation-invariant whole->part components at the EPISODE level.
    p0 = "atom_whole_to_part0_g"
    ps = "atom_whole_to_parts_g"
    p1 = "atom_whole_to_part1_g"
    for c in (p0, ps, p1):
        if c not in df.columns:
            raise KeyError(f"Missing required atom column: {c}")

    df["whole_to_single_sum"] = df[p0] + df[p1]
    df["whole_to_parts"] = df[ps]
    df["downward_reconstructed"] = df["whole_to_single_sum"] + df["whole_to_parts"]
    df["downward_reconstruction_error"] = (
        df["downward_reconstructed"] - df["group_downward_g"]
    )

    max_err = float(np.nanmax(np.abs(df["downward_reconstruction_error"])))
    if STRICT and max_err > 1e-10:
        raise RuntimeError(f"Whole->part reconstruction mismatch: max error={max_err}")

    # Completeness.
    expected = 3 * len(SEEDS) * N_EPISODES
    if STRICT and len(df) != expected:
        raise RuntimeError(f"Unified episode rows: got {len(df)}, expected {expected}")

    # Three-way matched external-history validation if hashes are available.
    if "external_sha256" in df.columns and df["external_sha256"].notna().all():
        piv = df.pivot_table(
            index=["seed", "episode"],
            columns=["condition", "checkpoint"],
            values="external_sha256",
            aggfunc="first",
        )
        cols = [("default", STEP0), ("default", STEP48), ("coupled", STEP48)]
        if all(c in piv.columns for c in cols):
            ok = (
                (piv[cols[0]] == piv[cols[1]])
                & (piv[cols[0]] == piv[cols[2]])
            )
            print(
                f"Matched external histories: {int(ok.sum())}/{len(ok)} episode triples identical"
            )
            if STRICT and not bool(ok.all()):
                bad = ok[~ok]
                raise RuntimeError(
                    "External-history mismatch across step0/default48/coupled48:\n"
                    + str(bad.head())
                )

    df.to_csv(UNIFIED_EP_CSV, index=False)
    return df


def make_seed_table(ep: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "phi_r_g", "phi_r_z", "phi_r_g_shuffled",
        "group_decoupling_g", "group_downward_g", "group_part_driven_g",
        *ATOM_COLS,
        "whole_to_single_sum", "whole_to_parts", "downward_reconstructed",
    ]
    metric_cols = [c for c in metric_cols if c in ep.columns]

    seed_df = (
        ep.groupby(["condition", "checkpoint", "seed"], as_index=False)[metric_cols]
        .median()
        .sort_values(["condition", "checkpoint", "seed"])
        .reset_index(drop=True)
    )
    seed_df.to_csv(UNIFIED_SEED_CSV, index=False)
    return seed_df

# ============================================================================
# AR(1)-MATCHED SURROGATES
# ============================================================================

def lag1_corr(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return float(np.clip(r, -AR1_RHO_CLIP, AR1_RHO_CLIP))


def make_independent_ar1_surrogate(X, rng):
    """
    Independent Gaussian AR(1) per dimension.

    Preserves each dimension's empirical mean and SD exactly after rescaling,
    and targets its empirical lag-1 autocorrelation in expectation.
    """
    X = np.asarray(X, dtype=np.float64)
    T, d = X.shape
    Y = np.empty_like(X)

    empirical_rho = np.empty(d, dtype=float)
    surrogate_rho = np.empty(d, dtype=float)
    empirical_sd = np.std(X, axis=0, ddof=0)
    surrogate_sd = np.empty(d, dtype=float)

    for j in range(d):
        x = X[:, j]
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=0))
        rho = lag1_corr(x)
        empirical_rho[j] = rho

        if sd < 1e-12:
            y = np.full(T, mu, dtype=np.float64)
        else:
            n = T + AR1_BURNIN
            u = np.empty(n, dtype=np.float64)
            u[0] = rng.normal()
            innov_scale = np.sqrt(max(1.0 - rho * rho, 1e-10))
            eps = rng.normal(size=n - 1)
            for t in range(1, n):
                u[t] = rho * u[t - 1] + innov_scale * eps[t - 1]
            u = u[AR1_BURNIN:]

            # Exact empirical mean / SD match; correlation is unchanged by affine rescaling.
            us = float(np.std(u, ddof=0))
            if us < 1e-12:
                y = np.full(T, mu, dtype=np.float64)
            else:
                y = (u - np.mean(u)) / us
                y = mu + sd * y

        Y[:, j] = y
        surrogate_rho[j] = lag1_corr(y)
        surrogate_sd[j] = float(np.std(y, ddof=0))

    diag = {
        "mean_empirical_rho": float(np.mean(empirical_rho)),
        "mean_surrogate_rho": float(np.mean(surrogate_rho)),
        "mean_abs_rho_error": float(np.mean(np.abs(empirical_rho - surrogate_rho))),
        "median_empirical_sd": float(np.median(empirical_sd)),
        "median_surrogate_sd": float(np.median(surrogate_sd)),
    }
    return Y, diag


def compute_ar1_surrogates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    expected_rep_rows = len(SEEDS) * N_EPISODES * N_AR1_SURROGATES

    if (
        AR1_REP_CSV.exists()
        and AR1_EP_CSV.exists()
        and AR1_SEED_CSV.exists()
        and not FORCE_AR1_RECOMPUTE
    ):
        rep = pd.read_csv(AR1_REP_CSV)
        ep = pd.read_csv(AR1_EP_CSV)
        seed = pd.read_csv(AR1_SEED_CSV)
        if (
            len(rep) == expected_rep_rows
            and len(ep) == len(SEEDS) * N_EPISODES
            and len(seed) == len(SEEDS)
        ):
            print(f"Loaded AR(1) cache: {AR1_REP_CSV}")
            return rep, ep, seed

    rows = []
    diag_rows = []

    for k, seed in enumerate(SEEDS, start=1):
        path = raw_path("default", STEP48, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[AR1 48k default] seed {seed:02d} ({k}/{len(SEEDS)})")

        df = pd.read_parquet(path)
        g_cols = sg.latent_columns(df, "g")
        if not g_cols:
            raise RuntimeError(f"No g columns in {path}")

        for ep in range(N_EPISODES):
            e = df[df.episode == ep].sort_values("t").reset_index(drop=True)
            if len(e) != N_STEPS:
                raise RuntimeError(
                    f"AR1 source seed={seed} ep={ep}: got {len(e)} rows, expected {N_STEPS}"
                )
            G = e[g_cols].to_numpy(np.float64)

            for rep in range(N_AR1_SURROGATES):
                gen_seed = (
                    AR1_RANDOM_SEED
                    + stable_seed("ar1gen", seed, ep, rep)
                ) & 0xFFFFFFFF
                rng = np.random.default_rng(gen_seed)
                S, diag = make_independent_ar1_surrogate(G, rng)

                analysis_seed = sg.episode_analysis_seed(
                    seed,
                    STEP48,
                    ep,
                    f"g_ar1_rep{rep}",
                )
                sm = sg.decompose_trajectory(
                    S,
                    analysis_seed,
                    include_atoms=False,
                )

                rows.append({
                    "seed": seed,
                    "episode": ep,
                    "rep": rep,
                    "phi_r_g_ar1": sm["phi_r"],
                })
                diag_rows.append({
                    "seed": seed,
                    "episode": ep,
                    "rep": rep,
                    **diag,
                })

        # Safe-restart cache by seed.
        pd.DataFrame(rows).to_csv(AR1_REP_CSV, index=False)
        pd.DataFrame(diag_rows).to_csv(AR1_DIAG_CSV, index=False)

    rep_df = pd.DataFrame(rows).sort_values(["seed", "episode", "rep"]).reset_index(drop=True)
    diag_df = pd.DataFrame(diag_rows).sort_values(["seed", "episode", "rep"]).reset_index(drop=True)

    ep_df = (
        rep_df.groupby(["seed", "episode"], as_index=False)["phi_r_g_ar1"]
        .median()
        .rename(columns={"phi_r_g_ar1": "phi_r_g_ar1_episode_median"})
    )
    seed_df = (
        ep_df.groupby("seed", as_index=False)["phi_r_g_ar1_episode_median"]
        .median()
        .rename(columns={"phi_r_g_ar1_episode_median": "phi_r_g_ar1"})
    )

    rep_df.to_csv(AR1_REP_CSV, index=False)
    ep_df.to_csv(AR1_EP_CSV, index=False)
    seed_df.to_csv(AR1_SEED_CSV, index=False)
    diag_df.to_csv(AR1_DIAG_CSV, index=False)
    return rep_df, ep_df, seed_df

# ============================================================================
# TABLE ASSEMBLY / EFFECTS
# ============================================================================

def state_seed(seed_df, condition, checkpoint):
    s = seed_df[
        (seed_df.condition == condition)
        & (seed_df.checkpoint == checkpoint)
    ].set_index("seed")
    return s.loc[list(SEEDS)]


def prepare_effects(seed_df, ar1_seed):
    s0 = state_seed(seed_df, "default", STEP0)
    sd = state_seed(seed_df, "default", STEP48)
    sc = state_seed(seed_df, "coupled", STEP48)
    sa = ar1_seed.set_index("seed").loc[list(SEEDS)]

    # Figure 2 stats.
    fig2_rows = []
    fig2_rows.append(
        paired_stats(
            sd["phi_r_g"].to_numpy(float),
            sd["phi_r_z"].to_numpy(float),
            "48k_default_phi_g_minus_phi_z",
            "phi_r_g - phi_r_z",
        )
    )
    fig2_rows.append(
        paired_stats(
            sd["phi_r_g"].to_numpy(float),
            sd["phi_r_g_shuffled"].to_numpy(float),
            "48k_default_intact_minus_temporal_shuffle",
            "intact_g - shuffled_g",
        )
    )
    fig2_rows.append(
        paired_stats(
            sd["phi_r_g"].to_numpy(float),
            sa["phi_r_g_ar1"].to_numpy(float),
            "48k_default_intact_minus_ar1",
            "intact_g - independent_AR1_g",
        )
    )
    fig2_stats = pd.DataFrame(fig2_rows)
    fig2_stats.to_csv(FIG2_STATS_CSV, index=False)

    # Figure 3 learning effects.
    fig3_rows = []
    learning_effects = {}
    for metric in ["phi_r_g", *PRIMARY_GROUP_METRICS, "phi_r_z"]:
        eff = sd[metric].to_numpy(float) - s0[metric].to_numpy(float)
        learning_effects[metric] = eff
        row = effect_stats(eff, f"learning_48k_minus_0k|{metric}")
        row["metric"] = metric
        row["median_step0"] = float(np.median(s0[metric]))
        row["median_step48"] = float(np.median(sd[metric]))
        fig3_rows.append(row)

    for metric in ["whole_to_single_sum", "whole_to_parts", "downward_reconstructed"]:
        eff = sd[metric].to_numpy(float) - s0[metric].to_numpy(float)
        learning_effects[metric] = eff
        row = effect_stats(eff, f"learning_48k_minus_0k|{metric}")
        row["metric"] = metric
        row["median_step0"] = float(np.median(s0[metric]))
        row["median_step48"] = float(np.median(sd[metric]))
        fig3_rows.append(row)

    fig3_stats = pd.DataFrame(fig3_rows)
    fig3_stats.to_csv(FIG3_STATS_CSV, index=False)

    # Figure 4 learning vs coupling effects.
    fig4_rows = []
    coupling_effects = {}
    for metric in ["phi_r_g", *PRIMARY_GROUP_METRICS, "phi_r_z"]:
        le = sd[metric].to_numpy(float) - s0[metric].to_numpy(float)
        ce = sc[metric].to_numpy(float) - sd[metric].to_numpy(float)
        coupling_effects[metric] = ce

        lr = effect_stats(le, f"fig4_learning|{metric}")
        lr.update({"metric": metric, "effect_type": "learning_48kD_minus_0kD"})
        cr = effect_stats(ce, f"fig4_coupling|{metric}")
        cr.update({"metric": metric, "effect_type": "coupled48k_minus_default48k"})
        fig4_rows.extend([lr, cr])

    fig4_stats = pd.DataFrame(fig4_rows)
    fig4_stats.to_csv(FIG4_STATS_CSV, index=False)

    return {
        "s0": s0,
        "sd": sd,
        "sc": sc,
        "sa": sa,
        "learning": learning_effects,
        "coupling": coupling_effects,
        "fig2_stats": fig2_stats,
        "fig3_stats": fig3_stats,
        "fig4_stats": fig4_stats,
    }

# ============================================================================
# PLOTTING HELPERS: HALF-VIOLIN RAINCLOUDS
# ============================================================================

def set_plot_style():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
    })


def _half_violin(ax, values, x, side="left", width=0.72, color="C0", alpha=0.22):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2 or np.nanstd(vals) < 1e-12:
        return

    vp = ax.violinplot(
        [vals], positions=[x], widths=width,
        showmeans=False, showmedians=False, showextrema=False,
        points=120,
    )
    for body in vp["bodies"]:
        verts = body.get_paths()[0].vertices
        if side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], x)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], x)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(0.8)


def _median_ci_marker(ax, x, values, tag, color="k", xoffset=0.0):
    med, lo, hi = bootstrap_median_ci(values, tag)
    xx = x + xoffset
    ax.vlines(xx, lo, hi, color=color, linewidth=2.0, zorder=6)
    ax.plot([xx - 0.055, xx + 0.055], [med, med], color=color, linewidth=3.0, zorder=7)
    return med, lo, hi


def paired_absolute_panel(ax, a, b, labels, title, tag, colors=("C0", "C1")):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, x2 = 0.0, 1.0

    # paired seed lines behind the rainclouds
    for i in range(len(a)):
        ax.plot([x1 + 0.07, x2 + 0.07], [a[i], b[i]], color="0.76", linewidth=0.65, alpha=0.42, zorder=1)

    rng = np.random.default_rng(stable_seed("pairedplot", tag))
    for vals, x, color, lab in [(a, x1, colors[0], labels[0]), (b, x2, colors[1], labels[1])]:
        _half_violin(ax, vals, x - 0.03, side="left", color=color)
        jitter = rng.uniform(0.075, 0.19, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, alpha=0.72, color=color, edgecolors="none", zorder=4)
        _median_ci_marker(ax, x, vals, f"abs|{tag}|{lab}", color=color, xoffset=-0.005)

    ax.set_xticks([x1, x2], labels)
    ax.set_xlim(-0.45, 1.45)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)


def effect_raincloud(ax, groups, labels, title, tag, colors=None, ylabel=None):
    if colors is None:
        colors = [f"C{i}" for i in range(len(groups))]
    rng = np.random.default_rng(stable_seed("effectplot", tag))
    xs = np.arange(len(groups), dtype=float)

    for x, vals, lab, color in zip(xs, groups, labels, colors):
        vals = np.asarray(vals, dtype=float)
        _half_violin(ax, vals, x - 0.03, side="left", color=color)
        jitter = rng.uniform(0.075, 0.19, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, alpha=0.72, color=color, edgecolors="none", zorder=4)
        _median_ci_marker(ax, x, vals, f"effect|{tag}|{lab}", color=color, xoffset=-0.005)

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.7, zorder=0)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.45, len(groups) - 0.55)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)


def save_figure(fig, stem):
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIG_ROOT / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", dpi=DPI)
    if SAVE_PDF:
        fig.savefig(FIG_ROOT / f"{stem}.pdf", bbox_inches="tight")
    print(f"Saved: {png}")

# ============================================================================
# FIGURES
# ============================================================================

def make_figure2(ctx):
    sd = ctx["sd"]
    sa = ctx["sa"]

    g = sd["phi_r_g"].to_numpy(float)
    z = sd["phi_r_z"].to_numpy(float)
    sh = sd["phi_r_g_shuffled"].to_numpy(float)
    ar = sa["phi_r_g_ar1"].to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    paired_absolute_panel(
        axes[0], g, z,
        [r"$g$", r"$z$"],
        r"A. Architectural localization",
        "fig2A",
        colors=("C0", "C1"),
    )
    axes[0].set_ylabel(r"Seed-level $\Phi_r$")

    paired_absolute_panel(
        axes[1], g, sh,
        ["Intact", "Time-shuffled"],
        r"B. Temporal-order dependence",
        "fig2B",
        colors=("C0", "C2"),
    )

    paired_absolute_panel(
        axes[2], g, ar,
        ["Intact", "AR(1) null"],
        r"C. Beyond first-order persistence",
        "fig2C",
        colors=("C0", "C3"),
    )

    fig.suptitle(
        r"Figure 2. The slow latent provides the relevant temporal substrate",
        y=1.03,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure2_temporal_substrate")
    return fig


def make_figure3(ctx):
    s0 = ctx["s0"]
    sd = ctx["sd"]
    le = ctx["learning"]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))

    paired_absolute_panel(
        axes[0],
        s0["phi_r_g"].to_numpy(float),
        sd["phi_r_g"].to_numpy(float),
        ["Step 0", "48k"],
        r"A. Scalar $\Phi_r(g)$",
        "fig3A",
        colors=("C4", "C0"),
    )
    axes[0].set_ylabel(r"Seed-level $\Phi_r(g)$")

    effect_raincloud(
        axes[1],
        [
            le["group_decoupling_g"],
            le["group_downward_g"],
            le["group_part_driven_g"],
        ],
        ["Decoupling", "Whole-to-part", "Part-driven"],
        "B. Learning changes PhiID composition",
        "fig3B",
        colors=("C0", "C1", "C2"),
        ylabel="48k default - step0 default",
    )
    axes[1].tick_params(axis="x", rotation=18)

    effect_raincloud(
        axes[2],
        [
            le["whole_to_single_sum"],
            le["whole_to_parts"],
            le["downward_reconstructed"],
        ],
        ["Whole→single\nsum", "Whole→parts", "Total\nwhole→part"],
        "C. Whole-to-part decomposition",
        "fig3C",
        colors=("C1", "C5", "C6"),
    )

    fig.suptitle(
        "Figure 3. Learning reorganizes the informational structure of the slow latent",
        y=1.03,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure3_learning_reorganization")
    return fig


def make_figure4(ctx):
    le = ctx["learning"]
    ce = ctx["coupling"]

    metrics = [
        "phi_r_g",
        "group_decoupling_g",
        "group_downward_g",
        "group_part_driven_g",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        effect_raincloud(
            ax,
            [le[metric], ce[metric]],
            ["Learning\n48kD - 0kD", "Coupling\n48kC - 48kD"],
            METRICS[metric],
            f"fig4|{metric}",
            colors=("C0", "C3"),
            ylabel="Paired seed effect",
        )

    fig.suptitle(
        "Figure 4. Paired effects of learning and policy-side gradient coupling",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure4_learning_vs_coupling")
    return fig

# ============================================================================
# CONSOLE REPORT
# ============================================================================

def print_stats(ctx, ar1_diag):
    print("\n" + "=" * 84)
    print("FIGURE 2 — MEASUREMENT QUALIFICATION")
    print("=" * 84)
    for _, r in ctx["fig2_stats"].iterrows():
        print(
            f"{r['contrast']:48s} "
            f"effect={r['median_effect']:+.6f} "
            f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] "
            f"p={r['wilcoxon_p']:.4g}"
        )

    if ar1_diag is not None and len(ar1_diag):
        print("\nAR(1) diagnostic (across all surrogate replicates):")
        print(
            f"  mean |rho_emp-rho_sur| = "
            f"{float(ar1_diag['mean_abs_rho_error'].mean()):.4f}"
        )
        print(
            f"  median SD empirical/surrogate = "
            f"{float(ar1_diag['median_empirical_sd'].median()):.6f} / "
            f"{float(ar1_diag['median_surrogate_sd'].median()):.6f}"
        )

    print("\n" + "=" * 84)
    print("FIGURE 3 — LEARNING EFFECT: 48k DEFAULT - STEP0 DEFAULT")
    print("=" * 84)
    order = [
        "phi_r_g",
        "group_decoupling_g",
        "group_downward_g",
        "group_part_driven_g",
        "whole_to_single_sum",
        "whole_to_parts",
        "downward_reconstructed",
        "phi_r_z",
    ]
    fs = ctx["fig3_stats"].set_index("metric")
    for metric in order:
        if metric not in fs.index:
            continue
        r = fs.loc[metric]
        print(
            f"{metric:28s} "
            f"{r['median_effect']:+.6f} "
            f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] "
            f"p={r['wilcoxon_p']:.4g} "
            f"(+/-={int(r['n_positive'])}/{int(r['n_negative'])})"
        )

    print("\n" + "=" * 84)
    print("FIGURE 4 — LEARNING VS COUPLING")
    print("=" * 84)
    f4 = ctx["fig4_stats"]
    for metric in ["phi_r_g", *PRIMARY_GROUP_METRICS, "phi_r_z"]:
        s = f4[f4.metric == metric].set_index("effect_type")
        if len(s) != 2:
            continue
        l = s.loc["learning_48kD_minus_0kD"]
        c = s.loc["coupled48k_minus_default48k"]
        print(
            f"{metric:28s} "
            f"learning={l['median_effect']:+.6f} "
            f"[{l['ci95_low']:+.6f},{l['ci95_high']:+.6f}]   "
            f"coupling={c['median_effect']:+.6f} "
            f"[{c['ci95_low']:+.6f},{c['ci95_high']:+.6f}]"
        )

# ============================================================================
# MANIFEST
# ============================================================================

def write_manifest():
    manifest = {
        "analysis_window": "full 800-step matched serpentine trajectory; no cutoff",
        "conditions": [
            "default step0",
            "default step48000",
            "coupled step48000",
        ],
        "n_training_seeds": len(SEEDS),
        "episodes_per_seed": N_EPISODES,
        "steps_per_episode": N_STEPS,
        "seed_summary": "median across 10 episodes",
        "group_summary": "median across 30 training seeds",
        "uncertainty": f"percentile seed bootstrap, B={N_BOOTSTRAP}",
        "ar1_surrogates_per_episode": N_AR1_SURROGATES,
        "ar1_definition": (
            "independent per-dimension Gaussian AR(1); empirical mean and SD "
            "matched exactly after simulation; empirical lag-1 autocorrelation "
            "targeted per dimension; cross-dimensional coupling removed"
        ),
        "phi_pipeline_reused_from": "cear_pilot.analysis.phi.analyze_stopgrad_matched",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print("\n" + "=" * 84)
    print("CEAR P1 JOURNAL — FIGURES 2, 3, 4")
    print("=" * 84)
    print("Repo   :", REPO_ROOT)
    print("Raw    :", RAW_ROOT)
    print("Output :", OUT_ROOT)
    print("Window : full 800 steps (no cutoff)")
    print(f"AR(1)  : {N_AR1_SURROGATES} independent surrogate replicates / episode")

    ep = load_or_compute_unified_episode_table()
    seed_df = make_seed_table(ep)

    _, _, ar1_seed = compute_ar1_surrogates()
    ar1_diag = pd.read_csv(AR1_DIAG_CSV) if AR1_DIAG_CSV.exists() else None

    ctx = prepare_effects(seed_df, ar1_seed)
    print_stats(ctx, ar1_diag)
    write_manifest()

    fig2 = make_figure2(ctx)
    fig3 = make_figure3(ctx)
    fig4 = make_figure4(ctx)

    dt = time.time() - t0
    print("\n" + "=" * 84)
    print("DONE")
    print("=" * 84)
    print(f"Elapsed: {dt / 60:.2f} min")
    print("Figure 2:", FIG_ROOT / "figure2_temporal_substrate.png")
    print("Figure 3:", FIG_ROOT / "figure3_learning_reorganization.png")
    print("Figure 4:", FIG_ROOT / "figure4_learning_vs_coupling.png")
    print("Stats   :", FIG2_STATS_CSV)
    print("          ", FIG3_STATS_CSV)
    print("          ", FIG4_STATS_CSV)

    if SHOW_FIGURES:
        # In Spyder this sends all three figures to the Plots pane.
        plt.show()
    else:
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)


if __name__ == "__main__":
    main()
