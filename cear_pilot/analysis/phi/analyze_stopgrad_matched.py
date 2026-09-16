# cear_pilot/analysis/phi/analyze_stopgrad_matched.py
# -*- coding: utf-8 -*-

"""
Matched-replay stop-gradient analysis for the journal extension.

SPYDER:
    Save as:
        cear_pilot/analysis/phi/analyze_stopgrad_matched.py
    Then press Run.

NO command-line arguments are used.

Pipeline
--------
800-step episode
    -> episode-level Phi_r / PhiID
10 episodes
    -> seed median
30 seeds
    -> group median + seed-bootstrap 95% CI

Longitudinal effect
-------------------
Delta_default(t) = M_default(t) - M_default(12k)
Delta_coupled(t) = M_coupled(t) - M_coupled(12k)

DeltaDelta(t)
    = Delta_coupled(t) - Delta_default(t)

Main metrics
------------
- Phi_r(g)
- decoupling
- whole->part / downward contribution
- part-driven contribution

Optional control
----------------
- Phi_r(z)

Outputs
-------
outputs/journal_stopgrad_analysis/
    journal_phi_episode.csv
    journal_phi_seed.csv
    journal_phi_checkpoint_summary.csv
    journal_phi_seed_paired_changes.csv
    journal_phi_paired_effects.csv
    validation_report.txt
    analysis_manifest.json
    figures/
"""

from __future__ import annotations


# =====================================================================
# CONFIG
# =====================================================================

CONDITIONS = ("default", "coupled")

CHECKPOINTS = (
    12000,
    18000,
    24000,
    30000,
    36000,
    42000,
    48000,
)

SEEDS = tuple(range(1, 31))

N_EPISODES = 10
N_STEPS = 800

BASELINE_STEP = 12000
FINAL_STEP = 48000


# Phi_r(z) is a useful control but roughly doubles PhiID computation.
# Leave True for the complete first pass.
COMPUTE_Z_CONTROL = True


# Seed bootstrap
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260915


# corrected_zscore() adds tiny random noise to dead units.
# We explicitly seed this identically for matched default/coupled pairs.
ANALYSIS_RANDOM_SEED = 731991


# Safe restart:
# episode CSV is cached every few source files.
FORCE_RECOMPUTE = False
CACHE_EVERY_FILES = 5


STRICT_COMPLETENESS = True
STRICT_MATCHED_INPUTS = True
STRICT_BRANCHPOINT = True


MAKE_FIGURES = True
SHOW_FIGURES = True


# =====================================================================
# IMPORTS
# =====================================================================

import hashlib
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


# =====================================================================
# REPO PATH
# =====================================================================

THIS_FILE = Path(__file__).resolve()

# <repo>/cear_pilot/analysis/phi/this_file.py
REPO_ROOT = THIS_FILE.parents[3]

if not (REPO_ROOT / "cear_pilot").exists():

    # fallback if Spyder does something unusual with the file path
    cwd = Path.cwd().resolve()

    if (cwd / "cear_pilot").exists():
        REPO_ROOT = cwd

    else:
        raise RuntimeError(
            "Could not locate repository root.\n"
            "Please save this script under:\n"
            "cear_pilot/analysis/phi/"
        )


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from cear_pilot.analysis.phi import information as info
from cear_pilot.analysis.phi import phi_atoms


# =====================================================================
# PATHS
# =====================================================================

INPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "journal_replay_matched"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "journal_stopgrad_analysis"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"


EPISODE_CSV = (
    OUTPUT_ROOT
    / "journal_phi_episode.csv"
)

SEED_CSV = (
    OUTPUT_ROOT
    / "journal_phi_seed.csv"
)

CHECKPOINT_SUMMARY_CSV = (
    OUTPUT_ROOT
    / "journal_phi_checkpoint_summary.csv"
)

SEED_PAIRED_CSV = (
    OUTPUT_ROOT
    / "journal_phi_seed_paired_changes.csv"
)

PAIRED_EFFECTS_CSV = (
    OUTPUT_ROOT
    / "journal_phi_paired_effects.csv"
)

VALIDATION_TXT = (
    OUTPUT_ROOT
    / "validation_report.txt"
)

MANIFEST_JSON = (
    OUTPUT_ROOT
    / "analysis_manifest.json"
)


# =====================================================================
# MAIN METRICS
# =====================================================================

MAIN_METRICS = (

    "phi_r_g",

    "group_decoupling_g",

    "group_downward_g",

    "group_part_driven_g",

)


METRIC_LABELS = {

    "phi_r_g":
        r"$\Phi_r(g)$",

    "group_decoupling_g":
        "Decoupling",

    "group_downward_g":
        "Whole-to-part contribution",

    "group_part_driven_g":
        "Part-driven contribution",

    "phi_r_z":
        r"$\Phi_r(z)$",

}


# =====================================================================
# SMALL UTILITIES
# =====================================================================

def source_path(
    condition,
    checkpoint,
    seed,
):

    return (

        INPUT_ROOT
        / condition
        / f"step{checkpoint:05d}"
        / f"seed{seed}"
        / "serpentine_dwell4"
        / "traj.parquet"

    )


def latent_columns(
    df,
    prefix,
):

    cc = [

        c
        for c in df.columns
        if c.startswith(prefix + "_")

    ]

    return sorted(

        cc,

        key=lambda x:
        int(x.rsplit("_", 1)[1]),

    )


def stable_seed(*parts):

    text = "|".join(

        str(x)
        for x in parts

    ).encode("utf-8")

    return (
        zlib.crc32(text)
        & 0xFFFFFFFF
    )


def episode_analysis_seed(
    seed,
    checkpoint,
    episode,
    latent,
):

    # CONDITION deliberately omitted.
    #
    # matched default/coupled pair gets the
    # exact same analysis-side random seed.

    return (

        ANALYSIS_RANDOM_SEED

        + stable_seed(
            seed,
            checkpoint,
            episode,
            latent,
        )

    ) & 0xFFFFFFFF


# =====================================================================
# BOOTSTRAP
# =====================================================================

def bootstrap_median_ci(
    values,
    label,
):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]


    if len(x) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    med = float(
        np.median(x)
    )


    if len(x) == 1:

        return (
            med,
            med,
            med,
        )


    rng = np.random.default_rng(

        (
            BOOTSTRAP_SEED
            + stable_seed(label)
        )
        & 0xFFFFFFFF

    )


    idx = rng.integers(

        0,
        len(x),

        size=(
            N_BOOTSTRAP,
            len(x),
        ),

    )


    boot = np.median(
        x[idx],
        axis=1,
    )


    lo, hi = np.percentile(

        boot,

        [
            2.5,
            97.5,
        ],

    )


    return (

        med,

        float(lo),

        float(hi),

    )


def safe_wilcoxon(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]


    if len(x) == 0:

        return np.nan


    if np.allclose(
        x,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):

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


# =====================================================================
# PhiID CORE
# =====================================================================

def decompose_trajectory(
    X,
    rng_seed,
    include_atoms=True,
):

    """
    Compute Phi_r and, optionally, atom decomposition
    from ONE shared Fiedler partition / PhiID lattice.

    This intentionally avoids calling info.local_phi_r()
    because its current in-place += implementation can
    mutate the BASE atom stored in the lattice.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )


    if X.ndim != 2:

        raise ValueError(
            f"Expected (T,d), got {X.shape}"
        )


    if X.shape[0] < 10:

        raise ValueError(
            f"Trajectory too short: {X.shape}"
        )


    if X.shape[1] < 2:

        raise ValueError(
            f"Latent dimension too small: {X.shape}"
        )


    if not np.isfinite(X).all():

        raise ValueError(
            "NaN or inf in trajectory"
        )


    # deterministic dead-unit noise / spectral solver seed
    np.random.seed(
        int(rng_seed)
    )

    random.seed(
        int(rng_seed)
    )


    # --------------------------------------------------------------
    # same standardization / MI / spectral partition as repo code
    # --------------------------------------------------------------

    x = info.corrected_zscore(

        X.T.copy(),

        axis=1,

    )


    mi = info.mutual_information_matrix_fast(

        x,

        alpha=0.05,

        lag=1,

        bonferonni=True,

    )


    idx1, idx2 = (
        info.minimum_information_bipartition(

            mi,

            noise=True,

        )
    )


    if (
        len(idx1) == 0
        or len(idx2) == 0
    ):

        d = X.shape[1]

        idx1 = list(
            range(d // 2)
        )

        idx2 = list(
            range(
                d // 2,
                d,
            )
        )


    # --------------------------------------------------------------
    # reduce partition halves to two-node system
    # --------------------------------------------------------------

    x2 = np.vstack([

        x[idx1].mean(
            axis=0,
            keepdims=True,
        ),

        x[idx2].mean(
            axis=0,
            keepdims=True,
        ),

    ])


    # --------------------------------------------------------------
    # local PhiID
    # --------------------------------------------------------------

    lattice = info.local_phi_id(

        0,
        1,
        x2,

    )


    # copy every Phi_r atom so no lattice array can be
    # accidentally mutated by in-place arithmetic

    atom_series = {

        atom:
        np.asarray(
            lattice.nodes[atom]["pi"],
            dtype=np.float64,
        ).copy()

        for atom
        in phi_atoms.ALL_ATOMS

    }


    # canonical scalar Phi_r:
    #
    # local sum of nine atoms
    # THEN temporal median

    phi_local = np.sum(

        np.stack([

            atom_series[atom]

            for atom
            in phi_atoms.ALL_ATOMS

        ]),

        axis=0,

    )


    result = {

        "phi_r":
            float(
                np.median(phi_local)
            ),

        "partition_n1":
            int(len(idx1)),

        "partition_n2":
            int(len(idx2)),

        "partition_signature":

            "A["

            + ",".join(
                map(
                    str,
                    sorted(idx1),
                )
            )

            + "]|B["

            + ",".join(
                map(
                    str,
                    sorted(idx2),
                )
            )

            + "]",

    }


    if not include_atoms:

        return result


    # --------------------------------------------------------------
    # individual atom temporal medians
    # --------------------------------------------------------------

    atom_medians = {

        phi_atoms.ATOM_LABEL[atom]:

            float(
                np.median(
                    atom_series[atom]
                )
            )

        for atom
        in phi_atoms.ALL_ATOMS

    }


    result.update(
        atom_medians
    )


    # --------------------------------------------------------------
    # grouped contributions
    #
    # IMPORTANT:
    # matches current phi_atoms.py convention:
    #
    # median each atom across time first,
    # then sum member atoms.
    # --------------------------------------------------------------

    result["group_decoupling"] = float(

        sum(

            np.median(
                atom_series[a]
            )

            for a
            in phi_atoms.DECOUPLING_ATOMS

        )

    )


    result["group_downward"] = float(

        sum(

            np.median(
                atom_series[a]
            )

            for a
            in phi_atoms.DOWNWARD_ATOMS

        )

    )


    result["group_part_driven"] = float(

        sum(

            np.median(
                atom_series[a]
            )

            for a
            in phi_atoms.PART_DRIVEN_ATOMS

        )

    )


    # diagnostic:
    #
    # sum(median(atom)) generally does NOT have to equal
    # median(sum(atom))

    result["phi_r_atom_median_sum"] = float(

        sum(
            atom_medians.values()
        )

    )


    result["phi_r_minus_atom_median_sum"] = float(

        result["phi_r"]
        - result["phi_r_atom_median_sum"]

    )


    return result


# =====================================================================
# MATCHED INPUT HASH
# =====================================================================

def external_episode_hash(
    df,
):

    """
    Hash the actual exogenous stream presented to the network.

    default/coupled hashes must be identical for every
    checkpoint x seed x episode.
    """

    obs_cols = latent_columns(
        df,
        "obs",
    )


    int_cols = [

        "episode_seed",

        "t",

        "x",
        "y",

        "zone_id",

        "x_obs",
        "y_obs",

        "zone_obs",

        "action_env",

        "action_replay",

    ]


    missing = [

        c

        for c
        in int_cols + obs_cols

        if c not in df.columns

    ]


    if missing:

        raise KeyError(
            f"Missing matched-input columns: {missing}"
        )


    h = hashlib.sha256()


    arr_int = df[
        int_cols
    ].to_numpy(

        dtype=np.int64,

        copy=True,

    )


    arr_obs = df[
        obs_cols
    ].to_numpy(

        dtype=np.float64,

        copy=True,

    )


    h.update(

        np.ascontiguousarray(
            arr_int
        ).tobytes()

    )


    h.update(

        np.ascontiguousarray(
            arr_obs
        ).tobytes()

    )


    return h.hexdigest()


# =====================================================================
# INPUT SCAN
# =====================================================================

def scan_inputs():

    paths = []
    missing = []


    for condition in CONDITIONS:

        for checkpoint in CHECKPOINTS:

            for seed in SEEDS:

                p = source_path(

                    condition,
                    checkpoint,
                    seed,

                )

                paths.append(p)

                if not p.exists():

                    missing.append(p)


    print(
        f"Expected parquet files: {len(paths)}"
    )

    print(
        f"Missing: {len(missing)}"
    )


    if missing:

        for p in missing:

            print(
                "MISSING:",
                p.relative_to(REPO_ROOT),
            )


        if STRICT_COMPLETENESS:

            raise FileNotFoundError(
                "Matched-replay dataset is incomplete."
            )


    return [

        p

        for p
        in paths

        if p.exists()

    ]


# =====================================================================
# EPISODE TABLE + CACHE
# =====================================================================

def cache_complete(
    cache,
    condition,
    checkpoint,
    seed,
):

    if cache.empty:

        return False


    sub = cache[

        (cache["condition"] == condition)

        &

        (cache["checkpoint"] == checkpoint)

        &

        (cache["seed"] == seed)

    ]


    expected_eps = set(
        range(N_EPISODES)
    )


    got_eps = set(
        sub["episode"].astype(int)
    )


    needed = {

        "phi_r_g",

        "group_decoupling_g",

        "group_downward_g",

        "group_part_driven_g",

        "external_sha256",

    }


    if COMPUTE_Z_CONTROL:

        needed.add(
            "phi_r_z"
        )


    return (

        got_eps == expected_eps

        and

        needed.issubset(
            sub.columns
        )

    )


def compute_episode_table(
    paths,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    if (
        EPISODE_CSV.exists()
        and not FORCE_RECOMPUTE
    ):

        cache = pd.read_csv(
            EPISODE_CSV
        )

        print(
            f"Loaded episode cache: {len(cache)} rows"
        )


    else:

        cache = pd.DataFrame()


    since_save = 0


    for n, path in enumerate(
        paths,
        start=1,
    ):

        condition = (
            path.parents[3].name
        )

        checkpoint = int(
            path.parents[2].name.replace(
                "step",
                "",
            )
        )

        seed = int(
            path.parents[1].name.replace(
                "seed",
                "",
            )
        )


        if (

            not FORCE_RECOMPUTE

            and cache_complete(
                cache,
                condition,
                checkpoint,
                seed,
            )

        ):

            print(

                f"[{n:03d}/{len(paths):03d}] "
                f"cache "
                f"{condition:7s} "
                f"seed={seed:02d} "
                f"step={checkpoint}"

            )

            continue


        # remove stale / partial source cache

        if not cache.empty:

            mask = (

                (cache["condition"] == condition)

                &

                (cache["checkpoint"] == checkpoint)

                &

                (cache["seed"] == seed)

            )

            cache = cache.loc[
                ~mask
            ].copy()


        print(

            f"[{n:03d}/{len(paths):03d}] "
            f"calc  "
            f"{condition:7s} "
            f"seed={seed:02d} "
            f"step={checkpoint}"

        )


        df = pd.read_parquet(
            path
        )


        g_cols = latent_columns(
            df,
            "g",
        )


        z_cols = latent_columns(
            df,
            "z",
        )


        if not g_cols:

            raise KeyError(
                f"No g columns: {path}"
            )


        if (
            COMPUTE_Z_CONTROL
            and not z_cols
        ):

            raise KeyError(
                f"No z columns: {path}"
            )


        episode_ids = sorted(

            df["episode"]
            .unique()
            .astype(int)

        )


        if episode_ids != list(
            range(N_EPISODES)
        ):

            raise ValueError(
                f"{path}: bad episode IDs {episode_ids}"
            )


        rows = []


        for ep in episode_ids:

            e = (

                df[
                    df["episode"] == ep
                ]

                .sort_values("t")

                .reset_index(
                    drop=True
                )

            )


            if len(e) != N_STEPS:

                raise ValueError(

                    f"{path}: "
                    f"episode {ep} "
                    f"has {len(e)} rows; "
                    f"expected {N_STEPS}"

                )


            # ------------------------------------------------------
            # g
            # ------------------------------------------------------

            G = e[
                g_cols
            ].to_numpy(
                dtype=np.float64
            )


            gm = decompose_trajectory(

                G,

                episode_analysis_seed(
                    seed,
                    checkpoint,
                    ep,
                    "g",
                ),

                include_atoms=True,

            )


            row = {

                "condition":
                    condition,

                "checkpoint":
                    checkpoint,

                "seed":
                    seed,

                "episode":
                    ep,

                "episode_seed":
                    int(
                        e[
                            "episode_seed"
                        ].iloc[0]
                    ),

                "n_steps":
                    len(e),

                "external_sha256":
                    external_episode_hash(e),

                "phi_r_g":
                    gm["phi_r"],

                "group_decoupling_g":
                    gm["group_decoupling"],

                "group_downward_g":
                    gm["group_downward"],

                "group_part_driven_g":
                    gm["group_part_driven"],

                "phi_r_atom_median_sum_g":
                    gm[
                        "phi_r_atom_median_sum"
                    ],

                "phi_r_minus_atom_median_sum_g":
                    gm[
                        "phi_r_minus_atom_median_sum"
                    ],

                "g_partition_n1":
                    gm["partition_n1"],

                "g_partition_n2":
                    gm["partition_n2"],

                "g_partition_signature":
                    gm[
                        "partition_signature"
                    ],

            }


            # all nine individual atoms
            for atom in phi_atoms.ALL_ATOMS:

                atom_label = (
                    phi_atoms.ATOM_LABEL[
                        atom
                    ]
                )

                row[
                    f"atom_{atom_label}_g"
                ] = gm[
                    atom_label
                ]


            # ------------------------------------------------------
            # z scalar control
            # ------------------------------------------------------

            if COMPUTE_Z_CONTROL:

                Z = e[
                    z_cols
                ].to_numpy(
                    dtype=np.float64
                )


                zm = decompose_trajectory(

                    Z,

                    episode_analysis_seed(
                        seed,
                        checkpoint,
                        ep,
                        "z",
                    ),

                    include_atoms=False,

                )


                row["phi_r_z"] = (
                    zm["phi_r"]
                )


                row[
                    "z_partition_n1"
                ] = (
                    zm["partition_n1"]
                )


                row[
                    "z_partition_n2"
                ] = (
                    zm["partition_n2"]
                )


                row[
                    "z_partition_signature"
                ] = (
                    zm[
                        "partition_signature"
                    ]
                )


            rows.append(
                row
            )


        cache = pd.concat(

            [
                cache,
                pd.DataFrame(rows),
            ],

            ignore_index=True,

        )


        since_save += 1


        if since_save >= CACHE_EVERY_FILES:

            cache = cache.sort_values(

                [
                    "condition",
                    "checkpoint",
                    "seed",
                    "episode",
                ]

            ).reset_index(
                drop=True
            )


            cache.to_csv(

                EPISODE_CSV,

                index=False,

            )


            since_save = 0


            print(
                "    cache saved"
            )


    cache = cache.sort_values(

        [
            "condition",
            "checkpoint",
            "seed",
            "episode",
        ]

    ).reset_index(
        drop=True
    )


    cache.to_csv(

        EPISODE_CSV,

        index=False,

    )


    return cache


# =====================================================================
# VALIDATE MATCHED STREAMS
# =====================================================================

def validate_matched_inputs(
    episode_df,
):

    keys = [
        "checkpoint",
        "seed",
        "episode",
    ]


    A = (

        episode_df[
            episode_df["condition"]
            == "default"
        ]

        [
            keys
            + ["external_sha256"]
        ]

        .rename(
            columns={
                "external_sha256":
                    "hash_default"
            }
        )

    )


    B = (

        episode_df[
            episode_df["condition"]
            == "coupled"
        ]

        [
            keys
            + ["external_sha256"]
        ]

        .rename(
            columns={
                "external_sha256":
                    "hash_coupled"
            }
        )

    )


    M = A.merge(

        B,

        on=keys,

        how="outer",

        indicator=True,

    )


    both = M[
        M["_merge"] == "both"
    ]


    missing = M[
        M["_merge"] != "both"
    ]


    mismatch = both[

        both["hash_default"]
        !=
        both["hash_coupled"]

    ]


    expected = (

        len(CHECKPOINTS)

        * len(SEEDS)

        * N_EPISODES

    )


    ok = (

        len(both) == expected

        and len(missing) == 0

        and len(mismatch) == 0

    )


    lines = [

        "MATCHED INPUT VALIDATION",

        "=" * 70,

        f"Expected episode pairs : {expected}",

        f"Found episode pairs    : {len(both)}",

        f"Missing pairs          : {len(missing)}",

        f"External mismatches    : {len(mismatch)}",

        "",

        (
            "PASS: every default/coupled pair received "
            "identical external history."
            if ok
            else
            "FAIL: matched external histories are not identical."
        ),

    ]


    if (
        STRICT_MATCHED_INPUTS
        and not ok
    ):

        raise RuntimeError(
            "\n".join(lines)
        )


    return lines


# =====================================================================
# SEED AGGREGATION
# =====================================================================

def make_seed_table(
    episode_df,
):

    atom_cols = sorted([

        c

        for c in episode_df.columns

        if (
            c.startswith("atom_")
            and c.endswith("_g")
        )

    ])


    metrics = [

        "phi_r_g",

        "group_decoupling_g",

        "group_downward_g",

        "group_part_driven_g",

        "phi_r_atom_median_sum_g",

        "phi_r_minus_atom_median_sum_g",

        *atom_cols,

    ]


    if (
        COMPUTE_Z_CONTROL
        and "phi_r_z"
        in episode_df.columns
    ):

        metrics.append(
            "phi_r_z"
        )


    counts = (

        episode_df

        .groupby(
            [
                "condition",
                "checkpoint",
                "seed",
            ]
        )

        .size()

    )


    bad = counts[
        counts != N_EPISODES
    ]


    if len(bad):

        raise RuntimeError(
            f"Some seeds do not have {N_EPISODES} episodes:\n{bad}"
        )


    seed_df = (

        episode_df

        .groupby(

            [
                "condition",
                "checkpoint",
                "seed",
            ],

            as_index=False,

        )

        [metrics]

        .median()

    )


    seed_df = seed_df.sort_values(

        [
            "condition",
            "checkpoint",
            "seed",
        ]

    ).reset_index(
        drop=True
    )


    seed_df.to_csv(

        SEED_CSV,

        index=False,

    )


    return (
        seed_df,
        metrics,
    )


# =====================================================================
# 12k BRANCHPOINT CHECK
# =====================================================================

def validate_branchpoint(
    seed_df,
):

    check_metrics = list(
        MAIN_METRICS
    )


    if (
        COMPUTE_Z_CONTROL
        and "phi_r_z"
        in seed_df.columns
    ):

        check_metrics.append(
            "phi_r_z"
        )


    A = seed_df[

        (
            seed_df["condition"]
            == "default"
        )

        &

        (
            seed_df["checkpoint"]
            == BASELINE_STEP
        )

    ].set_index(
        "seed"
    )


    B = seed_df[

        (
            seed_df["condition"]
            == "coupled"
        )

        &

        (
            seed_df["checkpoint"]
            == BASELINE_STEP
        )

    ].set_index(
        "seed"
    )


    lines = [

        "",

        "12k BRANCHPOINT VALIDATION",

        "=" * 70,

    ]


    all_ok = True


    for metric in check_metrics:

        d = (

            B.loc[
                list(SEEDS),
                metric,
            ].to_numpy(float)

            -

            A.loc[
                list(SEEDS),
                metric,
            ].to_numpy(float)

        )


        max_abs = float(
            np.max(
                np.abs(d)
            )
        )


        ok = np.allclose(

            d,

            0,

            atol=1e-12,

            rtol=0,

        )


        all_ok &= bool(ok)


        lines.append(

            f"{metric:30s} "
            f"max |difference| = "
            f"{max_abs:.6g} "
            f"{'PASS' if ok else 'FAIL'}"

        )


    if (
        STRICT_BRANCHPOINT
        and not all_ok
    ):

        raise RuntimeError(
            "\n".join(lines)
        )


    return lines


# =====================================================================
# ABSOLUTE CHECKPOINT SUMMARY
# =====================================================================

def make_checkpoint_summary(
    seed_df,
    metrics,
):

    rows = []


    for condition in CONDITIONS:

        for checkpoint in CHECKPOINTS:

            s = seed_df[

                (
                    seed_df["condition"]
                    == condition
                )

                &

                (
                    seed_df["checkpoint"]
                    == checkpoint
                )

            ]


            for metric in metrics:

                x = s[
                    metric
                ].to_numpy(float)


                med, lo, hi = (
                    bootstrap_median_ci(

                        x,

                        f"abs|{condition}|"
                        f"{checkpoint}|{metric}",

                    )
                )


                rows.append({

                    "condition":
                        condition,

                    "checkpoint":
                        checkpoint,

                    "metric":
                        metric,

                    "n_seeds":
                        len(x),

                    "median":
                        med,

                    "ci95_low":
                        lo,

                    "ci95_high":
                        hi,

                })


    out = pd.DataFrame(
        rows
    )


    out.to_csv(

        CHECKPOINT_SUMMARY_CSV,

        index=False,

    )


    return out


# =====================================================================
# PAIRED LONGITUDINAL EFFECTS
# =====================================================================

def make_paired_effects(
    seed_df,
    metrics,
):

    D = (

        seed_df[
            seed_df["condition"]
            == "default"
        ]

        .set_index(
            [
                "checkpoint",
                "seed",
            ]
        )

    )


    C = (

        seed_df[
            seed_df["condition"]
            == "coupled"
        ]

        .set_index(
            [
                "checkpoint",
                "seed",
            ]
        )

    )


    seed_rows = []
    summary_rows = []


    for metric in metrics:

        for checkpoint in CHECKPOINTS:


            default_now = np.asarray([

                D.loc[
                    (checkpoint, seed),
                    metric,
                ]

                for seed
                in SEEDS

            ], dtype=float)


            coupled_now = np.asarray([

                C.loc[
                    (checkpoint, seed),
                    metric,
                ]

                for seed
                in SEEDS

            ], dtype=float)


            default_base = np.asarray([

                D.loc[
                    (BASELINE_STEP, seed),
                    metric,
                ]

                for seed
                in SEEDS

            ], dtype=float)


            coupled_base = np.asarray([

                C.loc[
                    (BASELINE_STEP, seed),
                    metric,
                ]

                for seed
                in SEEDS

            ], dtype=float)


            raw_diff = (

                coupled_now
                - default_now

            )


            delta_default = (

                default_now
                - default_base

            )


            delta_coupled = (

                coupled_now
                - coupled_base

            )


            delta_delta = (

                delta_coupled
                - delta_default

            )


            for i, seed in enumerate(
                SEEDS
            ):

                seed_rows.append({

                    "checkpoint":
                        checkpoint,

                    "seed":
                        seed,

                    "metric":
                        metric,

                    "default":
                        default_now[i],

                    "coupled":
                        coupled_now[i],

                    "raw_diff_coupled_minus_default":
                        raw_diff[i],

                    "delta_default_from_12k":
                        delta_default[i],

                    "delta_coupled_from_12k":
                        delta_coupled[i],

                    "delta_delta":
                        delta_delta[i],

                })


            raw_m, raw_l, raw_h = (
                bootstrap_median_ci(

                    raw_diff,

                    f"raw|{checkpoint}|{metric}",

                )
            )


            dd_m, dd_l, dd_h = (
                bootstrap_median_ci(

                    delta_default,

                    f"dd|{checkpoint}|{metric}",

                )
            )


            dc_m, dc_l, dc_h = (
                bootstrap_median_ci(

                    delta_coupled,

                    f"dc|{checkpoint}|{metric}",

                )
            )


            d2_m, d2_l, d2_h = (
                bootstrap_median_ci(

                    delta_delta,

                    f"d2|{checkpoint}|{metric}",

                )
            )


            summary_rows.append({

                "checkpoint":
                    checkpoint,

                "metric":
                    metric,

                "n_paired_seeds":
                    len(SEEDS),

                "median_raw_diff":
                    raw_m,

                "raw_diff_ci95_low":
                    raw_l,

                "raw_diff_ci95_high":
                    raw_h,

                "raw_diff_wilcoxon_p":
                    safe_wilcoxon(
                        raw_diff
                    ),

                "median_delta_default":
                    dd_m,

                "delta_default_ci95_low":
                    dd_l,

                "delta_default_ci95_high":
                    dd_h,

                "median_delta_coupled":
                    dc_m,

                "delta_coupled_ci95_low":
                    dc_l,

                "delta_coupled_ci95_high":
                    dc_h,

                "median_delta_delta":
                    d2_m,

                "delta_delta_ci95_low":
                    d2_l,

                "delta_delta_ci95_high":
                    d2_h,

                "delta_delta_wilcoxon_p":
                    safe_wilcoxon(
                        delta_delta
                    ),

            })


    seed_out = pd.DataFrame(
        seed_rows
    )


    summary_out = pd.DataFrame(
        summary_rows
    )


    seed_out.to_csv(

        SEED_PAIRED_CSV,

        index=False,

    )


    summary_out.to_csv(

        PAIRED_EFFECTS_CSV,

        index=False,

    )


    return (
        seed_out,
        summary_out,
    )


# =====================================================================
# FIGURES
# =====================================================================

def savefig(
    fig,
    name,
):

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig.savefig(

        FIGURE_ROOT
        / f"{name}.png",

        dpi=220,

        bbox_inches="tight",

    )


    fig.savefig(

        FIGURE_ROOT
        / f"{name}.pdf",

        bbox_inches="tight",

    )


def make_figures(
    absolute,
    pair_seed,
    paired,
):

    if not MAKE_FIGURES:

        return


    x = np.asarray(
        CHECKPOINTS
    )


    # --------------------------------------------------------------
    # A. absolute trajectories
    # --------------------------------------------------------------

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(11, 8),

        sharex=True,

    )


    axes = axes.ravel()


    for ax, metric in zip(
        axes,
        MAIN_METRICS,
    ):

        for condition in CONDITIONS:

            s = absolute[

                (
                    absolute["condition"]
                    == condition
                )

                &

                (
                    absolute["metric"]
                    == metric
                )

            ].set_index(
                "checkpoint"
            ).loc[
                list(CHECKPOINTS)
            ]


            y = s[
                "median"
            ].to_numpy(float)


            lo = s[
                "ci95_low"
            ].to_numpy(float)


            hi = s[
                "ci95_high"
            ].to_numpy(float)


            line = ax.plot(

                x,
                y,

                marker="o",

                label=condition,

            )[0]


            ax.fill_between(

                x,
                lo,
                hi,

                alpha=0.18,

                color=line.get_color(),

            )


        ax.set_title(
            METRIC_LABELS[metric]
        )


        ax.axvline(

            BASELINE_STEP,

            linestyle="--",

            linewidth=1,

            alpha=0.6,

        )


        ax.grid(
            alpha=0.2
        )


        ax.set_xlabel(
            "Training step"
        )


    axes[0].legend(
        frameon=False
    )


    fig.suptitle(
        "Matched replay: default vs coupled"
    )


    fig.tight_layout()


    savefig(
        fig,
        "fig_A_longitudinal_absolute",
    )


    # --------------------------------------------------------------
    # B. condition-specific Delta from 12k
    # --------------------------------------------------------------

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(11, 8),

        sharex=True,

    )


    axes = axes.ravel()


    for ax, metric in zip(
        axes,
        MAIN_METRICS,
    ):

        s = paired[

            paired["metric"]
            == metric

        ].set_index(
            "checkpoint"
        ).loc[
            list(CHECKPOINTS)
        ]


        for condition in CONDITIONS:

            if condition == "default":

                ycol = (
                    "median_delta_default"
                )

                lcol = (
                    "delta_default_ci95_low"
                )

                hcol = (
                    "delta_default_ci95_high"
                )

            else:

                ycol = (
                    "median_delta_coupled"
                )

                lcol = (
                    "delta_coupled_ci95_low"
                )

                hcol = (
                    "delta_coupled_ci95_high"
                )


            line = ax.plot(

                x,

                s[ycol],

                marker="o",

                label=condition,

            )[0]


            ax.fill_between(

                x,

                s[lcol],

                s[hcol],

                alpha=0.18,

                color=line.get_color(),

            )


        ax.axhline(
            0,
            linewidth=1,
            alpha=0.6,
        )


        ax.set_title(
            METRIC_LABELS[metric]
        )


        ax.set_xlabel(
            "Training step"
        )


        ax.set_ylabel(
            "Delta from 12k"
        )


        ax.grid(
            alpha=0.2
        )


    axes[0].legend(
        frameon=False
    )


    fig.suptitle(
        "Change from shared 12k branch point"
    )


    fig.tight_layout()


    savefig(
        fig,
        "fig_B_change_from_12k",
    )


    # --------------------------------------------------------------
    # C. DeltaDelta
    # --------------------------------------------------------------

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(11, 8),

        sharex=True,

    )


    axes = axes.ravel()


    for ax, metric in zip(
        axes,
        MAIN_METRICS,
    ):

        s = paired[

            paired["metric"]
            == metric

        ].set_index(
            "checkpoint"
        ).loc[
            list(CHECKPOINTS)
        ]


        line = ax.plot(

            x,

            s[
                "median_delta_delta"
            ],

            marker="o",

        )[0]


        ax.fill_between(

            x,

            s[
                "delta_delta_ci95_low"
            ],

            s[
                "delta_delta_ci95_high"
            ],

            alpha=0.18,

            color=line.get_color(),

        )


        ax.axhline(
            0,
            linewidth=1,
            alpha=0.6,
        )


        ax.set_title(
            METRIC_LABELS[metric]
        )


        ax.set_xlabel(
            "Training step"
        )


        ax.set_ylabel(
            "DeltaDelta"
        )


        ax.grid(
            alpha=0.2
        )


    fig.suptitle(
        "Effect of policy-gradient coupling"
    )


    fig.tight_layout()


    savefig(
        fig,
        "fig_C_delta_delta",
    )


    # --------------------------------------------------------------
    # D. seed-wise 48k DeltaDelta
    # --------------------------------------------------------------

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(11, 8),

    )


    axes = axes.ravel()


    jitter_rng = np.random.default_rng(
        2033
    )


    for ax, metric in zip(
        axes,
        MAIN_METRICS,
    ):

        vals = pair_seed[

            (
                pair_seed["metric"]
                == metric
            )

            &

            (
                pair_seed["checkpoint"]
                == FINAL_STEP
            )

        ][
            "delta_delta"
        ].to_numpy(float)


        s = paired[

            (
                paired["metric"]
                == metric
            )

            &

            (
                paired["checkpoint"]
                == FINAL_STEP
            )

        ].iloc[0]


        jitter = jitter_rng.normal(

            0,
            0.035,

            size=len(vals),

        )


        ax.scatter(

            jitter,
            vals,

            alpha=0.75,

        )


        ax.axhline(

            0,

            linewidth=1,

            alpha=0.6,

        )


        med = float(
            s["median_delta_delta"]
        )


        lo = float(
            s["delta_delta_ci95_low"]
        )


        hi = float(
            s["delta_delta_ci95_high"]
        )


        ax.plot(

            [-0.18, 0.18],

            [med, med],

            linewidth=2.5,

        )


        ax.vlines(

            0,

            lo,
            hi,

            linewidth=2,

        )


        ax.set_xlim(
            -0.35,
            0.35,
        )


        ax.set_xticks([])


        ax.set_title(
            METRIC_LABELS[metric]
        )


        ax.set_ylabel(
            "Seed-wise DeltaDelta at 48k"
        )


        ax.grid(
            axis="y",
            alpha=0.2,
        )


    fig.suptitle(
        "48k paired effects, n=30 seeds"
    )


    fig.tight_layout()


    savefig(
        fig,
        "fig_D_48k_paired_delta_delta",
    )


    if SHOW_FIGURES:

        plt.show()

    else:

        plt.close("all")


# =====================================================================
# MANIFEST
# =====================================================================

def write_manifest(
    episode_df,
    seed_df,
):

    data = {

        "conditions":
            list(CONDITIONS),

        "checkpoints":
            list(CHECKPOINTS),

        "seeds":
            list(SEEDS),

        "episodes_per_seed":
            N_EPISODES,

        "steps_per_episode":
            N_STEPS,

        "baseline_step":
            BASELINE_STEP,

        "compute_z_control":
            COMPUTE_Z_CONTROL,

        "bootstrap_n":
            N_BOOTSTRAP,

        "bootstrap_unit":
            "training seed",

        "bootstrap_ci":
            "percentile 95%",

        "n_episode_rows":
            len(episode_df),

        "n_seed_rows":
            len(seed_df),

        "phi_r_aggregation":
            "temporal median of local sum of 9 Phi_r atoms",

        "atom_group_aggregation":
            "sum of per-atom temporal medians, matching phi_atoms.py",

    }


    MANIFEST_JSON.write_text(

        json.dumps(
            data,
            indent=2,
        ),

        encoding="utf-8",

    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    t0 = time.time()


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 78)
    print("CEAR STOP-GRADIENT MATCHED-REPLAY ANALYSIS")
    print("=" * 78)

    print(
        "Repo :",
        REPO_ROOT,
    )

    print(
        "Input:",
        INPUT_ROOT,
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )

    print(
        "Phi_r(z) control:",
        COMPUTE_Z_CONTROL,
    )

    print()


    # --------------------------------------------------------------
    # source scan
    # --------------------------------------------------------------

    paths = scan_inputs()


    # --------------------------------------------------------------
    # episode metrics
    # --------------------------------------------------------------

    episode_df = (
        compute_episode_table(
            paths
        )
    )


    # --------------------------------------------------------------
    # matched-exposure validation
    # --------------------------------------------------------------

    validation_lines = (
        validate_matched_inputs(
            episode_df
        )
    )


    # --------------------------------------------------------------
    # seed aggregation
    # --------------------------------------------------------------

    seed_df, all_metrics = (
        make_seed_table(
            episode_df
        )
    )


    # --------------------------------------------------------------
    # common 12k branch point
    # --------------------------------------------------------------

    branch_lines = (
        validate_branchpoint(
            seed_df
        )
    )


    validation_text = "\n".join(

        validation_lines
        + branch_lines

    )


    VALIDATION_TXT.write_text(

        validation_text
        + "\n",

        encoding="utf-8",

    )


    print()
    print(
        validation_text
    )


    # --------------------------------------------------------------
    # paper-facing metrics
    # --------------------------------------------------------------

    analysis_metrics = list(
        MAIN_METRICS
    )


    if COMPUTE_Z_CONTROL:

        analysis_metrics.append(
            "phi_r_z"
        )


    absolute = (
        make_checkpoint_summary(

            seed_df,

            analysis_metrics,

        )
    )


    pair_seed, paired = (
        make_paired_effects(

            seed_df,

            analysis_metrics,

        )
    )


    write_manifest(
        episode_df,
        seed_df,
    )


    make_figures(

        absolute,

        pair_seed,

        paired,

    )


    # --------------------------------------------------------------
    # print final 48k result
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "48k DeltaDelta = Delta_coupled - Delta_default"
    )
    print("=" * 78)


    for metric in MAIN_METRICS:

        r = paired[

            (
                paired["metric"]
                == metric
            )

            &

            (
                paired["checkpoint"]
                == FINAL_STEP
            )

        ].iloc[0]


        print(

            f"{metric:28s} "

            f"{r['median_delta_delta']:+.6f} "

            f"["
            f"{r['delta_delta_ci95_low']:+.6f}, "
            f"{r['delta_delta_ci95_high']:+.6f}"
            f"] "

            f"p={r['delta_delta_wilcoxon_p']:.4g}"

        )


    if (
        COMPUTE_Z_CONTROL
        and "phi_r_z"
        in paired["metric"].values
    ):

        r = paired[

            (
                paired["metric"]
                == "phi_r_z"
            )

            &

            (
                paired["checkpoint"]
                == FINAL_STEP
            )

        ].iloc[0]


        print()

        print(

            f"{'phi_r_z control':28s} "

            f"{r['median_delta_delta']:+.6f} "

            f"["
            f"{r['delta_delta_ci95_low']:+.6f}, "
            f"{r['delta_delta_ci95_high']:+.6f}"
            f"]"

        )


    dt = time.time() - t0


    print()
    print("=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        f"Elapsed: {dt / 60:.2f} min"
    )

    print(
        "Episode CSV:",
        EPISODE_CSV,
    )

    print(
        "Seed CSV:",
        SEED_CSV,
    )

    print(
        "Paired effects:",
        PAIRED_EFFECTS_CSV,
    )

    print(
        "Validation:",
        VALIDATION_TXT,
    )

    print(
        "Figures:",
        FIGURE_ROOT,
    )

    print()


if __name__ == "__main__":

    main()