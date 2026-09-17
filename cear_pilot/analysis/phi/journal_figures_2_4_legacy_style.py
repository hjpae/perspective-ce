# -*- coding: utf-8 -*-
"""
CEAR P1 journal rebuild — Figure cleanup pass in the style of the older paper figures.

This script reuses the matched-probe analysis backend from:
    cear_pilot.analysis.phi.journal_figures_2_4_matched
and redraws Figures 2, 3, and 4 using the older visual language:
    - scatter-left + half-violin-right rainclouds
    - compact typography
    - navy / orange / neutral palette
    - median + bootstrap 95% CI markers

Designed to be run directly in Spyder.
No CLI arguments required.

Recommended placement:
    cear_pilot/analysis/phi/journal_figures_2_4_legacy_style.py

Outputs:
    outputs/journal_figures_2_4_legacy_style/
        figures/
            figure2_temporal_substrate_legacy.png/.pdf
            figure3_learning_reorganization_legacy.png/.pdf
            figure4_learning_vs_coupling_legacy.png/.pdf
        fig2_stats.csv
        fig3_stats.csv
        fig4_stats.csv
        manifest.json
"""

from __future__ import annotations

import json
import runpy
import sys
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Repo location / imports
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3] if len(THIS_FILE.parents) >= 4 else Path.cwd().resolve()

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

# Load the analysis backend as a standalone script namespace rather than importing
# it as a package module.  The backend deliberately clears cear_pilot modules
# for Spyder/UMR safety when it starts; importing it normally from another
# cear_pilot module can therefore invalidate its own import and trigger a
# KeyError.  runpy avoids that circular package-loader interaction.
BACKEND_PATH = REPO_ROOT / "cear_pilot" / "analysis" / "phi" / "journal_figures_2_4_matched.py"
if not BACKEND_PATH.exists():
    raise FileNotFoundError(
        f"Required backend not found: {BACKEND_PATH}\n"
        "Place journal_figures_2_4_matched.py in cear_pilot/analysis/phi/."
    )
_backend_ns = runpy.run_path(str(BACKEND_PATH), run_name="__cear_fig_backend__")
backend = types.SimpleNamespace(**_backend_ns)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SHOW_FIGURES = True
SAVE_PDF = True
DPI = 260

OUT_ROOT = REPO_ROOT / "outputs" / "journal_figures_2_4_legacy_style"
FIG_ROOT = OUT_ROOT / "figures"
FIG2_STATS_CSV = OUT_ROOT / "fig2_stats.csv"
FIG3_STATS_CSV = OUT_ROOT / "fig3_stats.csv"
FIG4_STATS_CSV = OUT_ROOT / "fig4_stats.csv"
MANIFEST_JSON = OUT_ROOT / "manifest.json"

# -----------------------------------------------------------------------------
# Visual style (old-paper-like)
# -----------------------------------------------------------------------------

COLOR_TRAINED = "#1f4e79"      # navy
COLOR_UNTRAINED = "#d65f3e"    # orange / brick
COLOR_COUPLED = "#b64e5a"      # muted crimson
COLOR_NEUTRAL = "#444444"      # dark neutral
COLOR_LIGHT = "#c9c9c9"        # light gray
COLOR_SHUFFLE = "#6f8f5f"      # muted green
COLOR_AR1 = "#b74d4d"          # muted red for null
PAIR_LINE = "#bfbfbf"
GRID_COLOR = "#d9d9d9"


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


# -----------------------------------------------------------------------------
# Helper plotting functions
# -----------------------------------------------------------------------------

def median_ci(values, tag):
    return backend.bootstrap_median_ci(values, tag)


def raincloud(ax, data, position, color, width=0.72, alpha_v=0.28,
              alpha_s=0.70, scatter_size=28, scatter_offset=0.10,
              violin_offset=0.015, rng_seed=0):
    """
    Vertical raincloud at one x-position:
        - scatter on the left
        - half-violin on the right
        - median + 95% bootstrap CI drawn near center
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return

    rng = np.random.default_rng(rng_seed)

    # scatter left
    jit = rng.normal(0, 0.035, size=len(data))
    ax.scatter(
        position - scatter_offset + jit,
        data,
        s=scatter_size,
        color=color,
        alpha=alpha_s,
        zorder=3,
        edgecolor="none",
    )

    # half violin right
    if len(data) > 2 and float(np.std(data)) > 1e-12:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        ymin, ymax = float(np.min(data)), float(np.max(data))
        pad = (ymax - ymin) * 0.06 + 1e-6
        ys = np.linspace(ymin - pad, ymax + pad, 240)
        dens = kde(ys)
        if float(np.max(dens)) > 0:
            dens = dens / np.max(dens) * (width / 2.0)
            x_right = position + violin_offset + dens
            ax.fill_betweenx(
                ys,
                position + violin_offset,
                x_right,
                color=color,
                alpha=alpha_v,
                linewidth=0,
                zorder=2,
            )
            ax.plot(x_right, ys, color=color, lw=0.8, alpha=0.95, zorder=4)
            ax.plot(
                [position + violin_offset, position + violin_offset],
                [ymin - pad, ymax + pad],
                color=color,
                lw=0.7,
                alpha=0.65,
                zorder=4,
            )


def draw_median_ci(ax, values, position, color, tag):
    med, lo, hi = median_ci(values, tag)
    xx = position + 0.01
    ax.vlines(xx, lo, hi, color=color, lw=2.2, zorder=5)
    ax.hlines(med, xx - 0.07, xx + 0.07, color=color, lw=3.0, zorder=6)
    return med, lo, hi


def paired_absolute_panel(ax, a, b, labels, title, tag,
                          colors=(COLOR_UNTRAINED, COLOR_TRAINED),
                          ylabel=None, rotate_xticks=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, x2 = 0.0, 1.0

    for i in range(len(a)):
        ax.plot([x1 + 0.08, x2 + 0.08], [a[i], b[i]],
                color=PAIR_LINE, lw=0.9, alpha=0.45, zorder=1)

    raincloud(ax, a, x1, colors[0], rng_seed=backend.stable_seed("rain", tag, "a"))
    raincloud(ax, b, x2, colors[1], rng_seed=backend.stable_seed("rain", tag, "b"))
    draw_median_ci(ax, a, x1, colors[0], f"{tag}|a")
    draw_median_ci(ax, b, x2, colors[1], f"{tag}|b")

    ax.set_xticks([x1, x2], labels)
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=rotate_xticks)
    ax.set_xlim(-0.52, 1.58)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", alpha=0.45, color=GRID_COLOR, linewidth=0.8)


def effect_panel(ax, groups, labels, title, tag, colors,
                 ylabel=None, rotate_xticks=0):
    xs = np.arange(len(groups), dtype=float)

    for j, (x, vals, lab, color) in enumerate(zip(xs, groups, labels, colors)):
        vals = np.asarray(vals, dtype=float)
        raincloud(ax, vals, x, color, rng_seed=backend.stable_seed("effect", tag, j))
        draw_median_ci(ax, vals, x, color, f"{tag}|{lab}")

    ax.axhline(0.0, color=COLOR_NEUTRAL, lw=1.0, alpha=0.7, zorder=0)
    ax.set_xticks(xs, labels)
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=rotate_xticks)
    ax.set_xlim(-0.52, len(groups) - 0.48)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", alpha=0.45, color=GRID_COLOR, linewidth=0.8)


def save_figure(fig, stem):
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIG_ROOT / f"{stem}.png"
    fig.savefig(png)
    if SAVE_PDF:
        fig.savefig(FIG_ROOT / f"{stem}.pdf")
    print(f"Saved: {png}")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def make_figure2(ctx):
    sd = ctx["sd"]
    sa = ctx["sa"]

    g = sd["phi_r_g"].to_numpy(float)
    z = sd["phi_r_z"].to_numpy(float)
    sh = sd["phi_r_g_shuffled"].to_numpy(float)
    ar = sa["phi_r_g_ar1"].to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))

    paired_absolute_panel(
        axes[0], g, z,
        [r"$g$", r"$z$"],
        r"A. Architectural localization",
        "fig2A",
        colors=(COLOR_TRAINED, COLOR_NEUTRAL),
        ylabel=r"Seed-level $\Phi_r$",
    )

    paired_absolute_panel(
        axes[1], g, sh,
        ["Intact", "Time-shuffled"],
        r"B. Temporal-order dependence",
        "fig2B",
        colors=(COLOR_TRAINED, COLOR_SHUFFLE),
    )

    paired_absolute_panel(
        axes[2], g, ar,
        ["Intact", "AR(1) null"],
        r"C. Beyond first-order persistence",
        "fig2C",
        colors=(COLOR_TRAINED, COLOR_AR1),
    )

    fig.suptitle(
        r"Figure 2. The slow latent provides the relevant temporal substrate",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure2_temporal_substrate_legacy")
    return fig


def make_figure3(ctx):
    s0 = ctx["s0"]
    sd = ctx["sd"]
    le = ctx["learning"]

    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.35))

    paired_absolute_panel(
        axes[0],
        s0["phi_r_g"].to_numpy(float),
        sd["phi_r_g"].to_numpy(float),
        ["Step 0", "48k"],
        r"A. Scalar $\Phi_r(g)$",
        "fig3A",
        colors=(COLOR_UNTRAINED, COLOR_TRAINED),
        ylabel=r"Seed-level $\Phi_r(g)$",
    )

    effect_panel(
        axes[1],
        [
            le["group_decoupling_g"],
            le["group_downward_g"],
            le["group_part_driven_g"],
        ],
        ["Decoupling", "Whole-to-part", "Part-driven"],
        r"B. Learning changes $\Phi$ID composition",
        "fig3B",
        colors=(COLOR_TRAINED, COLOR_UNTRAINED, COLOR_NEUTRAL),
        ylabel="48k default - step0 default",
        rotate_xticks=16,
    )

    effect_panel(
        axes[2],
        [
            le["whole_to_single_sum"],
            le["whole_to_parts"],
            le["downward_reconstructed"],
        ],
        ["Whole→single\nsum", "Whole→parts", "Total\nwhole→part"],
        "C. Whole-to-part decomposition",
        "fig3C",
        colors=(COLOR_UNTRAINED, "#a67c73", "#d681c3"),
    )

    fig.suptitle(
        "Figure 3. Learning reorganizes the informational structure of the slow latent",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure3_learning_reorganization_legacy")
    return fig


def make_figure4(ctx):
    le = ctx["learning"]
    ce = ctx["coupling"]

    metric_order = [
        "phi_r_g",
        "group_decoupling_g",
        "group_downward_g",
        "group_part_driven_g",
    ]
    metric_titles = {
        "phi_r_g": r"$\Phi_r(g)$",
        "group_decoupling_g": "Decoupling",
        "group_downward_g": "Whole-to-part",
        "group_part_driven_g": "Part-driven",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.9))
    axes = axes.ravel()

    for ax, metric in zip(axes, metric_order):
        effect_panel(
            ax,
            [le[metric], ce[metric]],
            ["Learning\n48kD - 0kD", "Coupling\n48kC - 48kD"],
            metric_titles[metric],
            f"fig4|{metric}",
            colors=(COLOR_TRAINED, COLOR_COUPLED),
            ylabel="Paired seed effect",
        )

    fig.suptitle(
        "Figure 4. Paired effects of learning and policy-side gradient coupling",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "figure4_learning_vs_coupling_legacy")
    return fig


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------

def write_manifest():
    payload = {
        "style_basis": "legacy paper figure style (scatter-left + half-violin-right raincloud)",
        "summary_marker": "median + percentile-bootstrap 95% CI over training seeds",
        "palette": {
            "trained": COLOR_TRAINED,
            "untrained": COLOR_UNTRAINED,
            "coupled": COLOR_COUPLED,
            "neutral": COLOR_NEUTRAL,
            "shuffle": COLOR_SHUFFLE,
            "ar1": COLOR_AR1,
        },
        "backend": "cear_pilot.analysis.phi.journal_figures_2_4_matched",
        "backend_window": "full 800-step matched serpentine trajectory; no cutoff",
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print("\n" + "=" * 84)
    print("CEAR P1 JOURNAL — LEGACY-STYLE FIGURE REDRAW (FIGS 2–4)")
    print("=" * 84)
    print("Repo   :", REPO_ROOT)
    print("Output :", OUT_ROOT)

    # Reuse backend computation / caches.
    ep = backend.load_or_compute_unified_episode_table()
    seed_df = backend.make_seed_table(ep)
    _, _, ar1_seed = backend.compute_ar1_surrogates()
    ctx = backend.prepare_effects(seed_df, ar1_seed)

    # Copy stats tables for convenience.
    ctx["fig2_stats"].to_csv(FIG2_STATS_CSV, index=False)
    ctx["fig3_stats"].to_csv(FIG3_STATS_CSV, index=False)
    ctx["fig4_stats"].to_csv(FIG4_STATS_CSV, index=False)
    write_manifest()

    fig2 = make_figure2(ctx)
    fig3 = make_figure3(ctx)
    fig4 = make_figure4(ctx)

    dt = time.time() - t0
    print("\n" + "=" * 84)
    print("DONE")
    print("=" * 84)
    print(f"Elapsed: {dt / 60:.2f} min")
    print("Figure 2:", FIG_ROOT / "figure2_temporal_substrate_legacy.png")
    print("Figure 3:", FIG_ROOT / "figure3_learning_reorganization_legacy.png")
    print("Figure 4:", FIG_ROOT / "figure4_learning_vs_coupling_legacy.png")
    print("Stats   :", FIG2_STATS_CSV)
    print("          ", FIG3_STATS_CSV)
    print("          ", FIG4_STATS_CSV)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)


if __name__ == "__main__":
    main()
