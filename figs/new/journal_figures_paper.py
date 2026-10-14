# -*- coding: utf-8 -*-
"""
journal_figures_paper.py
------------------------
Result figures of the ALife paper (Figs. 3-5) in the visual language of the
earlier figures.py (v2): scatter-left + half-violin-right rainclouds, navy /
orange / neutral palette, a horizontal summary tick, a white stat box, lowercase
panel titles, no grid.

Changes relative to the earlier style, required by the present analysis:
  - points are training seeds (episode -> seed median), not episodes;
  - the summary tick is the seed MEDIAN, not the mean;
  - every number printed comes from the stats tables written by
    journal_figures_2_4_matched.py (paired median effect, seed-bootstrap
    95% CI, seeds in the direction of the effect, Wilcoxon p), so the
    figures and the text quote identical values. Nothing is recomputed.

Inputs (written by cear_pilot/analysis/phi/journal_figures_2_4_matched.py):
    outputs/journal_figures_2_4/unified_seed_metrics.csv
    outputs/journal_figures_2_4/ar1_seed_summary.csv
    outputs/journal_figures_2_4/fig2_stats.csv   (-> paper Fig. 3)
    outputs/journal_figures_2_4/fig3_stats.csv   (-> paper Fig. 4)
    outputs/journal_figures_2_4/fig4_stats.csv   (-> paper Fig. 5)
Outputs:
    outputs/journal_figures_paper/fig3_temporal_substrate.{pdf,png}
    outputs/journal_figures_paper/fig4_learning_reorganization.{pdf,png}
    outputs/journal_figures_paper/fig5_learning_vs_coupling.{pdf,png}

Run cells top-to-bottom in Spyder (Ctrl+Enter) or run the whole file.
"""

# %% [SETUP] -------------------------------------------------------------------
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

COLOR_TRAINED   = "#1f4e79"   # default agents, step 48,000
COLOR_UNTRAINED = "#d65f3e"   # default agents, step 0
COLOR_COUPLED   = "#2b9e9e"   # routing effect (coupled - default); CVD-checked against navy/orange
COLOR_NEUTRAL   = "#444444"   # controls: z, temporal shuffle, AR(1)
COLOR_FAINT     = "#cccccc"

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

try:
    ROOT = Path(__file__).resolve().parents[3]          # .../cear_pilot/analysis/phi/<this file>
except NameError:                                        # Spyder cell execution
    ROOT = Path.cwd()
DATA = ROOT / "outputs" / "journal_figures_2_4"
FIGS = ROOT / "outputs" / "journal_figures_paper"
FIGS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    print(f"  saved -> {FIGS / name}.{{pdf,png}}")


seed = pd.read_csv(DATA / "unified_seed_metrics.csv")
ar1 = pd.read_csv(DATA / "ar1_seed_summary.csv")
st_temporal = pd.read_csv(DATA / "fig2_stats.csv").set_index("contrast")
st_learning = pd.read_csv(DATA / "fig3_stats.csv").set_index("metric")
st_routing = pd.read_csv(DATA / "fig4_stats.csv")


def state(condition, checkpoint):
    df = seed[(seed.condition == condition) & (seed.checkpoint == checkpoint)]
    return df.sort_values("seed").set_index("seed")


D0, D48, C48 = state("default", 0), state("default", 48000), state("coupled", 48000)
assert (D0.index == D48.index).all() and (C48.index == D48.index).all()
AR1 = ar1.sort_values("seed").set_index("seed").loc[D48.index, "phi_r_g_ar1"]


# %% [HELPERS] -----------------------------------------------------------------
# Half-violin display settings. With 30 seeds, Scott's rule under-smooths, and a
# single isolated seed makes the density drop to ~0 between it and the bulk, so the
# violin pinches to a hairline and then reappears as a small blob ("broken" look).
# The violin is therefore drawn as a display-smoothed density over ALL seeds:
VIOLIN_BW = 1.5     # bandwidth multiplier on Scott's rule
VIOLIN_CUT = 0.5    # violin ends this many bandwidths beyond the extreme seeds, closing to a point
VIOLIN_TAIL = 0.18        # relative width at which a tail leaves the body (0 = no tail shaping)
VIOLIN_TAIL_TIP = 0.05    # minimum relative width the tail keeps at the end seed
VIOLIN_TAIL_SHAPE = 1.2   # taper exponent from body to end seed (1 = linear; larger = thins faster)


def violin_profile(data, n=400):
    """Relative half-violin width (max 1) on a y grid; display smoothing only."""
    kde = gaussian_kde(data)
    kde.set_bandwidth(kde.factor * VIOLIN_BW)
    h = float(np.sqrt(kde.covariance[0, 0]))
    ys = np.linspace(data.min() - VIOLIN_CUT * h, data.max() + VIOLIN_CUT * h, n)
    d = kde(ys)
    d = d / d.max()
    # fill dips: non-increasing away from the mode, so isolated seeds stay attached
    i = int(np.argmax(d))
    d[i:] = np.maximum.accumulate(d[i:][::-1])[::-1]
    d[:i + 1] = np.maximum.accumulate(d[:i + 1])
    # thin tails leave the body at VIOLIN_TAIL and taper down to the end seed
    if VIOLIN_TAIL > 0:
        body = np.where(d >= VIOLIN_TAIL)[0]
        k_lo = int(np.searchsorted(ys, data.min()))
        k_hi = int(np.searchsorted(ys, data.max())) - 1
        for edge, end in ((body.max(), k_hi), (body.min(), k_lo)):
            if abs(end - edge) < 2:
                continue
            idx = np.arange(min(edge, end), max(edge, end) + 1)
            s = np.abs(idx - edge) / abs(end - edge)          # 0 at the body edge, 1 at the end seed
            tip = max(d[end], VIOLIN_TAIL_TIP)
            d[idx] = np.maximum(d[idx], VIOLIN_TAIL * (1 - s) ** VIOLIN_TAIL_SHAPE + tip * s)
        d = gaussian_filter1d(d, 2)
    # close both ends to a rounded point just beyond the extreme seeds
    lo_ramp = np.clip((ys - ys[0]) / max(data.min() - ys[0], 1e-12), 0, 1)
    hi_ramp = np.clip((ys[-1] - ys) / max(ys[-1] - data.max(), 1e-12), 0, 1)
    d = d * np.sin(0.5 * np.pi * np.minimum(lo_ramp, hi_ramp)) ** 0.6
    return ys, d / d.max()


def raincloud(ax, data, position, color, width=0.7, alpha_v=0.7,
              alpha_s=0.55, scatter_size=14, scatter_offset=0.10,
              violin_offset=0.02, rng_seed=0):
    """Vertical raincloud: jittered seeds (left), half-violin (right), median tick."""
    data = np.asarray(data, float)
    data = data[~np.isnan(data)]
    rng_local = np.random.default_rng(rng_seed)

    jit = rng_local.normal(0, 0.04, size=len(data))
    ax.scatter(position - scatter_offset + jit, data, s=scatter_size, color=color,
               alpha=alpha_s, zorder=2, edgecolor="none")

    if data.std() > 1e-9 and len(data) > 2:
        ys, rel = violin_profile(data)
        ax._violin_ext = getattr(ax, "_violin_ext", []) + [(ys[0], ys[-1])]
        dens = rel * (width / 2.0)
        x0 = position + violin_offset
        ax.fill_betweenx(ys, x0, x0 + dens, color=color, alpha=alpha_v, linewidth=0, zorder=3)
        ax.plot(x0 + dens, ys, color=color, lw=0.6, alpha=0.9, zorder=4)
        ax.plot([x0, x0], [ys[0], ys[-1]], color=color, lw=0.6, alpha=0.6, zorder=4)

    ax.hlines(np.median(data), position - scatter_offset - 0.07,
              position + violin_offset + (width / 2.0) + 0.02,
              colors=COLOR_NEUTRAL, lw=1.3, zorder=5)


def num(v, signed=True):
    """Number as printed in the text: 3 decimals below 0.1 in magnitude, true minus sign."""
    d = 3 if abs(v) < 0.1 else 2
    out = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return out.replace("-", "\u2212")


def fmt_p(p):
    if p < 1e-4:
        return "p < 0.0001"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def n_dir(row):
    """Seeds whose paired effect has the sign of the median effect."""
    k = row["n_positive"] if row["median_effect"] > 0 else row["n_negative"]
    return f"{int(k)}/{int(row['n_seeds'])} seeds"


def stat_box(ax, label, row):
    text = (f"{label}: {num(row['median_effect'])} "
            f"[{num(row['ci95_low'])}, {num(row['ci95_high'])}]\n"
            f"{n_dir(row)}, {fmt_p(row['wilcoxon_p'])}")
    ax.text(0.5, 0.97, text, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_NEUTRAL, lw=0.7), zorder=10)


def effect_label(ax, x, y, row, color):
    ax.text(x, y, f"{num(row['median_effect'])}\n{fmt_p(row['wilcoxon_p'])}", ha="center",
            va="bottom", fontsize=8, color=color, zorder=10, linespacing=1.15)


def headroom(ax, values, frac_top, frac_bottom=0.06):
    """Set y-limits with room above the data (and violins); return the y of a label row above them."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    ext = getattr(ax, "_violin_ext", [])
    if ext:
        lo, hi = min(lo, min(e[0] for e in ext)), max(hi, max(e[1] for e in ext))
    span = hi - lo
    ax.set_ylim(lo - frac_bottom * span, hi + frac_top * span)
    return hi + 0.05 * span


# %% [FIG 3] Temporal substrate -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
panels = [
    ("(A) architectural localization", D48["phi_r_g"], D48["phi_r_z"],
     [r"$\Phi_r(g)$", r"$\Phi_r(z)$"], COLOR_NEUTRAL, r"$\Phi_r$",
     "48k_default_phi_g_minus_phi_z", "g − z"),
    ("(B) temporal-shuffle control", D48["phi_r_g"], D48["phi_r_g_shuffled"],
     ["intact", "shuffled"], COLOR_NEUTRAL, r"$\Phi_r(g)$",
     "48k_default_intact_minus_temporal_shuffle", "intact − shuffled"),
    ("(C) independent AR(1) null", D48["phi_r_g"], AR1,
     ["intact", "AR(1) null"], COLOR_NEUTRAL, r"$\Phi_r(g)$",
     "48k_default_intact_minus_ar1", "intact − AR(1)"),
]
for i, (title, a, b, ticks, col_b, ylab, key, lab) in enumerate(panels):
    ax = axes[i]
    raincloud(ax, a, 0, COLOR_TRAINED, rng_seed=1 + 10 * i)
    raincloud(ax, b, 1, col_b, rng_seed=2 + 10 * i)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ticks)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10, pad=8)
    headroom(ax, np.r_[a, b], 0.30)
    stat_box(ax, lab, st_temporal.loc[key])
plt.tight_layout()
save(fig, "fig3_temporal_substrate")
plt.show()


# %% [FIG 4] Learning reorganization --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25]))

# (A) scalar Phi_r(g): untrained vs trained
ax = axes[0]
raincloud(ax, D0["phi_r_g"], 0, COLOR_UNTRAINED, rng_seed=10)
raincloud(ax, D48["phi_r_g"], 1, COLOR_TRAINED, rng_seed=11)
ax.set_xticks([0, 1])
ax.set_xticklabels(["untrained\n(step 0)", "trained\n(48k)"])
ax.set_xlim(-0.5, 1.6)
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("(A) scalar $\\Phi_r(g)$", fontsize=10, pad=8)
headroom(ax, np.r_[D0["phi_r_g"], D48["phi_r_g"]], 0.30)
stat_box(ax, "48k − 0k", st_learning.loc["phi_r_g"])


def effect_panel(ax, metrics, labels, title, rng0):
    vals = []
    for j, m in enumerate(metrics):
        eff = (D48[m] - D0[m]).to_numpy()
        vals.append(eff)
        raincloud(ax, eff, j, COLOR_TRAINED, rng_seed=rng0 + j)
    y_lab = headroom(ax, np.concatenate(vals), 0.30)
    for j, m in enumerate(metrics):
        effect_label(ax, j + 0.12, y_lab, st_learning.loc[m], COLOR_NEUTRAL)
    ax.axhline(0, color=COLOR_NEUTRAL, lw=0.6, linestyle="--", zorder=0)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(metrics) - 0.4)
    ax.set_title(title, fontsize=10, pad=8)


effect_panel(axes[1], ["group_decoupling_g", "group_downward_g", "group_part_driven_g"],
             ["decoupling", "whole-to-part", "part-driven"],
             "(B) learning effect (48k − 0k) by group", 20)
axes[1].set_ylabel("paired effect")
effect_panel(axes[2], ["whole_to_single_sum", "whole_to_parts", "group_downward_g"],
             ["whole→single\nsum", "whole→parts", "whole-to-part\ntotal"],
             "(C) whole-to-part decomposition", 30)
plt.tight_layout()
save(fig, "fig4_learning_reorganization")
plt.show()


# %% [FIG 5] Learning vs policy-side gradient coupling ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
metrics = [("phi_r_g", "(A) scalar $\\Phi_r(g)$"),
           ("group_decoupling_g", "(B) decoupling"),
           ("group_downward_g", "(C) whole-to-part"),
           ("group_part_driven_g", "(D) part-driven")]
for i, (m, title) in enumerate(metrics):
    ax = axes.flat[i]
    learn = (D48[m] - D0[m]).to_numpy()
    coup = (C48[m] - D48[m]).to_numpy()
    raincloud(ax, learn, 0, COLOR_TRAINED, rng_seed=40 + i)
    raincloud(ax, coup, 1, COLOR_COUPLED, rng_seed=50 + i)
    ax.axhline(0, color=COLOR_NEUTRAL, lw=0.6, linestyle="--", zorder=0)
    y_lab = headroom(ax, np.r_[learn, coup], 0.30)
    rows = st_routing[st_routing.metric == m].set_index("effect_type")
    effect_label(ax, 0.12, y_lab, rows.loc["learning_48kD_minus_0kD"], COLOR_TRAINED)
    effect_label(ax, 1.12, y_lab, rows.loc["coupled48k_minus_default48k"], COLOR_COUPLED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["learning\n48k$_D$ − 0k$_D$", "coupling\n48k$_C$ − 48k$_D$"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_title(title, fontsize=10, pad=8)
    if i % 2 == 0:
        ax.set_ylabel("paired effect")
plt.tight_layout()
save(fig, "fig5_learning_vs_coupling")
plt.show()

# %%
print("\nDone:", FIGS)
