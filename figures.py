# -*- coding: utf-8 -*-
"""
figures.py (v2)
---------------
Five paper figures, revised:

  Fig 1 (merged): (a) Φ_r(g) vs Φ_r(z) violins  +  (b) shuffle collapse
  Fig 2: magnitude — Φ_r(g) untrained vs trained, violins (untrained as points)
  Fig 3: atom-group decomposition (clean), bars
  Fig 4: switch-aligned hysteresis (y ∈ [0.25, 3.0], no comment box)
  Fig 5: atom-level regime sensitivity (single panel, trained vs untrained side-by-side per group)

Run cells top-to-bottom in Spyder (Ctrl+Enter).
"""

# %% [SETUP] -------------------------------------------------------------------
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COLOR_TRAINED   = "#1f4e79"
COLOR_UNTRAINED = "#d65f3e"
COLOR_NEUTRAL   = "#444444"
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
})

DATA = Path("outputs")
FIGS = Path("figs"); FIGS.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    print(f"  saved → figs/{name}.{{pdf,png}}")

print("Loading CSVs...")
dt_clean = pd.read_csv(DATA / "v2_decomp_trained.csv")
du_clean = pd.read_csv(DATA / "v2_decomp_untrained.csv")
prepost  = pd.read_csv(DATA / "v2_prepost.csv")
shuffle  = pd.read_csv(DATA / "v2_shuffle.csv")
hyst     = pd.read_csv(DATA / "rq4_hysteresis_binned.csv")
atoms    = pd.read_csv(DATA / "rq5_atoms_decomp.csv")
print("  loaded.")


# %% [HELPER] raincloud plot (vertical: scatter left, half-violin right) -------
def raincloud(ax, data, position, color, width=0.7, alpha_v=0.7,
              alpha_s=0.35, scatter_size=8, scatter_offset=0.10,
              violin_offset=0.02, rng_seed=0):
    """
    Vertical raincloud at a single x-position.
        - Left half: scatter (jittered)
        - Right half: half-violin (KDE)
        - Mean shown as a short horizontal tick
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    rng_local = np.random.default_rng(rng_seed)

    # === left: scatter (jittered to the left of position) ===
    jit = rng_local.normal(0, 0.04, size=len(data))
    ax.scatter(position - scatter_offset + jit, data,
               s=scatter_size, color=color, alpha=alpha_s,
               zorder=2, edgecolor="none")

    # === right: half-violin via KDE ===
    if data.std() > 1e-9 and len(data) > 2:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        ymin, ymax = data.min(), data.max()
        pad = (ymax - ymin) * 0.05 + 1e-6
        ys = np.linspace(ymin - pad, ymax + pad, 200)
        dens = kde(ys)
        # normalize to half-width
        dens = dens / dens.max() * (width / 2.0)
        x_right = position + violin_offset + dens
        # filled half-violin
        ax.fill_betweenx(ys, position + violin_offset, x_right,
                         color=color, alpha=alpha_v, linewidth=0, zorder=3)
        # thin outline
        ax.plot(x_right, ys, color=color, lw=0.6, alpha=0.9, zorder=4)
        # baseline (vertical edge of the half-violin)
        ax.plot([position + violin_offset, position + violin_offset],
                [ymin - pad, ymax + pad],
                color=color, lw=0.6, alpha=0.6, zorder=4)

    # mean tick
    mean_val = data.mean()
    ax.hlines(mean_val,
              position - scatter_offset - 0.07,
              position + violin_offset + (width / 2.0) + 0.02,
              colors=COLOR_NEUTRAL, lw=1.3, zorder=5)


# %% [FIG 1] Architectural separation + Temporal origin (merged) ---------------
"""
Merged Fig 1:
  (a) Φ_r(z) vs Φ_r(g) — each as a vertical raincloud (scatter left, half-violin right).
  (b) Φ_r(g) original vs shuffled — same raincloud style.
"""

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

# === (a) z vs g ===
ax = axes[0]
z_vals = dt_clean["phi_r_z"].values
g_vals = dt_clean["phi_r_g"].values

raincloud(ax, z_vals, position=0, color=COLOR_NEUTRAL, rng_seed=1)
raincloud(ax, g_vals, position=1, color=COLOR_TRAINED,  rng_seed=2)

ax.set_xticks([0, 1])
ax.set_xticklabels([r"$\Phi_r(z)$", r"$\Phi_r(g)$"])
ax.set_ylabel(r"$\Phi_r$")
ax.set_title("(a) architectural separation (trained, clean)")
ax.set_xlim(-0.5, 1.6)

ratio = g_vals.mean() / max(z_vals.mean(), 1e-6)
ax.text(0.5, 0.96, f"ratio g/z = {ratio:.0f}×\np < 10⁻¹³⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

# === (b) original vs shuffled ===
ax = axes[1]
g_o = shuffle["phi_g_orig"].values
g_s = shuffle["phi_g_shuf"].values

raincloud(ax, g_o, position=0, color=COLOR_TRAINED, rng_seed=3)
raincloud(ax, g_s, position=1, color=COLOR_NEUTRAL, rng_seed=4)

ax.set_xticks([0, 1])
ax.set_xticklabels(["original", "shuffled\n(temporal)"])
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("(b) temporal origin: shuffle ablation")
ax.set_xlim(-0.5, 1.6)

collapse = (1 - g_s.mean() / g_o.mean()) * 100
ax.text(0.5, 0.96, f"{collapse:.1f}% collapse\np < 10⁻¹³⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

plt.tight_layout()
save(fig, "fig1_separation_and_shuffle")
plt.show()


# %% [FIG 2] Magnitude: Φ_r(g) untrained vs trained ----------------------------
"""
Raincloud-style: each cohort as scatter (left) + half-violin (right).
"""

fig, ax = plt.subplots(figsize=(4.8, 4.0))

g_un = du_clean["phi_r_g"].values
g_tr = dt_clean["phi_r_g"].values

raincloud(ax, g_un, position=0, color=COLOR_UNTRAINED, rng_seed=10)
raincloud(ax, g_tr, position=1, color=COLOR_TRAINED,    rng_seed=11)

ax.set_xticks([0, 1])
ax.set_xticklabels(["untrained", "trained"])
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("Magnitude: untrained vs trained (clean)")
ax.set_xlim(-0.5, 1.6)

dlearned = g_tr.mean() - g_un.mean()
ax.text(0.5, 0.96, f"Δ(learned) = {dlearned:+.2f}\np < 10⁻⁹⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

plt.tight_layout()
save(fig, "fig2_magnitude")
plt.show()


# %% [FIG 3] Atom-group decomposition (3 subplots, each with own y-range) ------
"""
Three side-by-side subplots, one per atom group.
Each shows untrained vs trained as rainclouds with its own y-axis range,
so decoupling and part-driven are not crushed by downward's scale.
"""

groups = ["group_decoupling", "group_downward", "group_part_driven"]
group_labels = ["decoupling\n(whole$\\to$whole)",
                "downward\n(whole$\\to$part)",
                "part-driven\n(part$\\to$$\\cdot$)"]

clean_atoms = atoms[atoms.condition == "clean"]

fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.2))

for i, (g, label) in enumerate(zip(groups, group_labels)):
    ax = axes[i]
    un_vals = clean_atoms[clean_atoms.group == "untrained"][g].values
    tr_vals = clean_atoms[clean_atoms.group == "trained"][g].values

    raincloud(ax, un_vals, position=0, color=COLOR_UNTRAINED, rng_seed=20 + i)
    raincloud(ax, tr_vals, position=1, color=COLOR_TRAINED,    rng_seed=30 + i)

    ax.axhline(0, color=COLOR_NEUTRAL, lw=0.6, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_title(f"{label}\nuntrained: {un_vals.mean():+.2f}  |  trained: {tr_vals.mean():+.2f}",
                 fontsize=10, pad=10)

    if i == 0:
        ax.set_ylabel(r"$\Phi_r$ contribution")

fig.suptitle("Atom-group decomposition (clean)", y=1.00, fontsize=11.5)
plt.tight_layout()
save(fig, "fig3_atom_decomposition")
plt.show()


# %% [FIG 4] Hysteresis curves -------------------------------------------------
"""
Switch-aligned Φ_r(g) trajectories. y ∈ [0.25, 3.0]. No comment box.
"""

fig, ax = plt.subplots(figsize=(6.5, 3.8))

TAU_MIN, TAU_MAX = -100, 140

tr_h = hyst[(hyst.group == "trained") &
            (hyst.tau_bin >= TAU_MIN) & (hyst.tau_bin <= TAU_MAX)].sort_values("tau_bin")
un_h = hyst[(hyst.group == "untrained") &
            (hyst.tau_bin >= TAU_MIN) & (hyst.tau_bin <= TAU_MAX)].sort_values("tau_bin")

ax.fill_between(tr_h["tau_bin"], tr_h["q25"], tr_h["q75"],
                color=COLOR_TRAINED, alpha=0.20, linewidth=0)
ax.plot(tr_h["tau_bin"], tr_h["median"], color=COLOR_TRAINED, lw=1.8,
        label="trained", marker="o", markersize=3.5)

ax.fill_between(un_h["tau_bin"], un_h["q25"], un_h["q75"],
                color=COLOR_UNTRAINED, alpha=0.20, linewidth=0)
ax.plot(un_h["tau_bin"], un_h["median"], color=COLOR_UNTRAINED, lw=1.8,
        label="untrained", marker="s", markersize=3.5)

ax.axvline(0, color=COLOR_NEUTRAL, lw=0.8, ls="--", alpha=0.6)
ax.text(0, 2.92, " regime switch",
        ha="left", va="top", fontsize=8.5, color=COLOR_NEUTRAL)

ax.set_xlabel(r"$\tau$  =  window center $-$ switch  (steps)")
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("Switch-aligned $\\Phi_r(g)$ trajectory")
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(TAU_MIN, TAU_MAX)
ax.set_ylim(0.25, 3.0)

plt.tight_layout()
save(fig, "fig4_hysteresis")
plt.show()


# %% [FIG 5] Atom-level regime sensitivity (unified panel) ---------------------
"""
Single panel: x-axis is atom group, with trained and untrained shown
as side-by-side pairs of pre/post markers. This mirrors Fig 3's layout
(trained/untrained side-by-side) so the eye can compare directly.

For each group:
  - untrained pre (circle, light)
  - untrained post (diamond, solid)  → arrow shows direction
  - trained pre (circle, light)
  - trained post (diamond, solid)    → arrow shows direction

Δ annotations placed beneath each pair.
"""

from scipy.stats import ttest_1samp

fig, ax = plt.subplots(figsize=(9.0, 4.6))

groups = ["group_decoupling", "group_downward", "group_part_driven"]
group_labels = ["decoupling\n(whole→whole)",
                "downward\n(whole→part)",
                "part-driven\n(part→·)"]

# offsets within each group: untrained pre/post, trained pre/post
offsets = {
    ("untrained", "pre"):  -0.30,
    ("untrained", "post"): -0.10,
    ("trained",   "pre"):  +0.10,
    ("trained",   "post"): +0.30,
}

def get_paired(grp_id, g):
    pre = atoms[(atoms.condition == "p20_pre") & (atoms.group == grp_id)]
    post = atoms[(atoms.condition == "p20_post") & (atoms.group == grp_id)]
    m = pre.merge(post, on=["seed", "episode"], suffixes=("_pre", "_post"))
    return m[f"{g}_pre"].values, m[f"{g}_post"].values

# plot each group
for j, g in enumerate(groups):
    for cohort in ["untrained", "trained"]:
        color = COLOR_UNTRAINED if cohort == "untrained" else COLOR_TRAINED
        pre_vals, post_vals = get_paired(cohort, g)
        x_pre  = j + offsets[(cohort, "pre")]
        x_post = j + offsets[(cohort, "post")]

        # arrow from pre mean to post mean
        ax.annotate("",
                    xy=(x_post, post_vals.mean()),
                    xytext=(x_pre, pre_vals.mean()),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
                    zorder=3)
        # markers
        ax.scatter([x_pre], [pre_vals.mean()], s=70, color=color,
                   marker="o", edgecolor="white", linewidth=1.3, zorder=4)
        ax.scatter([x_post], [post_vals.mean()], s=70, color=color,
                   marker="D", edgecolor="white", linewidth=1.3, zorder=4)

        # delta annotation, with p-value
        diffs = post_vals - pre_vals
        _, pv = ttest_1samp(diffs, 0)
        sig = "n.s." if pv >= 0.05 else f"p={pv:.0e}"
        # place below cohort group
        x_mid = (x_pre + x_post) / 2
        ax.text(x_mid, -2.3, f"Δ={diffs.mean():+.2f}\n{sig}",
                ha="center", va="top", fontsize=7.5, color=color)

# group separators
for j in range(len(groups) - 1):
    ax.axvline(j + 0.5, color=COLOR_FAINT, lw=0.6, zorder=0)
ax.axhline(0, color=COLOR_NEUTRAL, lw=0.5, zorder=0)

# cohort labels above each pair within each group
for j in range(len(groups)):
    ax.text(j + offsets[("untrained", "pre")] / 2 + offsets[("untrained", "post")] / 2,
            ax.get_ylim()[1] if False else 3.7,
            "untrained", ha="center", va="bottom",
            fontsize=8, color=COLOR_UNTRAINED)
    ax.text(j + offsets[("trained", "pre")] / 2 + offsets[("trained", "post")] / 2,
            3.7,
            "trained", ha="center", va="bottom",
            fontsize=8, color=COLOR_TRAINED)

ax.set_xticks(np.arange(len(groups)))
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylabel(r"$\Phi_r$ contribution")
ax.set_title("Atom-level regime sensitivity (pre $\\to$ post regime switch)")
ax.set_ylim(-2.5, 4.0)
ax.set_xlim(-0.55, len(groups) - 0.45)

# legend for markers
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NEUTRAL,
           markeredgecolor="white", markersize=9, label="pre"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_NEUTRAL,
           markeredgecolor="white", markersize=9, label="post"),
]
ax.legend(handles=legend_elems, loc="upper right",
          frameon=False, ncol=2)

plt.tight_layout()
save(fig, "fig5_atom_sensitivity")
plt.show()

print("\nAll five figures saved to figs/")