# -*- coding: utf-8 -*-
"""
figures.py
-----------
All five paper figures, one file, Spyder-cell-friendly.

Run cells top-to-bottom (Ctrl+Enter in Spyder):
  Cell [SETUP]   loads all CSVs once
  Cell [FIG 1]   Architectural separation: Φ_r(g) vs Φ_r(z)
  Cell [FIG 2]   Temporal origin: shuffled-g collapse
  Cell [FIG 3]   Learning's two effects (magnitude + atom composition)
  Cell [FIG 4]   Hysteresis curves: switch-aligned Φ_r(g)
  Cell [FIG 5]   Atom-level regime sensitivity

Outputs saved to figs/ as both .pdf (vector) and .png (preview).
"""

# %% [SETUP] -------------------------------------------------------------------
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- styling ----
COLOR_TRAINED   = "#1f4e79"     # deep blue
COLOR_UNTRAINED = "#d65f3e"     # red-orange
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

# ---- load all CSVs once ----
print("Loading CSVs...")
dt_clean = pd.read_csv(DATA / "v2_decomp_trained.csv")
du_clean = pd.read_csv(DATA / "v2_decomp_untrained.csv")
prepost  = pd.read_csv(DATA / "v2_prepost.csv")
shuffle  = pd.read_csv(DATA / "v2_shuffle.csv")
hyst     = pd.read_csv(DATA / "hysteresis_binned.csv")
atoms    = pd.read_csv(DATA / "atoms_decomp.csv")
print(f"  trained clean: {len(dt_clean)}")
print(f"  untrained clean: {len(du_clean)}")
print(f"  prepost: {len(prepost)} ({prepost.group.value_counts().to_dict()})")
print(f"  shuffle: {len(shuffle)}")
print(f"  hyst binned: {len(hyst)}")
print(f"  atoms: {len(atoms)} ({atoms.groupby(['group','condition']).size().to_dict()})")


# %% [FIG 1] Architectural separation: Φ_r(g) ≫ Φ_r(z) ------------------------
"""
Two panels:
  (a) per-seed mean bar plot of Φ_r(z) and Φ_r(g)
  (b) pooled violin / box of the two distributions
"""

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6),
                         gridspec_kw={"width_ratios": [1.7, 1.0]})

# --- (a) per-seed bars ---
ax = axes[0]
per_seed = dt_clean.groupby("seed").agg(z=("phi_r_z", "mean"),
                                          g=("phi_r_g", "mean")).reset_index()
seeds = per_seed["seed"].values
x = np.arange(len(seeds))
w = 0.4
ax.bar(x - w/2, per_seed["z"], w, color=COLOR_FAINT,
       label=r"$\Phi_r(z)$", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, per_seed["g"], w, color=COLOR_TRAINED,
       label=r"$\Phi_r(g)$", edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(seeds, fontsize=7)
ax.set_xlabel("seed")
ax.set_ylabel(r"$\Phi_r$")
ax.set_title("(a) per-seed mean (trained, clean)")
ax.legend(loc="upper left", frameon=False)
ax.set_xlim(-0.7, len(seeds) - 0.3)

# --- (b) pooled violins ---
ax = axes[1]
data = [dt_clean["phi_r_z"].values, dt_clean["phi_r_g"].values]
parts = ax.violinplot(data, positions=[0, 1], widths=0.7,
                       showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([COLOR_FAINT, COLOR_TRAINED][i])
    pc.set_alpha(0.7)
    pc.set_edgecolor("white")
parts["cmedians"].set_color(COLOR_NEUTRAL)
parts["cmedians"].set_linewidth(1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels([r"$\Phi_r(z)$", r"$\Phi_r(g)$"])
ax.set_ylabel(r"$\Phi_r$")
ax.set_title(f"(b) pooled (n={len(dt_clean)})")

# annotation
ratio = dt_clean["phi_r_g"].mean() / max(dt_clean["phi_r_z"].mean(), 1e-6)
ax.text(0.5, 0.96, f"ratio g/z = {ratio:.0f}×\np < 10⁻¹³⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

plt.tight_layout()
save(fig, "fig1_architectural_separation")
plt.show()


# %% [FIG 2] Temporal origin: shuffled-g collapse ------------------------------
"""
Side-by-side: Φ_r(g) original vs temporally shuffled.
Bar with individual episodes overlaid as scatter.
"""

fig, ax = plt.subplots(figsize=(4.2, 3.6))

g_o = shuffle["phi_g_orig"].values
g_s = shuffle["phi_g_shuf"].values

# bars
x = [0, 1]
means = [g_o.mean(), g_s.mean()]
sds = [g_o.std(), g_s.std()]
ax.bar(x, means, color=[COLOR_TRAINED, COLOR_FAINT], width=0.55,
       edgecolor="white", linewidth=1)

# scatter overlay (jittered)
rng = np.random.default_rng(0)
for xi, vals in zip(x, [g_o, g_s]):
    jit = rng.normal(0, 0.05, size=len(vals))
    ax.scatter(xi + jit, vals, s=8, color=COLOR_NEUTRAL, alpha=0.25, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(["original", "shuffled\n(temporal)"])
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("Temporal origin: shuffle collapses Φ_r(g)")

# annotation
collapse = (1 - g_s.mean() / g_o.mean()) * 100
ax.text(0.5, 0.96, f"{collapse:.1f}% collapse\np < 10⁻¹³⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

# connect mean points
ax.plot([0, 1], means, color=COLOR_NEUTRAL, alpha=0.4, zorder=1, lw=0.8)

plt.tight_layout()
save(fig, "fig2_shuffle_collapse")
plt.show()


# %% [FIG 3] Learning's two effects -------------------------------------------
"""
(a) Φ_r(g) magnitude: untrained > trained (paradox)
(b) Atom decomposition: stacked bar, group sums, untrained vs trained, clean
"""

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6),
                         gridspec_kw={"width_ratios": [1.0, 1.2]})

# --- (a) magnitude comparison ---
ax = axes[0]
data = [du_clean["phi_r_g"].values, dt_clean["phi_r_g"].values]
parts = ax.violinplot(data, positions=[0, 1], widths=0.7,
                       showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([COLOR_UNTRAINED, COLOR_TRAINED][i])
    pc.set_alpha(0.7)
    pc.set_edgecolor("white")
parts["cmedians"].set_color(COLOR_NEUTRAL)
parts["cmedians"].set_linewidth(1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["untrained", "trained"])
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("(a) magnitude (clean)")

dlearned = dt_clean["phi_r_g"].mean() - du_clean["phi_r_g"].mean()
ax.text(0.5, 0.96, f"Δ(learned) = {dlearned:+.2f}\np < 10⁻⁹⁰",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.7))

# --- (b) atom group composition ---
ax = axes[1]
groups = ["group_decoupling", "group_downward", "group_part_driven"]
group_labels = ["decoupling\n(whole→whole)", "downward\n(whole→part)",
                "part-driven\n(part→·)"]

clean_atoms = atoms[atoms.condition == "clean"]
tr = clean_atoms[clean_atoms.group == "trained"]
un = clean_atoms[clean_atoms.group == "untrained"]

x = np.arange(len(groups))
w = 0.38
un_vals = [un[g].mean() for g in groups]
tr_vals = [tr[g].mean() for g in groups]
un_err  = [un[g].std() / np.sqrt(len(un)) for g in groups]
tr_err  = [tr[g].std() / np.sqrt(len(tr)) for g in groups]

ax.bar(x - w/2, un_vals, w, yerr=un_err,
       color=COLOR_UNTRAINED, label="untrained",
       edgecolor="white", linewidth=0.5, error_kw={"lw": 0.8, "capsize": 2.5})
ax.bar(x + w/2, tr_vals, w, yerr=tr_err,
       color=COLOR_TRAINED, label="trained",
       edgecolor="white", linewidth=0.5, error_kw={"lw": 0.8, "capsize": 2.5})

ax.axhline(0, color=COLOR_NEUTRAL, lw=0.5, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=8.5)
ax.set_ylabel(r"$\Phi_r$ contribution")
ax.set_title("(b) atom-group decomposition (clean)")
ax.legend(loc="upper right", frameon=False)

# highlight decoupling sign flip
ax.annotate("sign flip:\nlearning makes\nthe whole irreducible",
            xy=(0, tr_vals[0]), xytext=(0.5, 1.6),
            fontsize=8, ha="left", color=COLOR_NEUTRAL,
            arrowprops=dict(arrowstyle="->", color=COLOR_NEUTRAL,
                            lw=0.7, connectionstyle="arc3,rad=-0.2"))

plt.tight_layout()
save(fig, "fig3_two_effects")
plt.show()


# %% [FIG 4] Hysteresis curves ------------------------------------------------
"""
Switch-aligned Φ_r(g) trajectories, trained vs untrained.
Median + IQR shaded band. Vertical line at switch (tau=0).
Focus window: tau ∈ [-100, +140] (avoid the early-episode drift).
"""

fig, ax = plt.subplots(figsize=(6.5, 3.8))

TAU_MIN, TAU_MAX = -100, 140

tr_h = hyst[(hyst.group == "trained") &
            (hyst.tau_bin >= TAU_MIN) & (hyst.tau_bin <= TAU_MAX)].sort_values("tau_bin")
un_h = hyst[(hyst.group == "untrained") &
            (hyst.tau_bin >= TAU_MIN) & (hyst.tau_bin <= TAU_MAX)].sort_values("tau_bin")

# trained
ax.fill_between(tr_h["tau_bin"], tr_h["q25"], tr_h["q75"],
                color=COLOR_TRAINED, alpha=0.25, linewidth=0)
ax.plot(tr_h["tau_bin"], tr_h["median"], color=COLOR_TRAINED, lw=1.8,
        label="trained", marker="o", markersize=3)

# untrained
ax.fill_between(un_h["tau_bin"], un_h["q25"], un_h["q75"],
                color=COLOR_UNTRAINED, alpha=0.25, linewidth=0)
ax.plot(un_h["tau_bin"], un_h["median"], color=COLOR_UNTRAINED, lw=1.8,
        label="untrained", marker="s", markersize=3)

# switch line
ax.axvline(0, color=COLOR_NEUTRAL, lw=0.8, ls="--", alpha=0.6)
ax.text(0, ax.get_ylim()[1] * 0.97, " regime switch",
        ha="left", va="top", fontsize=8, color=COLOR_NEUTRAL)

ax.set_xlabel(r"$\tau$  =  window center $-$ switch  (steps)")
ax.set_ylabel(r"$\Phi_r(g)$")
ax.set_title("Switch-aligned hysteresis: trained adapts slowly, untrained jumps")
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(TAU_MIN, TAU_MAX)

# annotation: overshoot for trained
ax.text(0.05, 0.05,
        "trained: transient overshoot →\n"
        "             slow relaxation\n"
        "untrained: immediate drop, then flat",
        transform=ax.transAxes, fontsize=8, color=COLOR_NEUTRAL,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=COLOR_NEUTRAL, lw=0.5))

plt.tight_layout()
save(fig, "fig4_hysteresis")
plt.show()


# %% [FIG 5] Atom-level regime sensitivity ------------------------------------
"""
Paired pre→post changes per atom group, trained and untrained side by side.
Two panels (trained, untrained), each shows three group-Δ paired plots.
"""

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)

groups = ["group_decoupling", "group_downward", "group_part_driven"]
group_short = ["decoupling", "downward", "part-driven"]

for ax_i, grp_id in enumerate(["trained", "untrained"]):
    ax = axes[ax_i]
    color = [COLOR_TRAINED, COLOR_UNTRAINED][ax_i]

    pre = atoms[(atoms.condition == "p20_pre") & (atoms.group == grp_id)]
    post = atoms[(atoms.condition == "p20_post") & (atoms.group == grp_id)]
    merged = pre.merge(post, on=["seed", "episode"], suffixes=("_pre", "_post"))

    for j, g in enumerate(groups):
        pre_vals = merged[f"{g}_pre"].values
        post_vals = merged[f"{g}_post"].values
        diffs = post_vals - pre_vals

        # individual paired lines (faint)
        rng = np.random.default_rng(j)
        for p, q in zip(pre_vals, post_vals):
            ax.plot([j - 0.18, j + 0.18], [p, q],
                    color=color, alpha=0.04, lw=0.5, zorder=1)

        # pre/post means as filled markers
        ax.scatter([j - 0.18], [pre_vals.mean()], s=55,
                   color=color, edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([j + 0.18], [post_vals.mean()], s=55,
                   color=color, edgecolor="white", linewidth=1.2,
                   marker="D", zorder=4)
        # connecting arrow on means
        ax.annotate("", xy=(j + 0.18, post_vals.mean()),
                    xytext=(j - 0.18, pre_vals.mean()),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

        # delta annotation
        # p-value via paired t (recompute here, kept simple)
        from scipy.stats import ttest_1samp
        _, pv = ttest_1samp(diffs, 0)
        sig = "ns" if pv >= 0.05 else f"p={pv:.0e}"
        ax.text(j, ax.get_ylim()[0] if hasattr(ax, '_y0') else -1.0,
                f"Δ={diffs.mean():+.2f}\n{sig}",
                ha="center", va="top", fontsize=7.5, color=COLOR_NEUTRAL)

    ax.axhline(0, color=COLOR_NEUTRAL, lw=0.5, zorder=0)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(group_short, fontsize=9)
    ax.set_title(f"{grp_id}", color=color)
    if ax_i == 0:
        ax.set_ylabel(r"$\Phi_r$ contribution")

# shared legend (markers)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NEUTRAL,
           markeredgecolor="white", markersize=8, label="pre"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_NEUTRAL,
           markeredgecolor="white", markersize=8, label="post"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Atom-level regime sensitivity: trained protects decoupling, "
             "lets downward adapt",
             y=1.07, fontsize=11)

plt.tight_layout()
save(fig, "fig5_atom_sensitivity")
plt.show()

print("\nAll five figures saved to figs/")
