"""Figure 2: the matched serpentine probe drawn on the original pygame render.

Usage:
    python fig2_probe_overlay.py --render aaai_render.png \
        --probe probes/serpentine_2roundtrip_dwell4.json --out fig2_setup.pdf [--timestamps] [--strip]

--timestamps  marks the probe start (circle) and reversal point (square) with the
              steps at which the agent is there.
--strip       adds panel B: observation-noise level sigma(t) of the occupied zone
              over the probe steps (implies --timestamps).

The render is the screenshot of PygameGridViewer (15 x 9 grid, info panel on
top); the info panel and window title bar are cropped automatically. The path
and the sigma(t) sequence are reconstructed from the replay action list, so they
are the ones actually imposed during evaluation.
"""
import argparse
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Polygon, FancyBboxPatch, Rectangle

GRID_W, GRID_H = 15, 9
ZONE_RGB = np.array([(85, 55, 40), (40, 75, 55), (35, 55, 90)])  # pygame_viewer.zone_colors
AGENT_RGB = (230 / 255, 230 / 255, 230 / 255)                      # pygame_viewer.agent_color
ZONE_SIGMA = (0.6, 0.3, 0.05)
PATH_YELLOW = "#FFD21F"
INK, MUTED = "#222222", "#666666"
MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}   # up, down, left, right, stay
WAYPOINTS = [(1, 2), (13, 2), (13, 4), (1, 4), (1, 6), (13, 6)]       # fallback without --probe


def zone_of_x(x):
    return 0 if x < GRID_W / 3 else (1 if x < 2 * GRID_W / 3 else 2)


# ----------------------------------------------------------------------------- capture
def crop_grid(img):
    """Return the grid part of the screenshot and its extent in cell coordinates."""
    a = np.asarray(img.convert("RGB")).astype(int)
    dist = np.min(np.linalg.norm(a[:, :, None, :] - ZONE_RGB[None, None], axis=-1), axis=-1)
    zone_rows = np.where((dist < 15).mean(axis=1) > 0.5)[0]
    top = int(zone_rows.min()) - 1          # include the top grid line
    grid = img.crop((0, top, img.width, img.height))
    cell = grid.width / GRID_W              # cells are square
    extent = (-0.5, GRID_W - 0.5, -0.5 + grid.height / cell, -0.5)
    return grid, extent


def cell_slices(a, margin=3):
    s = a.shape[1] / GRID_W
    out = {}
    for j in range(int(np.ceil(a.shape[0] / s))):
        for i in range(GRID_W):
            x0, x1 = int(np.ceil(i * s + margin)), int(np.floor((i + 1) * s - margin))
            y0, y1 = int(np.ceil(j * s + margin)), min(int(np.floor((j + 1) * s - margin)), a.shape[0])
            if x1 > x0 and y1 > y0:
                out[(i, j)] = (slice(y0, y1), slice(x0, x1))
    return out


def zone_colours(grid):
    """The capture's own colour for each zone (median over pixels near that zone colour)."""
    a = np.asarray(grid.convert("RGB"))
    cells = cell_slices(a)
    cols = []
    for k in range(3):
        px = np.concatenate([a[sl].reshape(-1, 3) for (i, j), sl in cells.items() if zone_of_x(i) == k])
        px = px[np.linalg.norm(px.astype(int) - ZONE_RGB[k], axis=1) < 20]
        cols.append(np.median(px, axis=0).astype(np.uint8))
    return np.array(cols)


def clean_capture(grid, colours):
    """Remove screenshot specks (and the drawn agent) inside cells and the window-border column."""
    a = np.asarray(grid.convert("RGB")).copy()
    a[:, -1] = a[:, -2]                     # last column is the window border
    for (i, j), sl in cell_slices(a).items():
        a[sl] = colours[zone_of_x(i)]
    return Image.fromarray(a)


# ----------------------------------------------------------------------------- probe
def probe_positions(probe_json):
    """Agent cell at every probe step t = 0..T (position at which x_t is observed)."""
    if probe_json is not None:
        spec = json.load(open(probe_json))
        start, actions = tuple(spec["start_xy"]), [int(a) for a in spec["actions"]]
        dwell = int(spec["dwell_after_each_move"])
        oneway = int(spec["temporal_steps"]) // (2 * int(spec["roundtrips"]))
    else:                                   # rebuild the same sequence from the waypoints
        dwell, start = 4, WAYPOINTS[0]
        fwd = []
        for (x0, y0), (x1, y1) in zip(WAYPOINTS, WAYPOINTS[1:]):
            step = {(1, 0): 3, (-1, 0): 2, (0, 1): 1, (0, -1): 0}[(np.sign(x1 - x0), np.sign(y1 - y0))]
            fwd += [step] * (abs(x1 - x0) + abs(y1 - y0))
        rev = [{0: 1, 1: 0, 2: 3, 3: 2}[a] for a in fwd[::-1]]
        actions = [a for m in (fwd + rev) * 2 for a in [m] + [4] * dwell]
        oneway = len(fwd) * (dwell + 1)
    x, y = start
    pos = [(x, y)]
    for a in actions:
        dx, dy = MOVES[a]
        x, y = x + dx, y + dy
        pos.append((x, y))
    return pos, oneway, len(actions)


def outbound_waypoints(pos, oneway):
    pts = [p for t, p in enumerate(pos[:oneway + 1]) if t == 0 or p != pos[t - 1]]
    way = [pts[0]]
    for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
        if (p1[0] - p0[0], p1[1] - p0[1]) != (p2[0] - p1[0], p2[1] - p1[1]):
            way.append(p1)
    way.append(pts[-1])
    return way


# ----------------------------------------------------------------------------- drawing helpers
def arrowhead(ax, x, y, dx, dy, size=0.21, zorder=6):
    """Filled triangle pointing along (dx, dy), centred at (x, y), in cell units."""
    px, py = -dy, dx
    tip = (x + dx * size, y + dy * size)
    b1 = (x - dx * size + px * size * 0.95, y - dy * size + py * size * 0.95)
    b2 = (x - dx * size - px * size * 0.95, y - dy * size - py * size * 0.95)
    ax.add_patch(Polygon([tip, b1, b2], closed=True, facecolor=PATH_YELLOW,
                         edgecolor="black", linewidth=0.8, zorder=zorder, joinstyle="miter"))


BOX = dict(boxstyle="round,pad=0.3,rounding_size=0.2", facecolor=(0, 0, 0, 0.55),
           edgecolor=(1, 1, 1, 0.75), linewidth=0.8)
ARROW = dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color="white",
             linewidth=1.0, shrinkA=2, shrinkB=1)
HALO = [pe.Stroke(linewidth=2.2, foreground=(0, 0, 0, 0.85)), pe.Normal()]


def draw_map(fig, ax, img, extent, way, agent, times=None, legend_text="Probe path (2 round trips)",
             two_way=False):
    ax.imshow(np.asarray(img), extent=extent, interpolation="none", zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.axis("off")

    xs, ys = zip(*way)
    ax.plot(xs, ys, color=PATH_YELLOW, linewidth=3.0, solid_joinstyle="miter", solid_capstyle="butt",
            zorder=5, path_effects=[pe.Stroke(linewidth=5.0, foreground="black"), pe.Normal()])
    for (x0, y0), (x1, y1) in zip(way, way[1:]):
        dx, dy = np.sign(x1 - x0), np.sign(y1 - y0)
        if dy == 0:                                  # horizontal leg: one arrow per zone
            for zx in (2.75, 7.0, 11.25):
                if min(x0, x1) < zx < max(x0, x1):
                    if (round(zx), y0) == agent:     # keep the arrow off the agent
                        zx -= dx * 1.0
                    arrowhead(ax, zx, y0, dx, dy)
        else:                                        # vertical connector
            arrowhead(ax, x0, (y0 + y1) / 2, dx, dy)

    ax.add_patch(Circle(agent, 1 / 3, facecolor=AGENT_RGB, edgecolor="black", linewidth=0.6, zorder=7))
    (sx, sy), (ex, ey) = way[0], way[-1]
    ax.add_patch(Circle((sx, sy), 0.2, facecolor=PATH_YELLOW, edgecolor="black", linewidth=1.0, zorder=7))

    for k, cx in enumerate((2.0, 7.0, 12.0)):
        ax.text(cx, 0.45, f"Zone {k}\nσ = {ZONE_SIGMA[k]:g}", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=8, linespacing=1.25,
                bbox=dict(BOX, boxstyle="round,pad=0.35,rounding_size=0.25"))
    ax.annotate("Start", xy=(sx, sy + 0.22), xytext=(sx + 0.35, sy + 1.05), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=9,
                bbox=dict(BOX, edgecolor=PATH_YELLOW, linewidth=1.0), arrowprops=ARROW)
    ax.annotate("Agent", xy=(agent[0] + 0.25, agent[1] - 0.25), xytext=(agent[0] + 1.35, agent[1] - 1.0),
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=9,
                bbox=BOX, arrowprops=ARROW)

    if times is not None:                            # when the agent is at the two ends of the route
        t_start, t_turn = times
        ax.add_patch(Rectangle((ex - 0.19, ey - 0.19), 0.38, 0.38, facecolor=PATH_YELLOW,
                               edgecolor="black", linewidth=1.0, zorder=7))
        ax.text(sx - 0.3, sy - 0.48, "t = " + ", ".join(map(str, t_start)), ha="left", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=9, path_effects=HALO)
        ax.text(ex + 0.3, ey + 0.52, "t = " + ", ".join(map(str, t_turn)), ha="right", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=9, path_effects=HALO)

    # legend, sized to its text and right-aligned inside the grid
    right, ly = GRID_W - 0.5 - 0.3, 7.8
    txt = ax.text(right - 0.2, ly, legend_text, ha="right", va="center",
                  fontsize=8.5, fontweight="bold", color="white", zorder=9)
    fig.canvas.draw()
    bb = txt.get_window_extent().transformed(ax.transData.inverted())
    sample_x1 = bb.x0 - 0.3
    sample_x0 = sample_x1 - (1.25 if two_way else 0.95)
    box_x0 = sample_x0 - 0.25
    ax.add_patch(FancyBboxPatch((box_x0, ly - 0.42), right - box_x0, 0.84,
                                boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=(0, 0, 0, 0.55), edgecolor=(1, 1, 1, 0.75), linewidth=0.8, zorder=8))
    line_x0 = sample_x0 + (0.1 if two_way else 0.0)
    ax.plot([line_x0, sample_x1 - 0.1], [ly, ly], color=PATH_YELLOW, linewidth=3.0, zorder=9,
            solid_capstyle="butt", path_effects=[pe.Stroke(linewidth=5.0, foreground="black"), pe.Normal()])
    arrowhead(ax, sample_x1 - 0.12, ly, 1, 0, size=0.2, zorder=9)
    if two_way:                                      # out and back along the same route
        arrowhead(ax, sample_x0 + 0.12, ly, -1, 0, size=0.2, zorder=9)


def draw_strip(ax, pos, n_steps, oneway, colours):
    """Panel B: sigma of the occupied zone at each probe step, coloured by zone."""
    zones = [zone_of_x(p[0]) for p in pos[:n_steps]]
    t0 = 0
    for t in range(1, n_steps + 1):
        if t == n_steps or zones[t] != zones[t0]:
            k = zones[t0]
            ax.add_patch(Rectangle((t0, 0), t - t0, ZONE_SIGMA[k], facecolor=colours[k] / 255,
                                   edgecolor="white", linewidth=0.6, zorder=2))
            t0 = t
    marks = np.arange(0, n_steps + 1, oneway)
    for i, t in enumerate(marks):
        kw = dict(markersize=6, markerfacecolor=PATH_YELLOW, markeredgecolor="black",
                  markeredgewidth=0.8, clip_on=False, zorder=4)
        ax.plot([t], [0.72], marker="o" if i % 2 == 0 else "s", **kw)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 0.8)
    ax.set_xticks(marks)
    ax.set_yticks(ZONE_SIGMA[::-1])
    ax.set_yticklabels([f"{s:g}" for s in ZONE_SIGMA[::-1]])
    ax.set_xlabel("Probe step", fontsize=8.5, color=INK, labelpad=2)
    ax.set_ylabel("σ", fontsize=9.5, color=INK, rotation=0, labelpad=6, va="center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK, width=0.6, length=2.5, pad=2)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="aaai_render.png")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", default="fig2_setup.pdf")
    ap.add_argument("--width_in", type=float, default=6.0)
    ap.add_argument("--agent_xy", type=int, nargs=2, default=(7, 4),
                    help="cell of the agent drawn in the render (redrawn above the path)")
    ap.add_argument("--no_clean", action="store_true", help="use the capture without speck removal")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--strip", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    grid, extent = crop_grid(Image.open(args.render))
    colours = zone_colours(grid)
    if not args.no_clean:
        grid = clean_capture(grid, colours)          # the agent is redrawn above the path
    up = grid.resize((grid.width * 4, grid.height * 4), Image.NEAREST)  # keeps cell edges sharp

    pos, oneway, n_steps = probe_positions(args.probe)
    way = outbound_waypoints(pos, oneway)
    times = None
    if args.timestamps or args.strip:
        ends = list(range(0, n_steps + 1, oneway))
        times = (ends[0::2], ends[1::2])
    legend = "Probe path: 2 round trips" if times else "Probe path (2 round trips)"
    two_way = times is not None                      # legend sample drawn as out-and-back
    ratio = (extent[2] - extent[3]) / (extent[1] - extent[0])

    w = args.width_in
    if not args.strip:
        fig = plt.figure(figsize=(w, w * ratio))
        ax = fig.add_axes([0, 0, 1, 1])
        draw_map(fig, ax, up, extent, way, tuple(args.agent_xy), times, legend, two_way)
    else:
        left, right = 0.56, 0.16                     # inches
        mw = w - left - right
        mh = mw * ratio
        title_h, gap, strip_h, bottom = 0.27, 0.22, 0.52, 0.40
        h = title_h + mh + gap + title_h + strip_h + bottom
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_axes([left / w, (h - title_h - mh) / h, mw / w, mh / h])
        draw_map(fig, ax, up, extent, way, tuple(args.agent_xy), times, legend, two_way)
        ax.set_title("A. Environment and probe path", fontsize=9.5, color=INK, pad=5)
        axb = fig.add_axes([left / w, bottom / h, mw / w, strip_h / h])
        draw_strip(axb, pos, n_steps, oneway, colours)
        axb.set_title("B. Observation noise along the probe", fontsize=9.5, color=INK, pad=9)

    fig.savefig(args.out)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=300)
    print("saved", args.out, "| waypoints", way, "| one-way", oneway, "| steps", n_steps)


if __name__ == "__main__":
    main()
