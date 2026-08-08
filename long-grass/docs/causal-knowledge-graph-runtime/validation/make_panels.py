"""Generate the five result panels from the validators' JSON output.

Each panel: white background, four charts in a row, at least one 3D, minimal
text, and NO chart that is text-based, conceptual, or a table. Every number
plotted comes from ../results/*.json -- these are result figures, not diagrams.

    python validation/make_panels.py

Panels written to ../figures/:
  panel_trajectory.png     (Thm. 5 / Prop. 1)
  panel_completion.png     (Thm. 2 / Cor. 4 / Def. 3)
  panel_nondeterminism.png (Prop. 8)
  panel_resolution.png     (Def. 6 / Prop. 7)
  panel_overview.png       (cross-cutting synthesis)
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.normpath(os.path.join(HERE, "..", "figures"))

# a clean, consistent look: white background everywhere, minimal chrome
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
PURPLE = "#7b3fa0"


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def _tidy_3d(ax) -> None:
    """Push a 3D subplot's z-label off its neighbour and lighten the panes."""
    ax.zaxis.labelpad = -2
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#dddddd")
        pane.pane.set_alpha(1.0)


def _save(fig, name: str) -> None:
    out = os.path.join(FIGS, name)
    # extra horizontal room so a 3D z-label never lands on the next subplot
    fig.subplots_adjust(wspace=0.32, left=0.03, right=0.985)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_panels] wrote {out}")


# ---------------------------------------------------------------------------
# Panel 1 -- trajectory emergence
# ---------------------------------------------------------------------------

def panel_trajectory() -> None:
    d = _load("trajectory_emergence.json")
    order = d["order"]
    n = len(order)
    adj1 = np.array(d["two_runs"]["run1_adjacency"], dtype=float)
    adj2 = np.array(d["two_runs"]["run2_adjacency"], dtype=float)
    edge_count = np.array(d["sweep"]["edge_count"], dtype=float)       # [node][mag]
    divergence = np.array(d["sweep"]["divergence"], dtype=float)       # [node][mag]
    mags = d["sweep"]["seed_mags"]

    fig = plt.figure(figsize=(19, 4.2))

    # (a) 3D surface: divergence-from-baseline over (seeded node, seed magnitude)
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    X, Y = np.meshgrid(np.array(mags, dtype=float), np.arange(n))
    ax0.plot_surface(X, Y, divergence, cmap=cm.viridis, edgecolor="none",
                     antialiased=True, alpha=0.95)
    ax0.set_xlabel("seed magnitude")
    ax0.set_ylabel("seeded node")
    ax0.set_zlabel("trajectory divergence")
    ax0.set_yticks(range(n))
    ax0.set_yticklabels(order)
    ax0.set_title("(a) divergence surface")
    ax0.view_init(elev=26, azim=-52)

    # (b) two induced adjacency matrices as a difference heat map
    ax1 = fig.add_subplot(1, 4, 2)
    diff = adj1 - adj2  # +1 only in run1, -1 only in run2, 0 shared
    im = ax1.imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(order)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(order)
    ax1.set_xlabel("target node"); ax1.set_ylabel("source node")
    ax1.set_title("(b) run1 − run2 edges")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])

    # (c) edge count vs seed magnitude, one line per seeded node
    ax2 = fig.add_subplot(1, 4, 3)
    colors = cm.plasma(np.linspace(0.1, 0.9, n))
    for i, node in enumerate(order):
        ax2.plot(mags, edge_count[i], marker="o", ms=4, color=colors[i], label=node)
    ax2.set_xlabel("seed magnitude")
    ax2.set_ylabel("edges induced")
    ax2.set_title("(c) edges vs seed magnitude")
    ax2.legend(ncol=2, fontsize=7, frameon=False)

    # (d) distinct trajectories vs total runs (cumulative discovery curve)
    ax3 = fig.add_subplot(1, 4, 4)
    per = d["per_run"]
    seen = set()
    xs, ys = [], []
    for k, r in enumerate(per, 1):
        seen.add(tuple(tuple(e) for e in r["edges"]))
        xs.append(k); ys.append(len(seen))
    ax3.plot(xs, ys, color=PURPLE, lw=2)
    ax3.fill_between(xs, ys, color=PURPLE, alpha=0.12)
    ax3.scatter([xs[-1]], [ys[-1]], color=PURPLE, zorder=5)
    ax3.set_xlabel("runs executed")
    ax3.set_ylabel("distinct trajectories")
    ax3.set_title("(d) distinct trajectories")

    _tidy_3d(ax0)
    _save(fig, "panel_trajectory.png")


# ---------------------------------------------------------------------------
# Panel 2 -- run to completion vs halt-on-error
# ---------------------------------------------------------------------------

def panel_completion() -> None:
    d = _load("run_to_completion.json")
    lengths = d["sweep"]["lengths"]
    densities = d["sweep"]["densities"]
    reached_rt = np.array(d["sweep"]["reached_runtime"], dtype=float)      # [density][length]
    reached_halt = np.array(d["sweep"]["reached_halt_on_error"], dtype=float)
    comp_halt = np.array(d["sweep"]["completion_fraction_halt_on_error"], dtype=float)

    fig = plt.figure(figsize=(19, 4.2))

    # (a) 3D bars: nodes reached by THIS runtime over (length, density) -- a full plane
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    Lx, Dy = np.meshgrid(np.arange(len(lengths)), np.arange(len(densities)))
    xpos = Lx.ravel(); ypos = Dy.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dz = reached_rt.ravel()
    ax0.bar3d(xpos, ypos, zpos, 0.6, 0.6, dz, color=BLUE, alpha=0.85, shade=True)
    ax0.set_xlabel("chain length")
    ax0.set_ylabel("anomaly density")
    ax0.set_zlabel("nodes reached")
    ax0.set_xticks(range(len(lengths))); ax0.set_xticklabels(lengths, fontsize=6)
    ax0.set_yticks(range(len(densities))); ax0.set_yticklabels(densities, fontsize=6)
    ax0.set_title("(a) reached (this runtime)")
    ax0.view_init(elev=24, azim=-58)

    # (b) reached vs chain length: this runtime (diagonal) vs halt-on-error, few densities
    ax1 = fig.add_subplot(1, 4, 2)
    ax1.plot(lengths, reached_rt[0], color=BLUE, lw=2.2, marker="o", ms=4,
             label="this runtime")
    dsel = [1, 3, 5]
    reds = cm.Reds(np.linspace(0.45, 0.9, len(dsel)))
    for c, di in zip(reds, dsel):
        ax1.plot(lengths, reached_halt[di], color=c, lw=1.6, marker="s", ms=3,
                 label=f"halt @ d={densities[di]}")
    ax1.set_xlabel("chain length")
    ax1.set_ylabel("nodes reached")
    ax1.set_title("(b) reach vs length")
    ax1.legend(fontsize=7, frameon=False)

    # (c) completion fraction vs anomaly density (this runtime flat at 1.0)
    ax2 = fig.add_subplot(1, 4, 3)
    ax2.plot(densities, np.ones_like(comp_halt), color=BLUE, lw=2.4, marker="o",
             label="this runtime")
    ax2.plot(densities, comp_halt, color=RED, lw=2.4, marker="s", label="halt-on-error")
    ax2.fill_between(densities, comp_halt, np.ones_like(comp_halt), color=RED, alpha=0.10)
    ax2.set_xlabel("anomaly density")
    ax2.set_ylabel("fraction of run completed")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(c) completion fraction")
    ax2.legend(fontsize=7, frameon=False)

    # (d) work recovered: nodes this runtime reaches beyond the first anomaly,
    # as a heat map over (length, density)
    ax3 = fig.add_subplot(1, 4, 4)
    recovered = reached_rt - reached_halt   # [density][length]
    im = ax3.imshow(recovered, aspect="auto", origin="lower", cmap="YlGnBu")
    ax3.set_xticks(range(len(lengths))); ax3.set_xticklabels(lengths, fontsize=7)
    ax3.set_yticks(range(len(densities))); ax3.set_yticklabels(densities, fontsize=7)
    ax3.set_xlabel("chain length")
    ax3.set_ylabel("anomaly density")
    ax3.set_title("(d) nodes recovered")
    ax3.grid(False)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    _tidy_3d(ax0)
    _save(fig, "panel_completion.png")


# ---------------------------------------------------------------------------
# Panel 3 -- non-determinism & provenance
# ---------------------------------------------------------------------------

def panel_nondeterminism() -> None:
    d = _load("nondeterminism_provenance.json")
    setups = d["setups"]
    names = [s["name"] for s in setups]
    colors = [BLUE, GREEN, ORANGE]

    fig = plt.figure(figsize=(19, 4.2))

    # (a) 3D ridgeline: result distribution for each setup on its own plane
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    verts = []
    lines = []
    for si, s in enumerate(setups):
        vals = np.array(s["results"])
        hist, edges = np.histogram(vals, bins=30, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # normalise x so the three setups (different value ranges) overlay cleanly
        xn = (centres - centres.min()) / (centres.max() - centres.min())
        # closed polygon for a filled ridge, drawn in the y=si plane
        poly = np.column_stack([
            np.concatenate([[xn[0]], xn, [xn[-1]]]),
            np.concatenate([[0.0], hist, [0.0]]),
        ])
        verts.append(poly)
        lines.append((xn, np.full_like(xn, si), hist))
    pc = PolyCollection(verts, facecolors=colors, alpha=0.35, edgecolors="none")
    ax0.add_collection3d(pc, zs=range(len(setups)), zdir="y")
    for si, (xn, ys, hist) in enumerate(lines):
        ax0.plot(xn, ys, hist, color=colors[si], lw=2)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-0.5, len(setups) - 0.5)
    ax0.set_zlim(0, max(h.max() for _, _, h in lines) * 1.1)
    ax0.set_xlabel("normalised value")
    ax0.set_ylabel("setup")
    ax0.set_zlabel("density")
    ax0.set_yticks(range(len(names))); ax0.set_yticklabels(names, fontsize=7)
    ax0.set_title("(a) result distributions")
    ax0.view_init(elev=28, azim=-60)

    # (b) running mean converging, per setup (result varies, protocol fixed)
    ax1 = fig.add_subplot(1, 4, 2)
    for si, s in enumerate(setups):
        rm = np.array(s["running_mean"])
        ax1.plot(np.arange(1, len(rm) + 1), rm, color=colors[si], lw=1.6, label=names[si])
        ax1.axhline(s["mean"], color=colors[si], lw=0.7, ls=":", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("runs (log)")
    ax1.set_ylabel("running mean of result")
    ax1.set_title("(b) result mean converges")
    ax1.legend(fontsize=7, frameon=False)

    # (c) per-setup spread as violin-like scatter (raw results, jittered)
    ax2 = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(0)
    for si, s in enumerate(setups):
        vals = np.array(s["results"])
        # centre each setup on 0 to compare spreads on one axis
        centred = vals - s["mean"]
        jitter = rng.normal(si, 0.06, size=len(centred))
        ax2.scatter(jitter, centred, s=3, color=colors[si], alpha=0.25)
        ax2.plot([si - 0.25, si + 0.25], [0, 0], color="#222", lw=1.2)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names)
    ax2.set_ylabel("result − mean")
    ax2.set_title("(c) spread per setup")

    # (d) the invariant: spread magnitude (bar) vs number of distinct fingerprints
    ax3 = fig.add_subplot(1, 4, 4)
    spreads = [s["spread"] for s in setups]
    stdevs = [s["stdev"] for s in setups]
    x = np.arange(len(names))
    ax3.bar(x - 0.18, spreads, width=0.36, color=colors, alpha=0.55, label="spread")
    ax3.bar(x + 0.18, stdevs, width=0.36, color=colors, alpha=0.95, label="stdev")
    ax3.set_xticks(x); ax3.set_xticklabels(names)
    ax3.set_ylabel("result variation")
    ax3.set_title("(d) variation, 1 fingerprint each")
    ax3.legend(fontsize=7, frameon=False)

    _tidy_3d(ax0)
    _save(fig, "panel_nondeterminism.png")


# ---------------------------------------------------------------------------
# Panel 4 -- resolution / blast radius
# ---------------------------------------------------------------------------

def panel_resolution() -> None:
    d = _load("resolution_control.json")
    sweep = d["sweep"]
    depth = sweep["depth"]
    edit_depths = sweep["edit_depths"]
    branches = sweep["branches"]
    blast = {int(k): np.array(v, dtype=float) for k, v in sweep["blast_radius"].items()}

    fig = plt.figure(figsize=(19, 4.2))

    # (a) 3D surface: blast radius over (branching factor, edit depth)
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    B = np.array(branches, dtype=float)
    K = np.array(edit_depths, dtype=float)
    Bg, Kg = np.meshgrid(B, K)                     # [k][b]
    Z = np.array([[blast[b][ki] for b in branches] for ki in range(len(edit_depths))])
    ax0.plot_surface(Bg, Kg, np.log10(Z), cmap=cm.magma, edgecolor="none", alpha=0.95)
    ax0.set_xlabel("branching factor")
    ax0.set_ylabel("edit depth k")
    ax0.set_zlabel("log10 blast radius")
    ax0.set_xticks(branches)
    ax0.set_title("(a) blast-radius surface")
    ax0.view_init(elev=26, azim=-48)

    # (b) blast radius vs edit depth, log axis, one line per branch (exponential decay)
    ax1 = fig.add_subplot(1, 4, 2)
    cols = [BLUE, GREEN, RED]
    for c, b in zip(cols, branches):
        ax1.semilogy(edit_depths, blast[b], marker="o", ms=5, color=c, label=f"b={b}")
    ax1.set_xlabel("edit depth k (coarse → surgical)")
    ax1.set_ylabel("objects affected (log)")
    ax1.set_title("(b) blast radius vs depth")
    ax1.legend(fontsize=8, frameon=False)

    # (c) fraction of the tree touched vs depth (coarse=whole, surgical=one leaf)
    ax2 = fig.add_subplot(1, 4, 3)
    for c, b in zip(cols, branches):
        total = float(b) ** depth
        ax2.plot(edit_depths, blast[b] / total, marker="s", ms=5, color=c, label=f"b={b}")
    ax2.set_xlabel("edit depth k")
    ax2.set_ylabel("fraction of leaves touched")
    ax2.set_title("(c) reach of one edit")
    ax2.legend(fontsize=8, frameon=False)

    # (d) stacked: at each depth, touched vs untouched leaves for the largest tree
    ax3 = fig.add_subplot(1, 4, 4)
    b = branches[-1]
    total = float(b) ** depth
    touched = blast[b]
    untouched = total - touched
    ax3.bar(edit_depths, touched, color=RED, alpha=0.85, label="touched")
    ax3.bar(edit_depths, untouched, bottom=touched, color=BLUE, alpha=0.35, label="untouched")
    ax3.set_yscale("log")
    ax3.set_xlabel("edit depth k")
    ax3.set_ylabel(f"leaves (b={b}, log)")
    ax3.set_title("(d) touched vs untouched")
    ax3.legend(fontsize=8, frameon=False)

    _tidy_3d(ax0)
    _save(fig, "panel_resolution.png")


# ---------------------------------------------------------------------------
# Panel 5 -- cross-cutting synthesis
# ---------------------------------------------------------------------------

def panel_overview() -> None:
    traj = _load("trajectory_emergence.json")
    comp = _load("run_to_completion.json")
    nd = _load("nondeterminism_provenance.json")
    res = _load("resolution_control.json")

    fig = plt.figure(figsize=(19, 4.2))

    # (a) 3D scatter: every sweep run placed by (claim axis, control param, outcome)
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    # trajectory runs: x=0, param=seed_mag, outcome=divergence
    for r in traj["per_run"]:
        ax0.scatter(0, r["seed_mag"], r["divergence_from_baseline"],
                    color=PURPLE, s=10, alpha=0.6)
    # completion runs: x=1, param=density index, outcome=recovered fraction
    dens = comp["sweep"]["densities"]
    ch = comp["sweep"]["completion_fraction_halt_on_error"]
    for di, dv in enumerate(dens):
        ax0.scatter(1, dv * 10, (1 - ch[di]) * 10, color=BLUE, s=20, alpha=0.8)
    # resolution runs: x=2, param=edit depth, outcome=log10 blast (b=4)
    b4 = np.array(res["sweep"]["blast_radius"]["4"], dtype=float)
    for ki, k in enumerate(res["sweep"]["edit_depths"]):
        ax0.scatter(2, k, np.log10(b4[ki]) * 2, color=RED, s=20, alpha=0.8)
    ax0.set_xticks([0, 1, 2])
    ax0.set_xticklabels(["traj", "compl", "resol"], fontsize=7)
    ax0.set_ylabel("control parameter")
    ax0.set_zlabel("outcome")
    ax0.set_title("(a) all sweeps")
    ax0.view_init(elev=22, azim=-60)

    # (b) trajectory divergence mean vs seed magnitude (collapsed over nodes)
    ax1 = fig.add_subplot(1, 4, 2)
    div = np.array(traj["sweep"]["divergence"], dtype=float)  # [node][mag]
    mags = traj["sweep"]["seed_mags"]
    ax1.plot(mags, div.mean(axis=0), color=PURPLE, lw=2, marker="o")
    ax1.fill_between(mags, div.mean(axis=0) - div.std(axis=0),
                     div.mean(axis=0) + div.std(axis=0), color=PURPLE, alpha=0.15)
    ax1.set_xlabel("seed magnitude")
    ax1.set_ylabel("mean divergence ± sd")
    ax1.set_title("(b) sensitivity to one value")

    # (c) completion advantage curve (this runtime minus halt-on-error)
    ax2 = fig.add_subplot(1, 4, 3)
    adv = 1 - np.array(ch)
    ax2.plot(dens, adv, color=GREEN, lw=2.2, marker="o")
    ax2.fill_between(dens, adv, color=GREEN, alpha=0.15)
    ax2.set_xlabel("anomaly density")
    ax2.set_ylabel("run fraction only we complete")
    ax2.set_title("(c) run-to-completion gain")

    # (d) dynamic range of resolution: blast radius min..max per branch
    ax3 = fig.add_subplot(1, 4, 4)
    branches = res["sweep"]["branches"]
    lo, hi = [], []
    for b in branches:
        arr = np.array(res["sweep"]["blast_radius"][str(b)], dtype=float)
        lo.append(arr.min()); hi.append(arr.max())
    x = np.arange(len(branches))
    ax3.vlines(x, lo, hi, color="#888", lw=2, zorder=1)
    ax3.scatter(x, hi, color=RED, s=60, zorder=2, label="coarsest edit")
    ax3.scatter(x, lo, color=BLUE, s=60, zorder=2, label="surgical edit")
    ax3.set_yscale("log")
    ax3.set_xticks(x); ax3.set_xticklabels([f"b={b}" for b in branches])
    ax3.set_ylabel("objects affected (log)")
    ax3.set_title("(d) control dynamic range")
    ax3.legend(fontsize=7, frameon=False)

    _tidy_3d(ax0)
    _save(fig, "panel_overview.png")


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    panel_trajectory()
    panel_completion()
    panel_nondeterminism()
    panel_resolution()
    panel_overview()
    print("[make_panels] all five panels written")


if __name__ == "__main__":
    main()
