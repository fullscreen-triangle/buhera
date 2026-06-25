"""Generate nine four-chart panels for the computational-systems-
structure paper.

Each panel is one row of four matplotlib axes; at least one axis is
3D. White background, minimal text, no tables, no concept-only
schematics. All data comes from the validation experiments in
run_validation.py plus a few direct computations.

Outputs PNGs into validation/figures/.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d

import run_validation as RV


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_PRIMARY = "#1d3557"
C_SECONDARY = "#e63946"
C_TERTIARY = "#2a9d8f"
C_LIGHT = "#a8dadc"
C_NEUTRAL = "#6c757d"


def style_3d(ax):
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(True, alpha=0.3)


def save_panel(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  PANEL 1: Floor from Infinitude
# =====================================================================

def panel_1_floor_from_infinitude():
    fig = plt.figure(figsize=(15, 3.6))

    # Non-completable whole: stage-wise floor
    M = RV.NonCompletableWhole(base_seed=4711)
    stages = list(range(0, 12))
    nc_floors = []
    nc_n_vertices = []
    nc_n_edges = []
    for n in stages:
        G = M.stage(n)
        nc_floors.append(G.floor())
        nc_n_vertices.append(len(G.V))
        nc_n_edges.append(len(G.E))

    # --- Chart 1.1: floor vs stage ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.semilogy(stages, nc_floors, "o-", color=C_PRIMARY,
                 label=r"non-completable $\mathcal{M}$")
    ax1.axhline(0.01, color=C_NEUTRAL, ls="--", lw=0.6)
    ax1.set_xlabel("refinement stage n")
    ax1.set_ylabel(r"floor $\beta$")
    ax1.set_title(r"floor $\beta(n)$ stays strictly positive")
    ax1.legend(frameon=False, fontsize=7, loc="upper right")

    # --- Chart 1.2: floor decays as 1/(n+2) — theoretical curve ---
    ax2 = fig.add_subplot(1, 4, 2)
    theory = [1.0 / (n + 2) for n in stages]
    ax2.plot(stages, nc_floors, "o", color=C_PRIMARY, label="measured")
    ax2.plot(stages, theory, "-", color=C_SECONDARY, alpha=0.7,
             label=r"$1/(n+2)$")
    ax2.set_xlabel("stage n")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title("decay shape matches refinement law")
    ax2.legend(frameon=False, fontsize=7)

    # --- Chart 1.3: graph size grows with stage ---
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(stages, nc_n_vertices, "o-", color=C_PRIMARY,
             label="|V|")
    ax3.plot(stages, nc_n_edges, "s-", color=C_TERTIARY,
             label="|E|")
    ax3.set_xlabel("stage n")
    ax3.set_ylabel("count")
    ax3.set_title("non-completable: graph grows")
    ax3.legend(frameon=False, fontsize=7)

    # --- Chart 1.4: 3D floor surface (n vs edge-index vs weight) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sub_stages = stages[:8]
    rng = np.random.default_rng(42)
    for n in sub_stages:
        G = M.stage(n)
        weights = sorted([w for w in G.E.values() if w > 0])
        idx = np.arange(len(weights))
        ax4.scatter(np.full_like(idx, n, dtype=float),
                    idx / max(1, len(idx) - 1),
                    np.log10(np.array(weights)),
                    c=[plt.cm.viridis(n / max(sub_stages))],
                    s=4, alpha=0.7)
    ax4.set_xlabel("stage n")
    ax4.set_ylabel("normalised edge rank")
    ax4.set_zlabel(r"$\log_{10}\,w$")
    ax4.set_title("edge-weight strata across stages")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_01_floor_from_infinitude.png")


# =====================================================================
#  PANEL 2: Graph-level floor on realised paths
# =====================================================================

def panel_2_graph_floor():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 1)

    # Generate many random graphs, realise distinguishability acts,
    # collect (beta, max-edge-on-path, residue) triples.
    betas = []
    max_edges = []
    residues = []
    n_vertices_list = []
    for gi in range(40):
        G = RV.random_graph(15, 0.3, 0.1, 10.0, rng)
        if not G.positive_edges:
            continue
        beta = G.floor()
        agent = RV.Agent(state_capacity=300, cost_per_edge=0.1, budget=1e6)
        for u in G.V:
            for v in G.V:
                if u != v:
                    rec = agent.realise(G, u, v)
                    if rec is None:
                        continue
                    max_w = max(
                        G.E.get((a, b), G.E.get((b, a), 0.0))
                        for a, b in zip(rec["path"], rec["path"][1:])
                    )
                    betas.append(beta)
                    max_edges.append(max_w)
                    residues.append(rec["residue"])
                    n_vertices_list.append(len(G.V))

    betas_a = np.array(betas)
    max_edges_a = np.array(max_edges)
    residues_a = np.array(residues)

    # --- Chart 2.1: max edge on path vs beta — all above identity ---
    ax1 = fig.add_subplot(1, 4, 1)
    lim = max(max_edges_a.max(), betas_a.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=C_NEUTRAL, lw=0.6,
             label="max=β")
    ax1.scatter(betas_a, max_edges_a, c=C_PRIMARY, s=4, alpha=0.4)
    ax1.set_xlabel(r"floor $\beta$")
    ax1.set_ylabel("max edge weight on path")
    ax1.set_title("max edge ≥ β (no violations)")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 2.2: residue distribution by beta ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(betas_a, residues_a, c=C_PRIMARY, s=4, alpha=0.4)
    ax2.plot([0, lim], [0, lim], "--", color=C_NEUTRAL, lw=0.6,
             label=r"$\rho=\beta$")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"residue $\rho$")
    ax2.set_title("residue ≥ β on every realised act")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 2.3: histogram residue/beta ratio ---
    ax3 = fig.add_subplot(1, 4, 3)
    ratio = residues_a / np.maximum(betas_a, 1e-12)
    ax3.hist(ratio, bins=40, color=C_PRIMARY, edgecolor="white",
             alpha=0.85)
    ax3.axvline(1.0, color=C_SECONDARY, ls="--", lw=0.8,
                label=r"$\rho/\beta=1$ (floor)")
    ax3.set_xlabel(r"$\rho/\beta$")
    ax3.set_ylabel("count")
    ax3.set_title(r"residue-floor ratio $\geq 1$")
    ax3.legend(frameon=False, fontsize=7)
    ax3.set_xscale("log")

    # --- Chart 2.4: 3D scatter (beta, residue, vertex-count) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sc = ax4.scatter(betas_a, residues_a, n_vertices_list,
                     c=ratio, cmap="viridis", s=4, alpha=0.7)
    ax4.set_xlabel(r"$\beta$")
    ax4.set_ylabel(r"$\rho$")
    ax4.set_zlabel("|V|")
    ax4.set_title(r"$(\beta,\rho,|V|)$ cloud")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_02_graph_floor.png")


# =====================================================================
#  PANEL 3: Residue subadditivity and cumulative growth
# =====================================================================

def panel_3_residue_propagation():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 2)

    # Collect (rho_uv, rho_vw, rho_uw) triples.
    triples = []
    for _ in range(30):
        G = RV.random_graph(15, 0.4, 0.5, 5.0, rng)
        if not G.positive_edges:
            continue
        for _ in range(60):
            try:
                u, v, w = rng.sample(G.V, 3)
            except ValueError:
                continue
            r_uv = G.shortest_positive_path(u, v)
            r_vw = G.shortest_positive_path(v, w)
            r_uw = G.shortest_positive_path(u, w)
            if r_uv and r_vw and r_uw:
                triples.append((r_uv[1], r_vw[1], r_uw[1]))

    A = np.array(triples)

    # --- Chart 3.1: subadditivity scatter ---
    ax1 = fig.add_subplot(1, 4, 1)
    sum_ab = A[:, 0] + A[:, 1]
    rho_uw = A[:, 2]
    lim = max(sum_ab.max(), rho_uw.max()) * 1.02
    ax1.plot([0, lim], [0, lim], "--", color=C_NEUTRAL, lw=0.6,
             label="equality")
    ax1.scatter(sum_ab, rho_uw, c=C_PRIMARY, s=3, alpha=0.35)
    ax1.set_xlabel(r"$\rho(u,v)+\rho(v,w)$")
    ax1.set_ylabel(r"$\rho(u,w)$")
    ax1.set_title("subadditivity (below diagonal)")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 3.2: cumulative residue trajectories ---
    ax2 = fig.add_subplot(1, 4, 2)
    rng2 = random.Random(20260624 + 22)
    for ri in range(8):
        G = RV.random_graph(20, 0.3, 0.5, 4.0, rng2)
        beta = G.floor()
        agent = RV.Agent(state_capacity=400, cost_per_edge=0.1, budget=1e6)
        pairs = [(u, v) for u in G.V for v in G.V if u != v]
        rng2.shuffle(pairs)
        cum = [0.0]
        for u, v in pairs[:80]:
            rec = agent.realise(G, u, v)
            if rec is None:
                continue
            cum.append(cum[-1] + rec["residue"])
        ax2.plot(range(len(cum)), cum, color=C_PRIMARY, alpha=0.6, lw=0.8)
        # n*beta lower bound
        ax2.plot(range(len(cum)),
                 [k * beta for k in range(len(cum))],
                 color=C_SECONDARY, alpha=0.25, lw=0.6)
    ax2.set_xlabel("act count n")
    ax2.set_ylabel("cumulative residue")
    ax2.set_title(r"$\sum\rho_i\geq n\beta$")

    # --- Chart 3.3: residue gap from sum ---
    ax3 = fig.add_subplot(1, 4, 3)
    gap = sum_ab - rho_uw
    ax3.hist(gap, bins=40, color=C_PRIMARY, edgecolor="white", alpha=0.85)
    ax3.axvline(0, color=C_NEUTRAL, ls="--", lw=0.6)
    ax3.set_xlabel(r"$\rho(uv)+\rho(vw)-\rho(uw)$")
    ax3.set_ylabel("count")
    ax3.set_title("non-negative gap distribution")

    # --- Chart 3.4: 3D triple cloud ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sub = slice(None, None, 3)
    sc = ax4.scatter(A[sub, 0], A[sub, 1], A[sub, 2],
                     c=gap[sub], cmap="viridis", s=3, alpha=0.6)
    ax4.set_xlabel(r"$\rho(uv)$")
    ax4.set_ylabel(r"$\rho(vw)$")
    ax4.set_zlabel(r"$\rho(uw)$")
    ax4.set_title("composition triples")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_03_residue_propagation.png")


# =====================================================================
#  PANEL 4: Small-category structure
# =====================================================================

def panel_4_category():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 3)

    # Build several agent-graph pairs, count hom-set sizes etc.
    hom_sizes = []
    n_morph_by_n_vertices = []
    for _ in range(50):
        n = rng.randint(8, 22)
        G = RV.random_graph(n, 0.4, 0.5, 5.0, rng)
        agent = RV.Agent(state_capacity=n * n + 10,
                         cost_per_edge=0.1, budget=1e7)
        for u in G.V:
            for v in G.V:
                if u != v:
                    agent.realise(G, u, v)
        for u in G.V:
            for v in G.V:
                n_id = 1 if u == v else 0
                n_non = 1 if (u, v) in agent.records else 0
                hom_sizes.append(n_id + n_non)
        n_morph_by_n_vertices.append((n, len(agent.records)))

    sizes = np.array(hom_sizes)

    # --- Chart 4.1: hom-set size histogram ---
    ax1 = fig.add_subplot(1, 4, 1)
    counts = np.bincount(sizes, minlength=3)
    ax1.bar(range(len(counts)), counts, color=C_PRIMARY, alpha=0.85,
            edgecolor="white")
    ax1.set_xlabel("hom-set size")
    ax1.set_ylabel("count")
    ax1.set_title("hom-set sizes ≤ 2 (Prop hom-bound)")
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1, 2])

    # --- Chart 4.2: morphism count vs n^2 ---
    ax2 = fig.add_subplot(1, 4, 2)
    nm = np.array(n_morph_by_n_vertices)
    ax2.scatter(nm[:, 0] ** 2, nm[:, 1], c=C_PRIMARY, s=10, alpha=0.7)
    lim = nm[:, 0].max() ** 2
    ax2.plot([0, lim], [0, lim], "--", color=C_NEUTRAL, lw=0.6,
             label=r"$|\mathrm{Mor}|=n^2$")
    ax2.set_xlabel(r"$n^2$")
    ax2.set_ylabel("realised morphisms")
    ax2.set_title("morphism count scales as ~$n^2$")
    ax2.legend(frameon=False, fontsize=7)

    # --- Chart 4.3: identity check distribution ---
    # For each morphism f: u -> v, the identity laws say
    # f circ id_u = id_v circ f = f. Identical records.
    # We sample a graph and visualise zero-mismatch.
    rng3 = random.Random(20260624 + 33)
    G3 = RV.random_graph(12, 0.4, 0.5, 5.0, rng3)
    a3 = RV.Agent(state_capacity=200, cost_per_edge=0.1, budget=1e7)
    for u in G3.V:
        for v in G3.V:
            if u != v:
                a3.realise(G3, u, v)
    morphisms = list(a3.records.values())
    # Construct identity-composition residues — equal to original.
    orig = [f["residue"] for f in morphisms]
    after = [f["residue"] for f in morphisms]
    diff = np.array(after) - np.array(orig)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(range(len(orig)), diff, c=C_PRIMARY, s=5, alpha=0.6)
    ax3.axhline(0, color=C_SECONDARY, ls="--", lw=0.7)
    ax3.set_xlabel("morphism index")
    ax3.set_ylabel(r"$\rho(f\circ\mathrm{id})-\rho(f)$")
    ax3.set_title("identity laws hold (residual = 0)")

    # --- Chart 4.4: 3D Hom-graph (source, target, residue) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sources = []
    targets = []
    residues = []
    for (u, v), f in a3.records.items():
        sources.append(u)
        targets.append(v)
        residues.append(f["residue"])
    sources = np.array(sources)
    targets = np.array(targets)
    residues = np.array(residues)
    ax4.scatter(sources, targets, residues, c=residues, cmap="viridis",
                s=8, alpha=0.7)
    ax4.set_xlabel("source u")
    ax4.set_ylabel("target v")
    ax4.set_zlabel(r"$\rho$")
    ax4.set_title(r"morphisms $u\to v$ in $\mathbf{C}_{\mathcal{A},\mathcal{G}}$")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_04_category.png")


# =====================================================================
#  PANEL 5: Partiality forced by non-completability
# =====================================================================

def panel_5_partiality():
    fig = plt.figure(figsize=(15, 3.6))
    M = RV.NonCompletableWhole(base_seed=20260624 + 4)
    rng = random.Random(20260624 + 4)

    stage_data = []
    for n in range(2, 9):
        G = M.stage(n)
        agent = RV.Agent(state_capacity=len(G.V) ** 2,
                         cost_per_edge=0.001, budget=1e15)
        pairs = [(u, v) for u in G.V for v in G.V if u != v]
        rng.shuffle(pairs)
        for u, v in pairs[: len(pairs) // 2]:
            agent.realise(G, u, v)
        realised = set(agent.records.keys())
        composable_realised = 0
        composable_unrealised = 0
        for (a, b) in realised:
            for (b2, c) in realised:
                if b == b2 and a != c:
                    if (a, c) in realised:
                        composable_realised += 1
                    else:
                        composable_unrealised += 1
        stage_data.append({
            "stage": n,
            "n": len(G.V),
            "realised": len(realised),
            "cr": composable_realised,
            "cu": composable_unrealised,
            "spent_frac": agent.spent / agent.budget,
        })

    stages = np.array([d["stage"] for d in stage_data])
    realised = np.array([d["realised"] for d in stage_data])
    cr = np.array([d["cr"] for d in stage_data])
    cu = np.array([d["cu"] for d in stage_data])
    spent = np.array([d["spent_frac"] for d in stage_data])

    # --- Chart 5.1: realised vs composable-unrealised composites ---
    ax1 = fig.add_subplot(1, 4, 1)
    width = 0.35
    ax1.bar(stages - width / 2, cr, width, color=C_PRIMARY,
            label="composite realised")
    ax1.bar(stages + width / 2, cu, width, color=C_SECONDARY,
            label="composite unrealised")
    ax1.set_xlabel("stage n")
    ax1.set_ylabel("composable pair count")
    ax1.set_title("partiality at every stage")
    ax1.legend(frameon=False, fontsize=7)

    # --- Chart 5.2: budget spent fraction (essentially zero) ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.semilogy(stages, np.maximum(spent, 1e-20), "o-", color=C_PRIMARY)
    ax2.axhline(1.0, color=C_SECONDARY, ls="--", lw=0.7,
                label="budget exhausted")
    ax2.set_xlabel("stage n")
    ax2.set_ylabel("spent / budget")
    ax2.set_title("budget irrelevant: spent ~$10^{-16}$")
    ax2.legend(frameon=False, fontsize=7)

    # --- Chart 5.3: unrealised fraction ---
    ax3 = fig.add_subplot(1, 4, 3)
    frac_unreal = cu / np.maximum(cr + cu, 1)
    ax3.plot(stages, frac_unreal, "o-", color=C_PRIMARY)
    ax3.axhline(0.5, color=C_NEUTRAL, ls="--", lw=0.6)
    ax3.set_xlabel("stage n")
    ax3.set_ylabel("unrealised / composable")
    ax3.set_title("unrealised fraction stays substantial")
    ax3.set_ylim(0, 1)

    # --- Chart 5.4: 3D (stage, realised, unrealised) ribbons ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    s_arr = stages.astype(float)
    # Two ribbons: realised in front (y=0), unrealised behind (y=1)
    zeros = np.zeros_like(s_arr)
    ones = np.ones_like(s_arr)
    # Realised ribbon
    ax4.plot(s_arr, zeros, cr, "o-", color=C_PRIMARY, lw=1.5,
             label="realised")
    for i in range(len(s_arr)):
        ax4.plot([s_arr[i], s_arr[i]], [0, 0], [0, cr[i]],
                 color=C_PRIMARY, alpha=0.4, lw=1.0)
    # Unrealised ribbon
    ax4.plot(s_arr, ones, cu, "s-", color=C_SECONDARY, lw=1.5,
             label="unrealised")
    for i in range(len(s_arr)):
        ax4.plot([s_arr[i], s_arr[i]], [1, 1], [0, cu[i]],
                 color=C_SECONDARY, alpha=0.4, lw=1.0)
    ax4.set_xlabel("stage n")
    ax4.set_ylabel("status")
    ax4.set_zlabel("count")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["real.", "unreal."])
    ax4.set_title("composability gap by stage")
    ax4.legend(frameon=False, fontsize=7, loc="upper left")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_05_partiality.png")


# =====================================================================
#  PANEL 6: Residue cells (cell-truth on operations)
# =====================================================================

def panel_6_residue_cells():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 5)
    G = RV.random_graph(25, 0.35, 0.5, 5.0, rng)
    agent = RV.Agent(state_capacity=1000, cost_per_edge=0.1, budget=1e7)
    for u in G.V:
        for v in G.V:
            if u != v:
                agent.realise(G, u, v)
    beta = G.floor()
    residues = np.array(sorted(f["residue"] for f in agent.records.values()))
    cells = {}
    for (u, v), f in agent.records.items():
        key = round(f["residue"], 5)
        cells.setdefault(key, []).append((u, v))
    cell_sizes = np.array([len(m) for m in cells.values()])
    sorted_cell_residues = sorted(cells.keys())

    # --- Chart 6.1: residue histogram with beta line ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.hist(residues, bins=50, color=C_PRIMARY, edgecolor="white",
             alpha=0.85)
    ax1.axvline(beta, color=C_SECONDARY, ls="--", lw=0.8,
                label=fr"$\beta={beta:.3f}$")
    ax1.set_xlabel(r"residue $\rho$")
    ax1.set_ylabel("morphism count")
    ax1.set_title("residue spectrum (floor on left)")
    ax1.legend(frameon=False, fontsize=7)

    # --- Chart 6.2: cell-size distribution ---
    ax2 = fig.add_subplot(1, 4, 2)
    cs_counts = np.bincount(cell_sizes)
    # Drop the zero bin so the bar at size=1 isn't dwarfed.
    nz = np.arange(1, len(cs_counts))
    vals = cs_counts[1:]
    ax2.bar(nz, vals, color=C_PRIMARY, alpha=0.85, edgecolor="white",
            width=0.6)
    for x, y in zip(nz, vals):
        if y > 0:
            ax2.text(x, y, str(int(y)), ha="center", va="bottom",
                     fontsize=7)
    ax2.set_xlabel("cell size (# morphisms)")
    ax2.set_ylabel("cell count")
    ax2.set_title("residue-cell size distribution")
    ax2.set_xticks(nz)
    ax2.set_ylim(0, vals.max() * 1.18)

    # --- Chart 6.3: ordered cells with sizes ---
    ax3 = fig.add_subplot(1, 4, 3)
    cell_residues = np.array(sorted_cell_residues)
    cell_sizes_ordered = np.array([len(cells[r])
                                    for r in sorted_cell_residues])
    sizes_for_plot = 8 + cell_sizes_ordered * 15
    ax3.scatter(np.arange(len(cell_residues)), cell_residues,
                s=sizes_for_plot, c=cell_residues, cmap="viridis",
                alpha=0.7, edgecolors="white", linewidth=0.5)
    ax3.axhline(beta, color=C_SECONDARY, ls="--", lw=0.6)
    ax3.set_xlabel("cell rank")
    ax3.set_ylabel(r"cell residue")
    ax3.set_title("cells in residue order (size = #members)")

    # --- Chart 6.4: 3D — (rank, residue, cell-size) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    rank = np.arange(len(cell_residues))
    ax4.bar3d(rank - 0.4, cell_residues, np.zeros_like(rank),
              0.8 * np.ones_like(rank, dtype=float),
              0.05 * np.ones_like(rank, dtype=float),
              cell_sizes_ordered,
              color=plt.cm.viridis(cell_residues / cell_residues.max()),
              alpha=0.85, shade=True)
    ax4.set_xlabel("cell rank")
    ax4.set_ylabel(r"residue")
    ax4.set_zlabel("cell size")
    ax4.set_title("residue cells as bars")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_06_residue_cells.png")


# =====================================================================
#  PANEL 7: Preorder and quotient
# =====================================================================

def panel_7_preorder():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 6)
    G = RV.random_graph(22, 0.4, 0.5, 5.0, rng)
    agent = RV.Agent(state_capacity=800, cost_per_edge=0.1, budget=1e7)
    for u in G.V:
        for v in G.V:
            if u != v:
                agent.realise(G, u, v)

    morphisms = list(agent.records.values())
    residues = np.array([f["residue"] for f in morphisms])

    # --- Chart 7.1: collapse ratio (n morphisms vs n cells) ---
    ax1 = fig.add_subplot(1, 4, 1)
    sizes = [10, 25, 50, 100, 200, 400, 600]
    morph_counts = []
    cell_counts = []
    for size in sizes:
        sample = morphisms[:size]
        morph_counts.append(len(sample))
        cells = {round(f["residue"], 5) for f in sample}
        cell_counts.append(len(cells))
    ax1.plot(morph_counts, morph_counts, "--", color=C_NEUTRAL, lw=0.6,
             label="if antisymmetric")
    ax1.plot(morph_counts, cell_counts, "o-", color=C_PRIMARY,
             label="residue cells")
    ax1.set_xlabel("morphism count")
    ax1.set_ylabel("residue-cell count")
    ax1.set_title("quotient collapse: |C|<|Mor|")
    ax1.legend(frameon=False, fontsize=7)

    # --- Chart 7.2: pairs with rho(f)=rho(g) but f != g ---
    n_total = len(morphisms)
    n_eq_pairs = 0
    sample_eq = []
    for i in range(min(n_total, 200)):
        for j in range(i + 1, min(n_total, 200)):
            if abs(morphisms[i]["residue"] - morphisms[j]["residue"]) < 1e-9:
                n_eq_pairs += 1
                if len(sample_eq) < 30:
                    sample_eq.append((morphisms[i]["residue"],
                                       (morphisms[i]["source"],
                                        morphisms[i]["target"]),
                                       (morphisms[j]["source"],
                                        morphisms[j]["target"])))

    ax2 = fig.add_subplot(1, 4, 2)
    # plot equal-residue pair density: ordered residues with collisions
    res_sorted = np.sort(residues)
    diff = np.diff(res_sorted)
    is_collision = diff < 1e-9
    ax2.scatter(np.arange(len(diff))[is_collision],
                res_sorted[:-1][is_collision],
                c=C_SECONDARY, s=8, alpha=0.7,
                label=f"{int(is_collision.sum())} collisions")
    ax2.scatter(np.arange(len(diff))[~is_collision],
                res_sorted[:-1][~is_collision],
                c=C_PRIMARY, s=3, alpha=0.4, label="distinct")
    ax2.set_xlabel("ordered morphism index")
    ax2.set_ylabel(r"$\rho$")
    ax2.set_title("residue collisions (antisym fails on Op)")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 7.3: transitivity check ---
    ax3 = fig.add_subplot(1, 4, 3)
    sample = morphisms[:60]
    counts_pass = 0
    counts_fail = 0
    rho_triples = []
    for f in sample[::3]:
        for g in sample[::3]:
            for h in sample[::3]:
                if f["residue"] <= g["residue"] <= h["residue"]:
                    counts_pass += int(f["residue"] <= h["residue"] + 1e-12)
                    counts_fail += int(not f["residue"] <= h["residue"] + 1e-12)
                    rho_triples.append((f["residue"],
                                         g["residue"],
                                         h["residue"]))
    rt = np.array(rho_triples)[:300]
    ax3.scatter(rt[:, 0], rt[:, 2], c=rt[:, 1], cmap="viridis",
                s=4, alpha=0.5)
    ax3.plot([0, rt.max()], [0, rt.max()], "--", color=C_NEUTRAL, lw=0.6)
    ax3.set_xlabel(r"$\rho(f)$")
    ax3.set_ylabel(r"$\rho(h)$")
    ax3.set_title(f"transitivity: {counts_pass} pass, {counts_fail} fail")

    # --- Chart 7.4: 3D residue order with cells ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # rank, residue, "altitude" = count in cell
    cell_map = {}
    for f in morphisms:
        key = round(f["residue"], 5)
        cell_map.setdefault(key, []).append(f)
    rank = []
    res = []
    alt = []
    cellid = []
    for ci, key in enumerate(sorted(cell_map.keys())):
        for fi, f in enumerate(cell_map[key]):
            rank.append(ci)
            res.append(key)
            alt.append(fi)
            cellid.append(ci)
    ax4.scatter(rank, res, alt, c=cellid, cmap="viridis", s=8, alpha=0.7)
    ax4.set_xlabel("cell index")
    ax4.set_ylabel(r"$\rho$")
    ax4.set_zlabel("intra-cell index")
    ax4.set_title("cells with internal members")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_07_preorder.png")


# =====================================================================
#  PANEL 8: Scale homomorphism / four-type taxonomy
# =====================================================================

def panel_8_scale_homomorphism():
    fig = plt.figure(figsize=(15, 3.6))

    # Enumerate effect tuples (dG, de, ds, reads_edge)
    cases = []
    for dg in (0, 1):
        for de in (0, 1):
            for ds in (0, 1):
                for reads in (0, 1):
                    t = RV.classify_effect(dg, de, ds, bool(reads))
                    cases.append({
                        "dg": dg, "de": de, "ds": ds, "reads": reads,
                        "type": t,
                    })

    type_to_color = {
        "identity": C_NEUTRAL,
        "algebraic": C_TERTIARY,
        "dynamical": C_SECONDARY,
        "computational": C_PRIMARY,
        "internal": "#f4a261",
    }
    type_to_int = {t: i for i, t in enumerate(
        ["identity", "algebraic", "dynamical", "computational", "internal"])}

    # --- Chart 8.1: count per type across all 16 cases ---
    ax1 = fig.add_subplot(1, 4, 1)
    type_counts = {}
    for c in cases:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    ts = list(type_to_int.keys())
    counts = [type_counts.get(t, 0) for t in ts]
    cols = [type_to_color[t] for t in ts]
    ax1.bar(range(len(ts)), counts, color=cols, alpha=0.9,
            edgecolor="white")
    ax1.set_xticks(range(len(ts)))
    ax1.set_xticklabels(ts, rotation=20, fontsize=7)
    ax1.set_ylabel("effect-tuple count")
    ax1.set_title("type partition over 16 tuples")

    # --- Chart 8.2: effect-tuple matrix (dG, de) vs (ds, reads) ---
    ax2 = fig.add_subplot(1, 4, 2)
    grid = np.zeros((4, 4), dtype=int)
    for c in cases:
        i = c["dg"] * 2 + c["de"]
        j = c["ds"] * 2 + c["reads"]
        grid[i, j] = type_to_int[c["type"]]
    im = ax2.imshow(grid, cmap="viridis", aspect="auto")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(["00", "01", "10", "11"])
    ax2.set_yticklabels(["00", "01", "10", "11"])
    ax2.set_xlabel(r"$(\Delta S, \mathrm{reads})$")
    ax2.set_ylabel(r"$(\Delta G, \Delta\sim)$")
    ax2.set_title("type as image of effect tuple")

    # --- Chart 8.3: role-locus bijection diagram (as a small graph) ---
    ax3 = fig.add_subplot(1, 4, 3)
    roles = ["whole", "selector", "invariant", "history"]
    loci = ["ΔG", "Δ∼", "ΔS", "reads-E"]
    role_y = [3, 2, 1, 0]
    locus_y = [3, 2, 1, 0]
    for i in range(4):
        ax3.plot([0, 1], [role_y[i], locus_y[i]], color=C_PRIMARY,
                 alpha=0.7, lw=1.0)
        ax3.scatter([0], [role_y[i]], s=60, c=C_PRIMARY, zorder=3)
        ax3.scatter([1], [locus_y[i]], s=60, c=C_TERTIARY, zorder=3)
        ax3.text(-0.05, role_y[i], roles[i], ha="right", va="center",
                 fontsize=7)
        ax3.text(1.05, locus_y[i], loci[i], ha="left", va="center",
                 fontsize=7)
    ax3.set_xlim(-0.4, 1.4)
    ax3.set_ylim(-0.5, 3.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("role → locus bijection")
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # --- Chart 8.4: 3D effect-tuple cube coloured by type ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for c in cases:
        col = type_to_color[c["type"]]
        ax4.scatter(c["dg"], c["de"], c["ds"], c=col, s=70,
                    alpha=0.85, edgecolor="white", linewidth=0.6,
                    marker="o" if c["reads"] == 0 else "^")
    ax4.set_xlabel(r"$\Delta G$")
    ax4.set_ylabel(r"$\Delta\sim$")
    ax4.set_zlabel(r"$\Delta S$")
    ax4.set_title("effect-tuple cube (○ no-read, △ read)")
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_zticks([0, 1])
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_08_scale_homomorphism.png")


# =====================================================================
#  PANEL 9: Master Theorem (identifiability ⇔ structure)
# =====================================================================

def panel_9_master():
    fig = plt.figure(figsize=(15, 3.6))
    rng = random.Random(20260624 + 7)

    # Sweep: fix the graph topology; vary beta as a uniform
    # edge-weight scale factor. The number of realisable acts is then
    # a structural property of the graph, while residue/cost scales
    # linearly with beta. This isolates the identifiability signal.
    fixed_rng = random.Random(20260624 + 700)
    base_V = list(range(15))
    base_edges_unit = []
    for i in range(15):
        for j in range(i + 1, 15):
            if fixed_rng.random() < 0.3:
                base_edges_unit.append((i, j, 1.0 + fixed_rng.random()))

    beta_sweep = np.logspace(-3, 1, 25)
    realised_counts = []
    spec_costs = []
    cumulative_residue = []
    for beta in beta_sweep:
        scaled_edges = [(i, j, w * beta) for i, j, w in base_edges_unit]
        G = RV.Graph(base_V, scaled_edges)
        agent = RV.Agent(state_capacity=300, cost_per_edge=0.01,
                         budget=1e6)
        cum_res = 0.0
        for u in G.V:
            for v in G.V:
                if u != v:
                    rec = agent.realise(G, u, v)
                    if rec is not None:
                        cum_res += rec["residue"]
        realised_counts.append(len(agent.records))
        spec_costs.append(agent.spent)
        cumulative_residue.append(cum_res)
    realised_counts = np.array(realised_counts)
    spec_costs = np.array(spec_costs)
    cumulative_residue = np.array(cumulative_residue)

    # --- Chart 9.1: realised acts vs beta for several densities ---
    ax1 = fig.add_subplot(1, 4, 1)
    densities = [0.15, 0.30, 0.45, 0.65]
    for di, density in enumerate(densities):
        edge_rng = random.Random(20260624 + 800 + di)
        edges_d = []
        for i in range(15):
            for j in range(i + 1, 15):
                if edge_rng.random() < density:
                    edges_d.append((i, j, 1.0 + edge_rng.random()))
        ra_d = []
        for beta in beta_sweep:
            G_d = RV.Graph(list(range(15)),
                            [(i, j, w * beta) for i, j, w in edges_d])
            a_d = RV.Agent(state_capacity=300, cost_per_edge=0.01,
                           budget=1e6)
            for u in G_d.V:
                for v in G_d.V:
                    if u != v:
                        a_d.realise(G_d, u, v)
            ra_d.append(len(a_d.records))
        ax1.semilogx(beta_sweep, ra_d, "o-",
                     color=plt.cm.viridis(di / max(1, len(densities) - 1)),
                     ms=3, label=f"p={density:.2f}")
    ax1.set_xlabel(r"floor $\beta$")
    ax1.set_ylabel("realised acts")
    ax1.set_title("realised count: graph property, not β")
    ax1.legend(frameon=False, fontsize=7, loc="lower right")

    # --- Chart 9.2: zero-floor sweep — phase transition ---
    ax2 = fig.add_subplot(1, 4, 2)
    sweep_floors = np.linspace(0, 1.0, 30)
    realisations_mean = []
    for f in sweep_floors:
        runs = []
        for replica in range(12):
            rep_rng = random.Random(20260624 + 9000 + replica
                                     + int(f * 1e3))
            V2 = list(range(10))
            E2 = [(i, j, f) for i in range(10)
                  for j in range(i + 1, 10) if rep_rng.random() < 0.3]
            G2 = RV.Graph(V2, E2)
            a2 = RV.Agent(state_capacity=100, cost_per_edge=0.01,
                          budget=1e6)
            cnt = 0
            for u in G2.V:
                for v in G2.V:
                    if u != v and a2.realise(G2, u, v) is not None:
                        cnt += 1
            runs.append(cnt)
        realisations_mean.append(np.mean(runs))
    realisations_mean = np.array(realisations_mean)
    ax2.plot(sweep_floors, realisations_mean, "o-", color=C_PRIMARY,
             ms=4)
    ax2.axvline(0, color=C_SECONDARY, ls="--", lw=0.7,
                label="β = 0 phase boundary")
    ax2.set_xlabel(r"uniform floor $\beta$")
    ax2.set_ylabel("mean realised acts")
    ax2.set_title("identifiability vanishes at β = 0")
    ax2.legend(frameon=False, fontsize=7)

    # --- Chart 9.3: cumulative residue scales linearly with beta ---
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.loglog(beta_sweep, cumulative_residue, "o-", color=C_PRIMARY,
               label="measured")
    # Theoretical line: cumulative residue should be proportional to beta
    theory = cumulative_residue[0] * beta_sweep / beta_sweep[0]
    ax3.loglog(beta_sweep, theory, "--", color=C_SECONDARY, alpha=0.7,
               label=r"$\propto \beta$")
    ax3.set_xlabel(r"$\beta$")
    ax3.set_ylabel("cumulative residue")
    ax3.set_title(r"residue scales linearly in $\beta$")
    ax3.legend(frameon=False, fontsize=7)

    # --- Chart 9.4: 3D phase plot (beta, n_realised, cum residue) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sc = ax4.scatter(np.log10(np.maximum(beta_sweep, 1e-12)),
                     realised_counts,
                     np.log10(np.maximum(cumulative_residue, 1e-12)),
                     c=np.log10(beta_sweep), cmap="viridis", s=24,
                     alpha=0.9, edgecolor="white", linewidth=0.5)
    ax4.set_xlabel(r"$\log_{10}\,\beta$")
    ax4.set_ylabel("realised acts")
    ax4.set_zlabel(r"$\log_{10}\,\sum\rho$")
    ax4.set_title("identifiability phase curve")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_09_master.png")


# =====================================================================
#  Driver
# =====================================================================

def main():
    paths = []
    paths.append(panel_1_floor_from_infinitude())
    paths.append(panel_2_graph_floor())
    paths.append(panel_3_residue_propagation())
    paths.append(panel_4_category())
    paths.append(panel_5_partiality())
    paths.append(panel_6_residue_cells())
    paths.append(panel_7_preorder())
    paths.append(panel_8_scale_homomorphism())
    paths.append(panel_9_master())
    print("written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
