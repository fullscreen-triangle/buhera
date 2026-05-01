"""
Generate 6 publication panels for the unconstrained-subtasks-trajectory-completion paper.

Each panel: 4 charts in a row, white background, at least one 3D chart,
minimal text, all data-driven (no conceptual/text/table charts).

Reads JSON results from driven/data/uts_*.json and writes PNGs to
long-grass/docs/unconstrained-subtasks-trajectory/figures/.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

if sys.platform == "win32" and not getattr(sys, "_buhera_stdout_wrapped", False):
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys._buhera_stdout_wrapped = True
    except (ValueError, AttributeError):
        pass

plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 10,
})

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
CORAL = "#e76f51"
STEEL = "#457b9d"
GOLD = "#e9c46a"
PURPLE = "#6a4c93"
FOREST = "#264653"

PANEL_SIZE = (18, 4.2)

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT.parent / "long-grass" / "docs" / "unconstrained-subtasks-trajectory" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    with open(DATA / f"uts_{name}_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def stamp(ax, label: str) -> None:
    ax.text(0.04, 0.95, label, transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", fontsize=11)


def stamp3d(ax, label: str) -> None:
    ax.text2D(0.04, 0.95, label, transform=ax.transAxes, fontweight="bold",
              va="top", ha="left", fontsize=11)


def save(fig, name: str) -> Path:
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===========================================================================
# Panel 1: Floor and Information Bound
# ===========================================================================

def panel_1_floor():
    R1 = load("01_floor")
    R2 = load("02_info_bound")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Floor vs |K| log-log
    ax1 = fig.add_subplot(1, 4, 1)
    Ks = [r["K"] for r in R1["records"]]
    Sf = [r["S_floor"] for r in R1["records"]]
    ax1.loglog(Ks, Sf, "o-", color=NAVY, lw=2, ms=7)
    ax1.set_xlabel(r"Cognitive capacity $|K|$")
    ax1.set_ylabel(r"Floor $S_\flat$")
    ax1.grid(alpha=0.3, which="both")
    stamp(ax1, "(a)")

    # (b) Information content as bilinear surface vs (S_floor, eps) — heatmap
    ax2 = fig.add_subplot(1, 4, 2)
    # Group records by (S_floor, eps) into a grid
    floors = sorted(set(r["S_floor"] for r in R2["records"]))
    epss = sorted(set(r["eps"] for r in R2["records"]))
    Z = np.zeros((len(floors), len(epss)))
    for r in R2["records"]:
        i = floors.index(r["S_floor"])
        j = epss.index(r["eps"])
        Z[i, j] = r["predicted_bits"]
    im = ax2.imshow(Z, aspect="auto", origin="lower",
                    extent=[np.log10(min(epss)), np.log10(max(epss)),
                            np.log10(min(floors)), np.log10(max(floors))],
                    cmap="viridis")
    ax2.set_xlabel(r"$\log_{10}\epsilon$")
    ax2.set_ylabel(r"$\log_{10}S_\flat$")
    plt.colorbar(im, ax=ax2, label="bits", fraction=0.046, pad=0.04)
    stamp(ax2, "(b)")

    # (c) 3D surface of bits over (log10 S_floor, log10 eps)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    X, Y = np.meshgrid(np.log10(epss), np.log10(floors))
    ax3.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3.set_xlabel(r"$\log_{10}\epsilon$")
    ax3.set_ylabel(r"$\log_{10}S_\flat$")
    ax3.set_zlabel("bits")
    ax3.view_init(elev=24, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) Floor halving rate per capacity doubling
    ax4 = fig.add_subplot(1, 4, 4)
    capacities = np.array(Ks, dtype=float)
    floors_arr = np.array(Sf)
    ax4.semilogx(capacities, 100.0 / capacities, "--", color=CORAL, lw=1.5, label=r"$100/|K|$ prediction")
    ax4.semilogx(capacities, floors_arr, "o", color=NAVY, ms=8, label="measured")
    ax4.set_xlabel(r"$|K|$")
    ax4.set_ylabel(r"$S_\flat$")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, which="both")
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_1_floor")


# ===========================================================================
# Panel 2: Triple Equivalence and Optimal Representation
# ===========================================================================

def panel_2_triple_equiv():
    R3 = load("03_triple_equiv")
    R4 = load("04_optimal_rep")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Round-trip omega in vs omega out
    ax1 = fig.add_subplot(1, 4, 1)
    omega_in = [r["omega_in"] for r in R3["records"]]
    omega_out = [r["omega_out"] for r in R3["records"]]
    ax1.loglog(omega_in, omega_out, "o", color=TEAL, ms=7, alpha=0.75)
    diag = np.array([min(omega_in), max(omega_in)])
    ax1.loglog(diag, diag, "--", color="grey", lw=1)
    ax1.set_xlabel(r"$\omega_\mathrm{in}$")
    ax1.set_ylabel(r"$\omega_\mathrm{out}$")
    ax1.grid(alpha=0.3, which="both")
    stamp(ax1, "(a)")

    # (b) phase recovery
    ax2 = fig.add_subplot(1, 4, 2)
    phi_in = [r["phi_in"] for r in R3["records"]]
    phi_out = [r["phi_out"] for r in R3["records"]]
    ax2.plot(phi_in, phi_out, "o", color=NAVY, ms=7, alpha=0.85)
    ax2.plot([0, 2*math.pi], [0, 2*math.pi], "--", color="grey", lw=1)
    ax2.set_xlabel(r"$\phi_\mathrm{in}$")
    ax2.set_ylabel(r"$\phi_\mathrm{out}$")
    ax2.set_xticks(np.linspace(0, 2*math.pi, 5))
    ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.grid(alpha=0.3)
    stamp(ax2, "(b)")

    # (c) 3D scatter of round-trips: log10 omega_in, log10 omega_out, freq err
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    err = np.array([r["freq_rel_err"] for r in R3["records"]])
    sc = ax3.scatter(np.log10(omega_in), np.log10(omega_out), err,
                     c=err, cmap="inferno", s=30)
    ax3.set_xlabel(r"$\log_{10}\omega_\mathrm{in}$")
    ax3.set_ylabel(r"$\log_{10}\omega_\mathrm{out}$")
    ax3.set_zlabel("rel err")
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) Optimal representation cost histogram by output rep
    ax4 = fig.add_subplot(1, 4, 4)
    by_out = {"O": [], "C": [], "P": []}
    for r in R4["records"]:
        by_out[r["output_rep"]].append(r["optimal_cost"])
    positions = [0, 1, 2]
    parts = ax4.violinplot([by_out[k] for k in ("O", "C", "P")],
                            positions=positions, showmeans=True, showmedians=False)
    for body, c in zip(parts["bodies"], [STEEL, TEAL, CORAL]):
        body.set_facecolor(c)
        body.set_alpha(0.6)
        body.set_edgecolor("#333")
    ax4.set_xticks(positions)
    ax4.set_xticklabels(["O", "C", "P"])
    ax4.set_xlabel("Output representation")
    ax4.set_ylabel("Optimal cost")
    ax4.grid(alpha=0.3, axis="y")
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_2_triple_equiv_optrep")


# ===========================================================================
# Panel 3: Subtask Freedom and Local-Global Decoupling
# ===========================================================================

def panel_3_subtask():
    R5 = load("05_comp_mult")
    R6 = load("06_unconstrained_subtask")
    R7 = load("07_lg_decoupling")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Composition count vs n: 2^(n-1)
    ax1 = fig.add_subplot(1, 4, 1)
    ns = [r["n"] for r in R5["records"]]
    pred = [r["predicted"] for r in R5["records"]]
    meas = [r["measured"] for r in R5["records"]]
    ax1.semilogy(ns, pred, "--", color=CORAL, lw=2, label=r"$2^{n-1}$")
    ax1.semilogy(ns, meas, "o", color=NAVY, ms=8, label="measured")
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel("compositions")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3, which="both")
    stamp(ax1, "(a)")

    # (b) Bar chart of 19 expressions and their evaluated values
    ax2 = fig.add_subplot(1, 4, 2)
    vals = [r["value"] for r in R6["records"]]
    matches = [r["matches_target"] for r in R6["records"]]
    colors = [TEAL if m else CORAL for m in matches]
    ax2.bar(range(len(vals)), vals, color=colors, edgecolor="#333", linewidth=0.6)
    ax2.axhline(3.0, color="grey", lw=1, ls="--")
    ax2.set_xlabel("expression #")
    ax2.set_ylabel("evaluated value")
    ax2.set_xticks(range(0, len(vals), 4))
    ax2.set_ylim(0, 5)
    ax2.grid(alpha=0.3, axis="y")
    stamp(ax2, "(b)")

    # (c) 3D bar: local subtask values vs global preserved
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    locs = [r["eta_local"] for r in R7["records"]]
    globs = [r["global_value"] for r in R7["records"]]
    preserved = [1 if r["global_preserved"] else 0 for r in R7["records"]]
    xs = np.arange(len(locs))
    ax3.bar3d(xs, np.zeros_like(xs), np.zeros_like(xs),
              0.6 * np.ones_like(xs), 0.6 * np.ones_like(xs),
              np.array(globs),
              color=[TEAL if p else CORAL for p in preserved],
              alpha=0.85, edgecolor="#222", linewidth=0.3)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{int(v):+d}" if abs(v) < 100 else f"{int(v):+d}" for v in locs],
                        rotation=45, fontsize=8)
    ax3.set_xlabel(r"local $\eta$ value")
    ax3.set_zlabel("global value")
    ax3.set_ylabel("")
    ax3.set_yticks([])
    ax3.view_init(elev=22, azim=-65)
    stamp3d(ax3, "(c)")

    # (d) Global preservation rate vs local extreme magnitude
    ax4 = fig.add_subplot(1, 4, 4)
    abs_locs = np.abs(locs)
    diff = [abs(g - 3.0) for g in globs]
    ax4.semilogx(np.maximum(abs_locs, 1e-6), diff, "o-", color=PURPLE, ms=8, lw=2)
    ax4.set_xlabel(r"$|\eta|$ (local extreme)")
    ax4.set_ylabel(r"$|S^*_\mathrm{global} - \mathrm{target}|$")
    ax4.set_ylim(-0.0001, 0.0001)
    ax4.grid(alpha=0.3, which="both")
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_3_subtask_freedom")


# ===========================================================================
# Panel 4: Backward Navigation and Virtual Sub-States
# ===========================================================================

def panel_4_backward_virtual():
    R8 = load("08_backward_nav")
    R9 = load("09_virtual_substates")
    R10 = load("10_path_opacity")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Backward nav: depth vs measured steps
    ax1 = fig.add_subplot(1, 4, 1)
    depths = [r["depth"] for r in R8["records"]]
    expected = [r["expected_steps"] for r in R8["records"]]
    measured = [r["max_steps_observed"] for r in R8["records"]]
    ax1.plot(depths, expected, "--", color=CORAL, lw=1.5, label=r"$\log_3 N$")
    ax1.plot(depths, measured, "o", color=NAVY, ms=8, label="measured")
    ax1.set_xlabel("depth $k$")
    ax1.set_ylabel("backward steps")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)
    stamp(ax1, "(a)")

    # (b) Virtual fraction vs M
    ax2 = fig.add_subplot(1, 4, 2)
    Ms = [r["M"] for r in R9["records"]]
    vf = [r["virtual_fraction"] for r in R9["records"]]
    ax2.semilogx(Ms, vf, "o-", color=TEAL, ms=8, lw=2)
    ax2.axhline(1.0, color="grey", ls="--", lw=1)
    ax2.set_xlabel(r"decomposition magnitude $M$")
    ax2.set_ylabel("virtual fraction")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    stamp(ax2, "(b)")

    # (c) 3D scatter of sub-coordinate decompositions, coloured by virtuality
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    rng = np.random.default_rng(42)
    n = 600
    s_globals = rng.uniform(0.1, 0.9, n)
    M = 3.0
    s1 = rng.uniform(-M, M, n)
    s2 = rng.uniform(-M, M, n)
    s3 = 3 * s_globals - s1 - s2
    virtual = (s1 < 0) | (s1 > 1) | (s2 < 0) | (s2 > 1) | (s3 < 0) | (s3 > 1)
    ax3.scatter(s1[~virtual], s2[~virtual], s3[~virtual], c=TEAL, s=10, alpha=0.5, label="physical")
    ax3.scatter(s1[virtual], s2[virtual], s3[virtual], c=CORAL, s=10, alpha=0.5, label="virtual")
    ax3.set_xlabel(r"$s_1$")
    ax3.set_ylabel(r"$s_2$")
    ax3.set_zlabel(r"$s_3$")
    ax3.legend(frameon=False, fontsize=8, loc="upper left")
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) Path opacity: 100 trials, all indistinguishable
    ax4 = fig.add_subplot(1, 4, 4)
    op_rate = R10["summary"]["opacity_rate"]
    ax4.bar([0], [op_rate], color=PURPLE, edgecolor="#222")
    ax4.set_ylim(0, 1.05)
    ax4.set_xticks([0])
    ax4.set_xticklabels(["distinct paths,\nshared endpoint"])
    ax4.set_ylabel("opacity rate")
    ax4.grid(alpha=0.3, axis="y")
    ax4.text(0, op_rate + 0.02, f"{op_rate:.2f}", ha="center", fontsize=11)
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_4_backward_virtual")


# ===========================================================================
# Panel 5: Catalysts and Cascade Saturation
# ===========================================================================

def panel_5_catalysts():
    R11 = load("11_multiplicativity")
    R12 = load("12_cascade_saturation")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Heatmap: composite kappa over (kappa_1, kappa_2)
    ax1 = fig.add_subplot(1, 4, 1)
    KS = sorted(set(r["kappa_1"] for r in R11["records"]))
    Z = np.zeros((len(KS), len(KS)))
    for r in R11["records"]:
        i = KS.index(r["kappa_1"])
        j = KS.index(r["kappa_2"])
        Z[i, j] = r["measured"]
    im = ax1.imshow(Z, origin="lower", cmap="viridis",
                    extent=[KS[0], KS[-1], KS[0], KS[-1]])
    ax1.set_xlabel(r"$\kappa_2$")
    ax1.set_ylabel(r"$\kappa_1$")
    plt.colorbar(im, ax=ax1, label=r"$\kappa(\gamma_1\diamond\gamma_2)$",
                 fraction=0.046, pad=0.04)
    stamp(ax1, "(a)")

    # (b) Predicted vs measured composite kappa
    ax2 = fig.add_subplot(1, 4, 2)
    pred = [r["predicted"] for r in R11["records"]]
    meas = [r["measured"] for r in R11["records"]]
    ax2.plot(pred, meas, "o", color=NAVY, ms=6, alpha=0.75)
    ax2.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("measured")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    stamp(ax2, "(b)")

    # (c) 3D surface of composite kappa
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    K1, K2 = np.meshgrid(KS, KS)
    Zs = 1 - (1 - K1) * (1 - K2)
    ax3.plot_surface(K1, K2, Zs, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3.set_xlabel(r"$\kappa_1$")
    ax3.set_ylabel(r"$\kappa_2$")
    ax3.set_zlabel(r"$\kappa$")
    ax3.view_init(elev=24, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) Cascade residual decay for the four cascade types
    ax4 = fig.add_subplot(1, 4, 4)
    rec = R12["records"]
    short_names = ["const 0.1", "harmonic 1/i", "geom 2^-i", "zero"]
    residuals = [max(r["residual_n"], 1e-300) for r in rec]
    ax4.bar(range(len(rec)), [-math.log10(r) for r in residuals],
            color=[CORAL, GOLD, STEEL, NAVY], edgecolor="#222")
    ax4.set_xticks(range(len(rec)))
    ax4.set_xticklabels(short_names, rotation=20, fontsize=9)
    ax4.set_ylabel(r"$-\log_{10}$ residual")
    ax4.grid(alpha=0.3, axis="y")
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_5_catalysts_cascade")


# ===========================================================================
# Panel 6: Recursive Structure and Strict Hierarchy
# ===========================================================================

def panel_6_recursive_hierarchy():
    R13 = load("13_recursive_mult")
    R14 = load("14_strict_hierarchy")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) Recursive triple count (lower bound 3 * 4^(d-1))
    ax1 = fig.add_subplot(1, 4, 1)
    ds = [r["depth"] for r in R13["records"]]
    bounds = [r["lower_bound"] for r in R13["records"]]
    measured = [r["measured_count"] for r in R13["records"]]
    ax1.semilogy(ds, bounds, "--", color=CORAL, lw=1.5, label=r"$3\cdot 4^{d-1}$")
    ax1.semilogy(ds, measured, "o", color=NAVY, ms=8, label="measured")
    ax1.set_xlabel("depth $d$")
    ax1.set_ylabel("recursive triples")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3, which="both")
    stamp(ax1, "(a)")

    # (b) Traversal counts per class vs N
    ax2 = fig.add_subplot(1, 4, 2)
    Ns = [r["N"] for r in R14["records"]]
    classes = ["C_0", "C_1", "C_poly", "C_nav", "C_hard"]
    colors = [NAVY, STEEL, TEAL, GOLD, CORAL]
    for cls, c in zip(classes, colors):
        ys = [max(r["traversals"][cls], 1) for r in R14["records"]]
        ax2.loglog(Ns, ys, "o-", color=c, lw=2, ms=7, label=cls.replace("_", r"_\text{"))
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel("traversals")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3, which="both")
    stamp(ax2, "(b)")

    # (c) 3D bar: traversal count over (N, class)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    xs = []
    ys = []
    zs = []
    for i, r in enumerate(R14["records"]):
        for j, cls in enumerate(classes):
            xs.append(i)
            ys.append(j)
            v = r["traversals"][cls]
            zs.append(math.log10(max(v, 1)))
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    cmap = plt.get_cmap("viridis")
    cs = cmap((zs - zs.min()) / max(zs.max() - zs.min(), 1e-9))
    ax3.bar3d(xs, ys, np.zeros_like(zs),
              0.6, 0.6, zs, color=cs, alpha=0.85, edgecolor="#222", linewidth=0.3)
    ax3.set_xticks(range(len(R14["records"])))
    ax3.set_xticklabels([str(r["N"]) for r in R14["records"]], rotation=30, fontsize=8)
    ax3.set_yticks(range(len(classes)))
    ax3.set_yticklabels(classes, fontsize=8)
    ax3.set_zlabel(r"$\log_{10}$ traversals")
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) Hierarchy gap C_hard / C_1 vs N
    ax4 = fig.add_subplot(1, 4, 4)
    gap = [r["traversals"]["C_hard"] / max(r["traversals"]["C_1"], 1) for r in R14["records"]]
    pred = [r["N"] / max(int(math.log(r["N"], 3)), 1) for r in R14["records"]]
    ax4.loglog(Ns, gap, "o-", color=CORAL, lw=2, ms=8, label=r"measured $C_\text{hard}/C_1$")
    ax4.loglog(Ns, pred, "--", color=NAVY, lw=1.5, label=r"$N/\log_3 N$")
    ax4.set_xlabel(r"$N$")
    ax4.set_ylabel("hierarchy gap")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, which="both")
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_6_recursive_hierarchy")


def main():
    print("=" * 70)
    print("  UTS PAPER PANELS")
    print("=" * 70)
    panel_1_floor()
    panel_2_triple_equiv()
    panel_3_subtask()
    panel_4_backward_virtual()
    panel_5_catalysts()
    panel_6_recursive_hierarchy()
    print(f"\n  All panels written to {OUT_DIR}")


if __name__ == "__main__":
    main()
