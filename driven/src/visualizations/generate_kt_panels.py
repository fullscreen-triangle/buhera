"""Generate 6 publication panels for the knowledge-thermodynamics paper."""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

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
    "axes.edgecolor": "#222",
    "axes.labelcolor": "#111",
    "xtick.color": "#333",
    "ytick.color": "#333",
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
OUT_DIR = ROOT.parent / "long-grass" / "docs" / "knowledge-thermodynamics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.load(open(DATA / f"kt_{name}_results.json", "r", encoding="utf-8"))


def stamp(ax, label):
    ax.text(0.04, 0.95, label, transform=ax.transAxes, fontweight="bold", va="top", fontsize=11)


def stamp3d(ax, label):
    ax.text2D(0.04, 0.95, label, transform=ax.transAxes, fontweight="bold", va="top", fontsize=11)


def save(fig, name):
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ============================================================================
# Panel 1: Floor and Receiver Uncertainty
# ============================================================================

def panel_1_floor_uncertainty():
    R1 = load("01_floor")
    R3 = load("03_uncertainty")
    R4 = load("04_saturating_alloc")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) floor vs |K|
    ax1 = fig.add_subplot(1, 4, 1)
    Ks = [r["K"] for r in R1["records"]]
    bs = [r["beta"] for r in R1["records"]]
    ax1.loglog(Ks, bs, "o-", color=NAVY, lw=2, ms=7)
    ax1.set_xlabel(r"$|K|$"); ax1.set_ylabel(r"$\beta$")
    ax1.grid(alpha=0.3, which="both"); stamp(ax1, "(a)")

    # (b) uncertainty: sigma_K * sigma_Y vs hbar_R
    ax2 = fig.add_subplot(1, 4, 2)
    hbar = R3["summary"]["hbar_R"]
    products = [r["product"] for r in R3["records"]]
    ax2.scatter(range(len(products)), products, c=TEAL, s=14, alpha=0.7)
    ax2.axhline(hbar, color=CORAL, ls="--", lw=1.5, label=fr"$\hbar_R = {hbar}$")
    ax2.set_xlabel("methodology index"); ax2.set_ylabel(r"$\sigma_K \sigma_Y$")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3); stamp(ax2, "(b)")

    # (c) 3D saturating curve
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    sK = np.linspace(0.5, 5.0, 30)
    sY = np.linspace(0.5, 5.0, 30)
    K, Y = np.meshgrid(sK, sY)
    Z = K * Y
    ax3.plot_surface(K, Y, Z, cmap="viridis", alpha=0.7, edgecolor="none")
    # saturating contour at hbar = 4
    theta = np.linspace(0.05, 0.95, 50) * np.pi / 2
    sat_K = 2 * np.sqrt(np.tan(theta) / (1 + np.tan(theta)))
    sat_Y = 2 * np.sqrt(1 / (1 + np.tan(theta)))
    sat_Z = np.full_like(sat_K, 4.0)
    ax3.plot(sat_K, sat_Y, sat_Z, color=CORAL, lw=2.5, label="saturating")
    ax3.set_xlabel(r"$\sigma_K$"); ax3.set_ylabel(r"$\sigma_Y$"); ax3.set_zlabel(r"$\sigma_K\sigma_Y$")
    ax3.view_init(elev=24, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) saturating allocation: joint sigma vs ratio
    ax4 = fig.add_subplot(1, 4, 4)
    ratios = [r["ratio"] for r in R4["records"]]
    joints = [r["joint"] for r in R4["records"]]
    ax4.semilogx(ratios, joints, "o-", color=PURPLE, ms=6, lw=1.5)
    ax4.axhline(R4["summary"]["optimal_joint"], color=CORAL, ls="--", lw=1.5,
                label=fr"$\sqrt{{2\hbar_R}} = {R4['summary']['optimal_joint']:.2f}$")
    ax4.set_xlabel(r"ratio $\sigma_K/\sigma_Y$"); ax4.set_ylabel(r"$\sigma$")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, which="both"); stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_1_floor_uncertainty")


# ============================================================================
# Panel 2: Methodology Floor and Phase Lock
# ============================================================================

def panel_2_method_phase():
    R2 = load("02_method_floor")
    R5 = load("05_phase_lock")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) measured vs predicted (Banach fixed point)
    ax1 = fig.add_subplot(1, 4, 1)
    pred = [r["predicted"] for r in R2["records"]]
    meas = [r["measured"] for r in R2["records"]]
    ax1.plot(pred, meas, "o", color=NAVY, ms=8)
    ax1.plot([0, max(pred)], [0, max(pred)], "--", color="grey", lw=1)
    ax1.set_xlabel("predicted floor"); ax1.set_ylabel("measured floor")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.3); stamp(ax1, "(a)")

    # (b) convergence trajectories
    ax2 = fig.add_subplot(1, 4, 2)
    kappas = [0.1, 0.3, 0.5, 0.7, 0.9]
    sigma = 0.5
    for k in kappas:
        s = 50.0
        traj = [s]
        for _ in range(80):
            s = k * s + sigma * k
            traj.append(s)
        ax2.plot(traj, lw=1.8, label=fr"$\kappa={k}$")
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration"); ax2.set_ylabel(r"$s_n$")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3, which="both"); stamp(ax2, "(b)")

    # (c) 3D method-floor surface over (kappa, sigma)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    K = np.linspace(0.05, 0.95, 30)
    S = np.linspace(0.05, 1.5, 30)
    KK, SS = np.meshgrid(K, S)
    FF = SS * KK / (1 - KK)
    ax3.plot_surface(KK, SS, FF, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3.set_xlabel(r"$\kappa$"); ax3.set_ylabel(r"$\sigma$"); ax3.set_zlabel(r"$S_\flat(\mathfrak{M})$")
    ax3.view_init(elev=24, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) phase lock: counts of compile vs execute
    ax4 = fig.add_subplot(1, 4, 4)
    compile_counts = [r["n_compile"] for r in R5["records"]]
    execute_counts = [r["n_execute"] for r in R5["records"]]
    ax4.scatter(compile_counts, execute_counts, c=NAVY, s=18, alpha=0.7)
    ax4.set_xlabel("# COMPILE steps"); ax4.set_ylabel("# EXECUTE steps")
    ax4.grid(alpha=0.3); stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_2_method_phase")


# ============================================================================
# Panel 3: Cell-Type Equivalence and Disjointness
# ============================================================================

def panel_3_cell_types():
    R6 = load("06_cell_type")
    R7 = load("07_cell_disjoint")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) cell sizes per predicate
    ax1 = fig.add_subplot(1, 4, 1)
    cells = [r["cell"] for r in R6["records"]]
    sizes = [r["n_members"] for r in R6["records"]]
    colors = [NAVY, STEEL, TEAL, CORAL][:len(cells)]
    ax1.bar(range(len(cells)), sizes, color=colors, edgecolor="#222")
    ax1.set_xticks(range(len(cells)))
    ax1.set_xticklabels(cells, rotation=20, fontsize=9)
    ax1.set_ylabel("# expressions in cell")
    ax1.grid(alpha=0.3, axis="y"); stamp(ax1, "(a)")

    # (b) S = beta inside every cell (uniform bar)
    ax2 = fig.add_subplot(1, 4, 2)
    beta = R6["summary"]["beta"]
    ax2.bar(range(len(cells)), [beta] * len(cells), color=TEAL, edgecolor="#222")
    ax2.set_xticks(range(len(cells)))
    ax2.set_xticklabels(cells, rotation=20, fontsize=9)
    ax2.set_ylabel(r"$S$ (in-cell)")
    ax2.set_ylim(0, beta * 1.5)
    ax2.grid(alpha=0.3, axis="y"); stamp(ax2, "(b)")

    # (c) 3D cell intersection matrix
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    n_pairs = len(R7["records"])
    expected = [1 if r["expected_disjoint"] else 0 for r in R7["records"]]
    measured = [1 if r["measured_disjoint"] else 0 for r in R7["records"]]
    xs = np.arange(n_pairs)
    cs = [TEAL if e == m else CORAL for e, m in zip(expected, measured)]
    ax3.bar3d(xs, np.zeros_like(xs), np.zeros_like(xs),
              0.6 * np.ones_like(xs), 0.6 * np.ones_like(xs),
              np.array([1.0] * n_pairs),
              color=cs, edgecolor="#222", alpha=0.85, linewidth=0.3)
    ax3.set_xlabel("type pair index"); ax3.set_ylabel(""); ax3.set_zlabel("match")
    ax3.set_yticks([])
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) confusion: predicted-disjoint vs measured-disjoint
    ax4 = fig.add_subplot(1, 4, 4)
    conf = np.zeros((2, 2))
    for r in R7["records"]:
        conf[int(r["expected_disjoint"]), int(r["measured_disjoint"])] += 1
    im = ax4.imshow(conf, cmap="viridis")
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["overlap", "disjoint"])
    ax4.set_yticklabels(["overlap", "disjoint"])
    ax4.set_xlabel("measured"); ax4.set_ylabel("predicted")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, int(conf[i, j]), ha="center", va="center",
                     color="white" if conf[i, j] > conf.max() / 2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_3_cell_types")


# ============================================================================
# Panel 4: Domain Lattice
# ============================================================================

def panel_4_domain_lattice():
    R8 = load("08_domain_lattice")
    R9 = load("09_floor_monotone")
    R10 = load("10_mult_composition")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) lattice axiom pass rates
    ax1 = fig.add_subplot(1, 4, 1)
    axioms = ["idempotent\n(meet)", "idempotent\n(join)", "commut.\n(meet)", "commut.\n(join)", "absorpt.\n(m,j)", "absorpt.\n(j,m)"]
    counts = [
        sum(r["idempotent_meet"] for r in R8["records"]),
        sum(r["idempotent_join"] for r in R8["records"]),
        sum(r["commutative_meet"] for r in R8["records"]),
        sum(r["commutative_join"] for r in R8["records"]),
        sum(r["absorption_meet_join"] for r in R8["records"]),
        sum(r["absorption_join_meet"] for r in R8["records"]),
    ]
    n = len(R8["records"])
    rates = [c / n for c in counts]
    ax1.bar(range(len(axioms)), rates, color=NAVY, edgecolor="#222")
    ax1.set_xticks(range(len(axioms)))
    ax1.set_xticklabels(axioms, fontsize=8)
    ax1.set_ylabel("pass rate")
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.3, axis="y"); stamp(ax1, "(a)")

    # (b) floor monotonicity: chain floors
    ax2 = fig.add_subplot(1, 4, 2)
    for i, r in enumerate(R9["records"][:10]):
        floors = r["floors"]
        ax2.plot(range(len(floors)), floors, "o-", color=plt.cm.viridis(i / 10), alpha=0.7, ms=4)
    ax2.set_xlabel("position in refinement chain")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both"); stamp(ax2, "(b)")

    # (c) 3D multiplicative composition surface
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    b1_grid = np.linspace(0.5, 50, 25)
    b2_grid = np.linspace(0.5, 50, 25)
    B1, B2 = np.meshgrid(b1_grid, b2_grid)
    SIGMA = 100.0
    Z = B1 + B2 - B1 * B2 / SIGMA
    ax3.plot_surface(B1, B2, Z, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3.set_xlabel(r"$\beta_1$"); ax3.set_ylabel(r"$\beta_2$"); ax3.set_zlabel(r"$\beta_{12}$")
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) predicted vs measured composite floor
    ax4 = fig.add_subplot(1, 4, 4)
    pred = [r["predicted"] for r in R10["records"]]
    meas = [r["measured"] for r in R10["records"]]
    ax4.plot(pred, meas, "o", color=PURPLE, ms=4, alpha=0.7)
    ax4.plot([0, max(pred)], [0, max(pred)], "--", color="grey", lw=1)
    ax4.set_xlabel("predicted"); ax4.set_ylabel("measured")
    ax4.set_aspect("equal", adjustable="box")
    ax4.grid(alpha=0.3); stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_4_domain_lattice")


# ============================================================================
# Panel 5: Cascade Switching and Replication Bifurcation
# ============================================================================

def panel_5_cascade_replication():
    R11 = load("11_cascade_switching")
    R12 = load("12_replication_bifurcation")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) cascade: brute force vs greedy
    ax1 = fig.add_subplot(1, 4, 1)
    bf = [r["brute_force_floor"] for r in R11["records"]]
    gv = [r["greedy_floor"] for r in R11["records"]]
    ax1.plot(bf, gv, "o", color=NAVY, ms=5, alpha=0.75)
    lim = max(max(bf), max(gv))
    ax1.plot([0, lim], [0, lim], "--", color="grey", lw=1)
    ax1.set_xlabel("brute-force floor"); ax1.set_ylabel("greedy floor")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.3); stamp(ax1, "(a)")

    # (b) value-density priorities (synthetic)
    ax2 = fig.add_subplot(1, 4, 2)
    rng = np.random.default_rng(42)
    SIGMA = 100.0
    betas = np.sort(rng.uniform(2, 80, 12))
    costs = rng.uniform(1, 5, 12)
    rho = -np.log(1 - betas / SIGMA) / costs
    order = np.argsort(rho)[::-1]
    ax2.bar(range(len(rho)), rho[order], color=TEAL, edgecolor="#222")
    ax2.set_xlabel("methodology rank"); ax2.set_ylabel(r"selection priority $\rho_i$")
    ax2.grid(alpha=0.3, axis="y"); stamp(ax2, "(b)")

    # (c) 3D rho surface over (beta, c)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    B = np.linspace(1, 80, 25)
    C = np.linspace(0.5, 5, 25)
    BB, CC = np.meshgrid(B, C)
    Rho = -np.log(1 - BB / SIGMA) / CC
    ax3.plot_surface(BB, CC, Rho, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3.set_xlabel(r"$\beta_i$"); ax3.set_ylabel(r"$c_i$"); ax3.set_zlabel(r"$\rho_i$")
    ax3.view_init(elev=24, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) replication: weak floor vs n
    ax4 = fig.add_subplot(1, 4, 4)
    ns = [r["n"] for r in R12["records"]]
    weak = [r["weak_floor"] for r in R12["records"]]
    strong = [r["strong_floor_best"] for r in R12["records"]]
    ax4.plot(ns, weak, "o-", color=CORAL, label="weak (multiplicative)", lw=1.8, ms=5)
    ax4.plot(ns, strong, "s-", color=NAVY, label="strong (best single)", lw=1.8, ms=5)
    ax4.set_xlabel("n"); ax4.set_ylabel(r"$\beta$")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3); stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_5_cascade_replication")


# ============================================================================
# Panel 6: Knowledge Entropy and Federation
# ============================================================================

def panel_6_entropy_federation():
    R13 = load("13_know_entropy")
    R14 = load("14_federation")
    R15 = load("15_marginal_reduction")

    fig = plt.figure(figsize=PANEL_SIZE)

    # (a) H vs beta on log axes
    ax1 = fig.add_subplot(1, 4, 1)
    betas = [r["beta"] for r in R13["records"]]
    Hs = [r["H_measured"] for r in R13["records"]]
    ax1.semilogx(betas, Hs, "o-", color=NAVY, ms=6, lw=1.8)
    ax1.set_xlabel(r"$\beta$"); ax1.set_ylabel(r"$\mathfrak{H}(\mathcal{R})$")
    ax1.grid(alpha=0.3, which="both"); stamp(ax1, "(a)")

    # (b) federation floor: min vs individuals
    ax2 = fig.add_subplot(1, 4, 2)
    individual = [min(r["betas"]) for r in R14["records"]]
    fed = [r["beta_fed"] for r in R14["records"]]
    ax2.plot(individual, fed, "o", color=TEAL, ms=6, alpha=0.75)
    lim = max(max(individual), max(fed))
    ax2.plot([0, lim], [0, lim], "--", color="grey", lw=1)
    ax2.set_xlabel(r"$\min_i \beta_i$"); ax2.set_ylabel(r"$\beta_\mathrm{fed}$")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.3); stamp(ax2, "(b)")

    # (c) 3D: K_before, K_after, delta_K over increments
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    Kb = [r["K_before"] for r in R15["records"]]
    Ka = [r["K_after"] for r in R15["records"]]
    dK = [r["delta_Know"] for r in R15["records"]]
    ax3.scatter(Kb, Ka, dK, c=dK, cmap="viridis", s=18, alpha=0.85)
    ax3.set_xlabel(r"$\mathrm{Know}(F)$")
    ax3.set_ylabel(r"$\mathrm{Know}(F\cup\{\mathcal{R}_*\})$")
    ax3.set_zlabel(r"$\Delta\mathrm{Know}$")
    ax3.view_init(elev=22, azim=-60)
    stamp3d(ax3, "(c)")

    # (d) marginal reduction histogram
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.hist(dK, bins=20, color=PURPLE, edgecolor="#222")
    ax4.axvline(0, color=CORAL, ls="--", lw=1.5, label=r"$\Delta = 0$")
    ax4.set_xlabel(r"$\Delta\mathrm{Know}$"); ax4.set_ylabel("count")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, axis="y"); stamp(ax4, "(d)")

    plt.tight_layout()
    save(fig, "panel_6_entropy_federation")


def main():
    print("=" * 70)
    print("  KNOWLEDGE-THERMODYNAMICS PANELS")
    print("=" * 70)
    panel_1_floor_uncertainty()
    panel_2_method_phase()
    panel_3_cell_types()
    panel_4_domain_lattice()
    panel_5_cascade_replication()
    panel_6_entropy_federation()
    print(f"\n  All panels written to {OUT_DIR}")


if __name__ == "__main__":
    main()
