"""Generate the six COE panels.

Each panel is a 2x2 grid; one subplot is a 3D projection. White background,
green-family palette. Saves PDFs to long-grass/docs/computational-operations-equivalence/figures.

Panel layout:
  1. Time-Count Identity (V1) + Linearity (V2)
  2. Three Routes: Residue / Confinement / Negation (V3, V4, V5)
  3. Three-Route Equivalence + MTIC (V6, V7)
  4. Reproducibility / Sliding-Endpoint / Rewind / Monotone (V8-V11)
  5. API Substitutability + Cell Stability/Collision (V12-V14)
  6. Cross-architecture invariance (V15)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .common import (
    setup_style, save_panel, PANEL_DIR,
    GREEN_DARK, GREEN_MID, GREEN_LIGHT, GREEN_PALE,
    ACCENT_RED, ACCENT_AMBER, GREY,
)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _load(name: str) -> dict:
    with open(DATA_DIR / f"coe_{name}_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Panel 1: Time-Count Identity & Linearity
# ---------------------------------------------------------------------------
def panel_1_identity():
    setup_style()
    d1 = _load("01_time_count_identity")
    d2 = _load("02_linearity")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Recovered f vs reference
    ax = fig.add_subplot(2, 2, 1)
    rec1 = d1["records"]
    f_ref = d1["summary"]["f_ref"]
    fr = np.array([r["f_recovered"] for r in rec1])
    trials = np.arange(len(fr))
    ax.plot(trials, fr, "-o", color=GREEN_DARK, ms=4)
    ax.axhline(f_ref, color=ACCENT_RED, ls="--", lw=1.0, label="$f_{\\mathrm{ref}}$")
    ax.set_xlabel("trial")
    ax.set_ylabel("recovered $f$")
    ax.set_title("(a) Time-Count identity: $M(t)/t = f$")
    ax.legend()

    # (b) Relative error of f recovery
    ax = fig.add_subplot(2, 2, 2)
    err = np.array([r["rel_err"] for r in rec1])
    ax.semilogy(trials, err + 1e-18, "o-", color=GREEN_MID, ms=3)
    ax.axhline(1e-3, color=ACCENT_RED, ls="--", lw=1.0, label="tol")
    ax.set_xlabel("trial")
    ax.set_ylabel("relative error")
    ax.set_title("(b) Recovery residual")
    ax.legend()

    # (c) Linearity scatter
    ax = fig.add_subplot(2, 2, 3)
    rec2 = d2["records"]
    m1 = np.array([r["m1"] for r in rec2])
    m2 = np.array([r["m2"] for r in rec2])
    mu = np.array([r["m_union"] for r in rec2])
    ax.scatter(m1 + m2, mu, c=GREEN_DARK, s=30, alpha=0.85)
    lo, hi = (m1 + m2).min(), (m1 + m2).max()
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel("$M(o_1) + M(o_2)$")
    ax.set_ylabel("$M(o_1 \\cup o_2)$")
    ax.set_title("(c) Count linearity")
    ax.legend(loc="lower right")

    # (d) 3D: t-M-f surface — for fixed Q, slice through (t,M,f) space
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    Q_grid = np.arange(1, 200, 5)
    f_grid = np.linspace(1e3, 1e7, 50)
    Q, F = np.meshgrid(Q_grid, f_grid)
    T = Q / F
    ax.plot_surface(Q, np.log10(F), T, cmap="Greens",
                    edgecolor=GREEN_DARK, linewidth=0.15, alpha=0.85)
    ax.set_xlabel("$Q$")
    ax.set_ylabel("$\\log_{10} f$")
    ax.set_zlabel("$t = Q/f$")
    ax.set_title("(d) $t = Q/f$ surface")
    ax.view_init(elev=22, azim=-58)

    fig.suptitle("Panel 1 — Time-Count Identity and linearity",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_01_identity")


# ---------------------------------------------------------------------------
# Panel 2: Three Routes
# ---------------------------------------------------------------------------
def panel_2_three_routes():
    setup_style()
    d3 = _load("03_route_residue")
    d4 = _load("04_route_confinement")
    d5 = _load("05_route_negation")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Q_I vs Q_true (Residue)
    ax = fig.add_subplot(2, 2, 1)
    rec3 = d3["records"]
    Qt = np.array([r["Q_true"] for r in rec3])
    QI = np.array([r["Q_I"] for r in rec3])
    ax.scatter(Qt, QI, c=GREEN_DARK, s=30, alpha=0.85)
    lo, hi = Qt.min(), Qt.max()
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel(r"$Q_{\mathrm{true}}$")
    ax.set_ylabel(r"$Q_{\mathrm{I}}$ (residue)")
    ax.set_title("(a) Route I: residue")
    ax.legend(loc="lower right")

    # (b) Q_II vs Q_true (Confinement)
    ax = fig.add_subplot(2, 2, 2)
    rec4 = d4["records"]
    Qt = np.array([r["Q_true"] for r in rec4])
    QII = np.array([r["Q_II"] for r in rec4])
    ax.scatter(Qt, QII, c=GREEN_MID, s=30, alpha=0.85)
    ax.plot([Qt.min(), Qt.max()], [Qt.min(), Qt.max()], "--",
            color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel(r"$Q_{\mathrm{true}}$")
    ax.set_ylabel(r"$Q_{\mathrm{II}}$ (confinement)")
    ax.set_title("(b) Route II: confinement")
    ax.legend(loc="lower right")

    # (c) Q_III vs Q_true (Negation Fixed Point)
    ax = fig.add_subplot(2, 2, 3)
    rec5 = d5["records"]
    Qt = np.array([r["Q_true"] for r in rec5])
    QIII = np.array([r["Q_III"] for r in rec5])
    ax.scatter(Qt, QIII, c=GREEN_LIGHT, s=30, alpha=0.85, edgecolor=GREEN_DARK)
    ax.plot([Qt.min(), Qt.max()], [Qt.min(), Qt.max()], "--",
            color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel(r"$Q_{\mathrm{true}}$")
    ax.set_ylabel(r"$Q_{\mathrm{III}}$ (negation)")
    ax.set_title("(c) Route III: negation fixed point")
    ax.legend(loc="lower right")

    # (d) 3D: three routes converge to identical line
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    rec3 = d3["records"]
    rec4 = d4["records"]
    rec5 = d5["records"]
    n = min(len(rec3), len(rec4), len(rec5))
    QI = np.array([r["Q_I"] for r in rec3[:n]])
    QII = np.array([r["Q_II"] for r in rec4[:n]])
    QIII = np.array([r["Q_III"] for r in rec5[:n]])
    # Plot the three-route point-cloud
    ax.scatter(QI, QII, QIII, c=GREEN_DARK, s=40, alpha=0.9)
    # Diagonal line
    lo, hi = min(QI.min(), QII.min(), QIII.min()), max(QI.max(), QII.max(), QIII.max())
    ax.plot([lo, hi], [lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.5,
            label="$Q_I = Q_{II} = Q_{III}$")
    ax.set_xlabel(r"$Q_{\mathrm{I}}$")
    ax.set_ylabel(r"$Q_{\mathrm{II}}$")
    ax.set_zlabel(r"$Q_{\mathrm{III}}$")
    ax.set_title("(d) Three-route point cloud")
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=22, azim=-58)

    fig.suptitle("Panel 2 — Three independent routes to operation weight",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_02_three_routes")


# ---------------------------------------------------------------------------
# Panel 3: Three-Route Equivalence + MTIC
# ---------------------------------------------------------------------------
def panel_3_equivalence():
    setup_style()
    d6 = _load("06_three_route_equivalence")
    d7 = _load("07_mtic_equivalence")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Pairwise differences (all should be 0)
    ax = fig.add_subplot(2, 2, 1)
    rec6 = d6["records"]
    QI = np.array([r["Q_I"] for r in rec6])
    QII = np.array([r["Q_II"] for r in rec6])
    QIII = np.array([r["Q_III"] for r in rec6])
    trials = np.arange(len(QI))
    ax.plot(trials, QI - QII, "-", color=GREEN_DARK, label=r"$Q_I - Q_{II}$")
    ax.plot(trials, QII - QIII, "--", color=GREEN_MID, label=r"$Q_{II} - Q_{III}$")
    ax.plot(trials, QI - QIII, ":", color=GREEN_LIGHT, label=r"$Q_I - Q_{III}$")
    ax.axhline(0, color=ACCENT_RED, lw=0.8)
    ax.set_xlabel("trial")
    ax.set_ylabel("pairwise difference")
    ax.set_title("(a) Three-route pairwise residuals")
    ax.legend()

    # (b) Q_I vs Q_II coloured by Q_III (heatmap-style)
    ax = fig.add_subplot(2, 2, 2)
    sc = ax.scatter(QI, QII, c=QIII, cmap="Greens", s=36, alpha=0.95,
                    edgecolor=GREEN_DARK, linewidth=0.4)
    ax.plot([QI.min(), QI.max()], [QI.min(), QI.max()], "--",
            color=ACCENT_RED, lw=1.0)
    ax.set_xlabel(r"$Q_{\mathrm{I}}$")
    ax.set_ylabel(r"$Q_{\mathrm{II}}$")
    ax.set_title("(b) $Q_I = Q_{II} = Q_{III}$ confirmed")
    plt.colorbar(sc, ax=ax, shrink=0.8, label=r"$Q_{\mathrm{III}}$")

    # (c) MTIC: Q vs t, coloured by f
    ax = fig.add_subplot(2, 2, 3)
    rec7 = d7["records"]
    Q = np.array([r["Q"] for r in rec7])
    t = np.array([r["t"] for r in rec7])
    M = np.array([r["M"] for r in rec7])
    Q_from_t = np.array([r["Q_from_t"] for r in rec7])
    ax.scatter(Q, t, c=GREEN_DARK, s=30, alpha=0.85, label="(Q, t)")
    ax.scatter(Q, Q_from_t / 1_000_000.0, c=ACCENT_AMBER, s=14,
               label="$(Q, Q_\\mathrm{from\\,t}/f)$", marker="x")
    ax.set_xlabel("$Q$")
    ax.set_ylabel("$t = Q/f$")
    ax.set_title("(c) MTIC: $t = Q/f$")
    ax.legend()

    # (d) 3D: MTIC tetrahedron — Q, t, M, f relations as a 3D scatter
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.scatter(Q, M, t, c=GREEN_DARK, s=42, alpha=0.85,
               label="$(Q, M, t)$")
    ax.set_xlabel("$Q$")
    ax.set_ylabel("$M$")
    ax.set_zlabel("$t$")
    ax.set_title("(d) MTIC unit-conversion locus")
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=20, azim=-58)

    fig.suptitle("Panel 3 — Three-route equivalence and Mass-Time-Identity-Count",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_03_equivalence")


# ---------------------------------------------------------------------------
# Panel 4: Reproducibility / Sliding-Endpoint / Rewind / Monotone log
# ---------------------------------------------------------------------------
def panel_4_irreversibility():
    setup_style()
    d8 = _load("08_reproducibility")
    d9 = _load("09_sliding_endpoint")
    d10 = _load("10_rewind_as_forward")
    d11 = _load("11_monotone_log")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Reproducibility: bar chart of unique outputs across N runs
    ax = fig.add_subplot(2, 2, 1)
    s8 = d8["summary"]
    keys = ["outputs", "weights", "hashes"]
    vals = [1 if s8[f"identical_{k}"] else 0 for k in keys]
    ax.bar(keys, vals, color=GREEN_MID, edgecolor=GREEN_DARK)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("identical (1) / not (0)")
    ax.set_title(f"(a) Reproducibility ({s8['n_runs']} runs)")

    # (b) Sliding-endpoint: baseline vs truncated
    ax = fig.add_subplot(2, 2, 2)
    s9 = d9["summary"]
    cats = ["baseline\nreproducible", "truncated\nbreaks"]
    vals = [int(s9["baseline_reproducible"]), int(s9["truncated_breaks_reproducibility"])]
    ax.bar(cats, vals, color=[GREEN_DARK, ACCENT_AMBER], edgecolor=GREEN_DARK)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("holds (1)")
    ax.set_title("(b) Sliding-endpoint theorem")

    # (c) Rewind as forward: M_after > M_before
    ax = fig.add_subplot(2, 2, 3)
    rec10 = d10["records"]
    Mb = np.array([r["M_before"] for r in rec10])
    Ma = np.array([r["M_after"] for r in rec10])
    ax.scatter(Mb, Ma, c=GREEN_DARK, s=30, alpha=0.85)
    lo, hi = min(Mb.min(), Ma.min()), max(Mb.max(), Ma.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0,
            label="$M_{\\mathrm{after}} = M_{\\mathrm{before}}$")
    ax.set_xlabel(r"$M$ before rollback")
    ax.set_ylabel(r"$M$ after rollback")
    ax.set_title("(c) Rewind-as-forward")
    ax.legend(loc="lower right")

    # (d) 3D: monotone log surface — log step over (op_index, time-since-last)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    log = np.array(d11["log_sample"])
    op_idx = np.arange(len(log))
    deltas = np.diff(np.concatenate([[0], log]))
    # 3D bar of (op_idx, log_value, delta)
    ax.bar3d(op_idx, np.zeros_like(op_idx), np.zeros_like(op_idx),
             0.8, 0.5, deltas,
             color=GREEN_LIGHT, edgecolor=GREEN_DARK, alpha=0.85, shade=True)
    ax.set_xlabel("op index")
    ax.set_ylabel("")
    ax.set_zlabel(r"$\Delta M$ per op")
    ax.set_title("(d) Monotone log: every step $\\geq 0$")
    ax.view_init(elev=22, azim=-58)

    fig.suptitle("Panel 4 — Reproducibility, sliding endpoint, and monotone log",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_04_irreversibility")


# ---------------------------------------------------------------------------
# Panel 5: API substitutability + Cell stability/collision
# ---------------------------------------------------------------------------
def panel_5_substitutability():
    setup_style()
    d12 = _load("12_api_substitutability")
    d13 = _load("13_cell_stability")
    d14 = _load("14_cell_collision")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Substitutability: out_A vs out_B
    ax = fig.add_subplot(2, 2, 1)
    rec12 = d12["records"]
    a = np.array([r["out_A"] for r in rec12])
    b = np.array([r["out_B"] for r in rec12])
    ax.scatter(a, b, c=GREEN_DARK, s=30, alpha=0.85)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel("output of chain A")
    ax.set_ylabel("output of chain B")
    ax.set_title("(a) Two cascade chains, same fixed point")
    ax.legend(loc="lower right")

    # (b) Cell stability: existing weights unchanged
    ax = fig.add_subplot(2, 2, 2)
    rec13 = d13["records"]
    Nx = np.array([r["N_existing"] for r in rec13])
    stable = np.array([1 if r["stable"] else 0 for r in rec13])
    ax.scatter(Nx, stable, c=GREEN_MID, s=30)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel(r"$N_{\mathrm{existing}}$")
    ax.set_ylabel("stable (1) / not (0)")
    ax.set_title("(b) Cell stability under disjoint addition")

    # (c) Cell collision: confusion-matrix-like
    ax = fig.add_subplot(2, 2, 3)
    rec14 = d14["records"]
    exp = np.array([1 if r["expected"] else 0 for r in rec14])
    det = np.array([1 if r["detected"] else 0 for r in rec14])
    # Confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for e, d in zip(exp, det):
        cm[e, d] += 1
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else GREEN_DARK,
                    fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["disjoint", "collision"])
    ax.set_yticklabels(["disjoint", "collision"])
    ax.set_xlabel("detected")
    ax.set_ylabel("expected")
    ax.set_title("(c) Collision-detection confusion matrix")
    ax.grid(False)

    # (d) 3D: cell occupancy — cell index x kernel x weight
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    rng = np.random.default_rng(42)
    n_cells = 12
    n_kernels = 4
    cell_idx = np.arange(n_cells)
    for k in range(n_kernels):
        weights = rng.integers(50, 500, n_cells)
        # Kernel k owns cells {k, k+n_kernels, ...} — disjoint by construction
        owned = (cell_idx % n_kernels) == k
        # Plot only owned cells as bars
        for c in cell_idx[owned]:
            ax.bar3d(c, k, 0, 0.6, 0.4, weights[c],
                     color=[GREEN_DARK, GREEN_MID, GREEN_LIGHT, ACCENT_AMBER][k],
                     edgecolor=GREEN_DARK, alpha=0.85, shade=True)
    ax.set_xlabel("cell index")
    ax.set_ylabel("kernel")
    ax.set_zlabel("$Q$ (weight)")
    ax.set_title("(d) Disjoint cell ownership")
    ax.view_init(elev=22, azim=-60)

    fig.suptitle("Panel 5 — API substitutability and cell stability",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_05_substitutability")


# ---------------------------------------------------------------------------
# Panel 6: Cross-architecture invariance
# ---------------------------------------------------------------------------
def panel_6_cross_arch():
    setup_style()
    d15 = _load("15_cross_arch")

    fig = plt.figure(figsize=(11, 8.5))

    rec15 = d15["records"]
    archs = sorted({r["architecture"] for r in rec15})
    arch_colours = {a: c for a, c in zip(archs, [GREEN_DARK, GREEN_MID, GREEN_LIGHT, ACCENT_AMBER])}

    # (a) Q vs t per architecture
    ax = fig.add_subplot(2, 2, 1)
    for arch in archs:
        arr = [r for r in rec15 if r["architecture"] == arch]
        Q = [r["Q"] for r in arr]
        t = [r["t"] for r in arr]
        ax.scatter(Q, t, color=arch_colours[arch], s=28, alpha=0.85, label=arch)
    ax.set_xlabel("$Q$")
    ax.set_ylabel("$t$")
    ax.set_title("(a) Per-architecture $t = Q/f$")
    ax.legend(loc="upper left", fontsize=7)

    # (b) f per architecture
    ax = fig.add_subplot(2, 2, 2)
    arch_freqs = {arch: next(r["f"] for r in rec15 if r["architecture"] == arch)
                  for arch in archs}
    ax.bar(archs, [arch_freqs[a] for a in archs],
           color=[arch_colours[a] for a in archs], edgecolor=GREEN_DARK)
    ax.set_ylabel("reference $f$ (Hz)")
    ax.set_yscale("log")
    ax.set_title("(b) Architecture-local frequencies")
    ax.tick_params(axis="x", rotation=10)

    # (c) Recovery error per architecture
    ax = fig.add_subplot(2, 2, 3)
    for arch in archs:
        arr = [r for r in rec15 if r["architecture"] == arch]
        Q = np.array([r["Q"] for r in arr])
        Qr = np.array([r["Q_recovered"] for r in arr])
        err = np.abs(Q - Qr)
        ax.plot(np.arange(len(err)), err + 1e-3, "-o", ms=4,
                color=arch_colours[arch], label=arch)
    ax.set_xlabel("trial")
    ax.set_ylabel("$|Q - Q_{\\mathrm{recovered}}|$ (+1e-3)")
    ax.set_yscale("log")
    ax.set_title("(c) Per-architecture recovery error")
    ax.legend(fontsize=7)

    # (d) 3D: architecture-frequency-Q surface
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for ai, arch in enumerate(archs):
        arr = [r for r in rec15 if r["architecture"] == arch]
        xs = [ai] * len(arr)
        Qs = [r["Q"] for r in arr]
        ts = [r["t"] for r in arr]
        ax.scatter(xs, Qs, ts, color=arch_colours[arch], s=42, alpha=0.85,
                   label=arch)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, fontsize=7)
    ax.set_ylabel("$Q$")
    ax.set_zlabel("$t$")
    ax.set_title("(d) Cross-architecture invariance: $t = Q/f$")
    ax.legend(loc="upper right", fontsize=6)
    ax.view_init(elev=20, azim=-60)

    fig.suptitle("Panel 6 — Cross-architecture invariance",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_06_cross_arch")


def main():
    paths = []
    for fn in (panel_1_identity, panel_2_three_routes, panel_3_equivalence,
               panel_4_irreversibility, panel_5_substitutability, panel_6_cross_arch):
        p = fn()
        print(f"  saved {p}")
        paths.append(p)
    print(f"\nGenerated {len(paths)} COE panels into {PANEL_DIR}")
    return paths


if __name__ == "__main__":
    main()
