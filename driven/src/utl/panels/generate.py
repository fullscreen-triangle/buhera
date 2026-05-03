"""Generate the six UTL panels.

Each panel is a 2x2 grid; one subplot is a 3D projection. White background,
blue-family palette. Saves PDFs to long-grass/docs/os-throughput-law/figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from .common import (
    setup_style, save_panel, PANEL_DIR,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ACCENT_RED, ACCENT_AMBER, GREY,
)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _load(name: str) -> dict:
    with open(DATA_DIR / f"utl_{name}_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Panel 1: Universal Transport Law calibration
# ---------------------------------------------------------------------------
def panel_1_universal_law():
    setup_style()
    d = _load("01_universal_law")
    records = d["records"]
    pred = np.array([r["predicted_tp_inv"] for r in records])
    meas = np.array([r["measured_tp_inv"] for r in records])
    rel_err = np.array([r["rel_err"] for r in records])

    fig = plt.figure(figsize=(11, 8.5))

    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(pred, meas, c=BLUE_DARK, s=42, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    lo, hi = min(pred.min(), meas.min()), max(pred.max(), meas.max())
    ax.plot([lo, hi], [lo, hi], color=ACCENT_RED, ls="--", lw=1.0, label="$y=x$")
    ax.set_xlabel("predicted $TP^{-1}$")
    ax.set_ylabel("measured $TP^{-1}$")
    ax.set_title("(a) Universal-law calibration")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(np.arange(len(rel_err)), rel_err, "-o", color=BLUE_MID, ms=4)
    ax.axhline(0.10, color=ACCENT_RED, ls="--", lw=1.0, label="tol $=0.10$")
    ax.set_xlabel("trial")
    ax.set_ylabel("relative error")
    ax.set_title("(b) Per-trial residual")
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.hist(rel_err, bins=12, color=BLUE_LIGHT, edgecolor=BLUE_DARK)
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title(f"(c) Residual distribution (max = {rel_err.max():.4f})")

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    n = 30
    tau_grid = np.linspace(0.5, 5.0, n)
    g_grid = np.linspace(0.1, 1.0, n)
    T, G = np.meshgrid(tau_grid, g_grid)
    Z = T * G
    ax.plot_surface(T, G, Z, cmap="Blues", edgecolor=BLUE_DARK,
                    linewidth=0.2, alpha=0.85)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$g$")
    ax.set_zlabel(r"$\tau \cdot g$")
    ax.set_title(r"(d) Integrand surface $\tau \cdot g$")
    ax.view_init(elev=22, azim=-58)

    fig.suptitle("Panel 1 — Universal OS Transport Law calibration",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_01_universal_law")


# ---------------------------------------------------------------------------
# Panel 2: Queueing specialisations (M/M/1, Jackson, Cascade)
# ---------------------------------------------------------------------------
def panel_2_queueing():
    setup_style()
    d2 = _load("02_mm1")
    d3 = _load("03_jackson")
    d4 = _load("04_cascade")

    fig = plt.figure(figsize=(11, 8.5))

    ax = fig.add_subplot(2, 2, 1)
    rec2 = d2["records"]
    rho = np.array([r["rho"] for r in rec2])
    mm1 = np.array([r["tp_inv_mm1"] for r in rec2])
    uni = np.array([r["tp_inv_universal"] for r in rec2])
    order = np.argsort(rho)
    ax.plot(rho[order], mm1[order], color=BLUE_DARK, lw=1.6, label="$M/M/1$")
    ax.scatter(rho[order], uni[order], color=ACCENT_AMBER, s=28, label="UTL")
    ax.set_xlabel(r"$\rho$ (utilisation)")
    ax.set_ylabel(r"$TP^{-1}$")
    ax.set_title("(a) M/M/1 specialisation")
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    rec3 = d3["records"]
    uni3 = np.array([r["universal"] for r in rec3])
    jak3 = np.array([r["jackson"] for r in rec3])
    ax.scatter(uni3, jak3, color=BLUE_DARK, s=30, alpha=0.85)
    lo, hi = min(uni3.min(), jak3.min()), max(uni3.max(), jak3.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel(r"$TP^{-1}$ (UTL)")
    ax.set_ylabel(r"$TP^{-1}$ (Jackson)")
    ax.set_title("(b) Jackson independence")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(2, 2, 3)
    rec4 = d4["records"]
    k = np.array([r["k"] for r in rec4])
    cs = np.array([r["cascade_sum"] for r in rec4])
    un = np.array([r["universal"] for r in rec4])
    order = np.argsort(k)
    ax.plot(k[order], cs[order], "-o", color=BLUE_DARK, ms=4, label="cascade sum")
    ax.scatter(k[order], un[order], color=ACCENT_AMBER, s=24, label="UTL")
    ax.set_xlabel("cascade depth")
    ax.set_ylabel(r"$TP^{-1}$")
    ax.set_title("(c) Cascade serialisation")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    rho_g = np.linspace(0.05, 0.95, 30)
    mu_g = np.linspace(0.5, 5.0, 30)
    R, M = np.meshgrid(rho_g, mu_g)
    W = R / (M * (1 - R))
    ax.plot_surface(R, M, np.log10(W + 1e-3), cmap="Blues",
                    edgecolor=BLUE_DARK, linewidth=0.2, alpha=0.85)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\mu$")
    ax.set_zlabel(r"$\log_{10}\,W$")
    ax.set_title("(d) M/M/1 wait surface")
    ax.view_init(elev=24, azim=-50)

    fig.suptitle("Panel 2 — Queueing specialisations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_02_queueing")


# ---------------------------------------------------------------------------
# Panel 3: Five regimes
# ---------------------------------------------------------------------------
def panel_3_regimes():
    setup_style()
    d8 = _load("08_bifurcation")
    REGIMES = ["turbulent", "aperture", "cascade", "coherent", "phase_locked"]
    bounds = [0.0, 0.3, 0.5, 0.8, 0.95, 1.0]

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Regime partition strip
    ax = fig.add_subplot(2, 2, 1)
    R_grid = np.linspace(0, 1, 1001)
    colours = [BLUE_PALE, BLUE_LIGHT, BLUE_MID, BLUE_DARK, "#031a52"]
    for i in range(len(bounds) - 1):
        ax.axvspan(bounds[i], bounds[i + 1], color=colours[i], alpha=0.55,
                   label=REGIMES[i])
    for x in (0.3, 0.5, 0.8, 0.95):
        ax.axvline(x, color=GREY, ls=":", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("$R$")
    ax.set_title("(a) Regime partition of $R$")
    ax.legend(loc="upper left", fontsize=7)

    # (b) Regime occupancy bar
    ax = fig.add_subplot(2, 2, 2)
    counts = [int(((R_grid >= bounds[i]) & (R_grid < bounds[i + 1])).sum())
              for i in range(len(bounds) - 1)]
    counts[-1] = int(((R_grid >= bounds[-2]) & (R_grid <= bounds[-1])).sum())
    ax.bar(REGIMES, counts, color=BLUE_MID, edgecolor=BLUE_DARK)
    ax.set_ylabel("count")
    ax.set_title("(b) Regime occupancy on uniform $R$")
    ax.tick_params(axis="x", rotation=15)

    # (c) Bifurcation: R_pred vs K
    ax = fig.add_subplot(2, 2, 3)
    rec8 = d8["records"]
    K = np.array([r["K"] for r in rec8])
    R_pred = np.array([r["R_pred"] for r in rec8])
    R_meas = np.array([r["R_meas"] for r in rec8])
    order = np.argsort(K)
    ax.plot(K[order], R_pred[order], "-", color=BLUE_DARK, lw=1.4, label="predicted")
    ax.scatter(K[order], R_meas[order], color=ACCENT_AMBER, s=20, label="measured")
    for x, lbl in zip((0.3, 0.5, 0.8, 0.95),
                      ("turb", "ap", "cas", "coh")):
        ax.axhline(x, color=GREY, ls=":", lw=0.6)
    ax.set_xlabel("coupling $K$")
    ax.set_ylabel("$R$")
    ax.set_title("(c) Bifurcation: $R(K)$")
    ax.legend()

    # (d) 3D: regime cone — R x perturbation x coherence
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    R_g = np.linspace(0.05, 0.99, 40)
    eps = np.linspace(0.0, 0.4, 40)
    Rg, Eg = np.meshgrid(R_g, eps)
    coh = (1 - Eg) * np.tanh(8 * (Rg - 0.5))
    ax.plot_surface(Rg, Eg, coh, cmap="Blues",
                    edgecolor=BLUE_DARK, linewidth=0.15, alpha=0.85)
    ax.set_xlabel("$R$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_zlabel("coherence")
    ax.set_title("(d) Regime cone")
    ax.view_init(elev=22, azim=-60)

    fig.suptitle("Panel 3 — Five operating regimes",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_03_regimes")


# ---------------------------------------------------------------------------
# Panel 4: Critical slowing & load indicator
# ---------------------------------------------------------------------------
def panel_4_critical():
    setup_style()
    d9 = _load("09_critical_slowing")
    d10 = _load("10_load_indicator")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) tau_relax vs |R - R_b|
    ax = fig.add_subplot(2, 2, 1)
    rec9 = d9["records"]
    R = np.array([r["R"] for r in rec9])
    R_b = np.array([r["R_b"] for r in rec9])
    tau = np.array([r["tau_relax"] for r in rec9])
    pred = np.array([r["predicted"] for r in rec9])
    dR = np.abs(R - R_b)
    ax.loglog(dR + 1e-6, tau, "o", color=BLUE_DARK, ms=4, label="measured")
    ax.loglog(dR + 1e-6, pred, "x", color=ACCENT_AMBER, ms=5, label="predicted")
    x = np.logspace(-3, 0, 50)
    ax.loglog(x, 1.0 / (x + 1e-6), "--", color=ACCENT_RED, lw=1.0,
              label=r"$\propto 1/|R-R_b|$")
    ax.set_xlabel("$|R - R_b|$")
    ax.set_ylabel(r"$\tau_\mathrm{relax}$")
    ax.set_title("(a) Critical slowing")
    ax.legend(loc="best")

    # (b) Load indicator: estimated d vs true d
    ax = fig.add_subplot(2, 2, 2)
    rec10 = d10["records"]
    d_true = np.array([r["d_true"] for r in rec10])
    d_est = np.array([r["d_estimated"] for r in rec10])
    correct = np.array([r["correct"] for r in rec10])
    colours = [BLUE_DARK if c else ACCENT_RED for c in correct]
    ax.scatter(d_true, d_est, c=colours, s=30, alpha=0.85)
    lo, hi = min(d_true.min(), d_est.min()), max(d_true.max(), d_est.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_RED, lw=1.0, label="$y=x$")
    ax.set_xlabel("true distance from $R_b$")
    ax.set_ylabel("estimated distance")
    ax.set_title("(b) Load indicator $d \\to R_b$")
    ax.legend(loc="lower right")

    # (c) Per-trial relative error
    ax = fig.add_subplot(2, 2, 3)
    rel = np.array([r["rel_err"] for r in rec10])
    ax.plot(np.arange(len(rel)), rel, "-o", color=BLUE_MID, ms=3)
    ax.set_xlabel("trial")
    ax.set_ylabel("relative error")
    ax.set_title(f"(c) Indicator residual (mean = {rel.mean():.3f})")

    # (d) 3D: tau_relax surface over R x perturbation
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    R_grid = np.linspace(0.1, 0.95, 40)
    pert = np.linspace(0.01, 0.3, 40)
    Rg, Pg = np.meshgrid(R_grid, pert)
    Tg = 1.0 / (np.abs(Rg - 0.5) + Pg + 1e-3)
    ax.plot_surface(Rg, Pg, np.log10(Tg), cmap="Blues",
                    edgecolor=BLUE_DARK, linewidth=0.15, alpha=0.85)
    ax.set_xlabel("$R$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_zlabel(r"$\log_{10}\tau_\mathrm{relax}$")
    ax.set_title("(d) Relaxation-time surface")
    ax.view_init(elev=22, azim=-50)

    fig.suptitle("Panel 4 — Critical slowing and load indicators",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_04_critical")


# ---------------------------------------------------------------------------
# Panel 5: Federation & cache extinction
# ---------------------------------------------------------------------------
def panel_5_federation():
    setup_style()
    d5 = _load("05_cache_extinction")
    d11 = _load("11_federation")
    d12 = _load("12_saturation")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Cache extinction speedup
    ax = fig.add_subplot(2, 2, 1)
    rec5 = d5["records"]
    spd = np.array([r["speedup"] for r in rec5])
    un = np.array([r["tp_inv_uncached"] for r in rec5])
    cn = np.array([r["tp_inv_cached"] for r in rec5])
    order = np.argsort(un)
    ax.plot(un[order], cn[order], "-o", color=BLUE_DARK, ms=4, label="cached")
    ax.plot(un[order], un[order], "--", color=ACCENT_RED, lw=1.0, label="uncached")
    ax.set_xlabel(r"uncached $TP^{-1}$")
    ax.set_ylabel(r"$TP^{-1}$ after cache")
    ax.set_title(f"(a) Cache extinction (mean speedup ${spd.mean():.2f}\\times$)")
    ax.legend()

    # (b) Federation composition: composite vs n
    ax = fig.add_subplot(2, 2, 2)
    rec11 = d11["records"]
    n = np.array([r["n_kernels"] for r in rec11])
    comp = np.array([r["tp_inv_fed"] for r in rec11])
    err = np.array([r["abs_err"] for r in rec11])
    order = np.argsort(n)
    ax.plot(n[order], comp[order], "-o", color=BLUE_DARK, ms=4)
    ax.set_xlabel("# kernels in federation")
    ax.set_ylabel(r"$TP^{-1}_\mathrm{fed}$")
    ax.set_title(f"(b) Federation composition (max err = {err.max():.2e})")

    # (c) Saturation: marginal benefit
    ax = fig.add_subplot(2, 2, 3)
    rec12 = d12["records"]
    n12 = np.array([r["n_kernels"] for r in rec12])
    marg = np.array([r["marginal"] for r in rec12])
    ax.bar(n12, marg, color=BLUE_LIGHT, edgecolor=BLUE_DARK)
    ax.set_xlabel("# kernels")
    ax.set_ylabel(r"marginal $\Delta TP^{-1}$")
    ax.set_title("(c) Diminishing returns")

    # (d) 3D: federation composition surface
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    n_grid = np.arange(1, 21)
    tp_per = np.linspace(5, 95, 30)
    N, T = np.meshgrid(n_grid, tp_per)
    Sigma = 100.0
    Comp = Sigma * (1 - (1 - T / Sigma) ** N)
    ax.plot_surface(N, T, Comp, cmap="Blues",
                    edgecolor=BLUE_DARK, linewidth=0.15, alpha=0.85)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"per-kernel $TP^{-1}$")
    ax.set_zlabel(r"$TP^{-1}_\mathrm{fed}$")
    ax.set_title("(d) Composition surface")
    ax.view_init(elev=22, azim=-58)

    fig.suptitle("Panel 5 — Federation and cache extinction",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_05_federation")


# ---------------------------------------------------------------------------
# Panel 6: Estimators & cross-architecture invariance
# ---------------------------------------------------------------------------
def panel_6_estimators():
    setup_style()
    d6 = _load("06_phase_coherence")
    d13 = _load("13_lag_estimator")
    d14 = _load("14_coupling_estimator")
    d15 = _load("15_cross_arch")

    fig = plt.figure(figsize=(11, 8.5))

    # (a) Phase coherence vs concentration
    ax = fig.add_subplot(2, 2, 1)
    rec6 = d6["records"]
    K = np.array([r["K_strength"] for r in rec6])
    R = np.array([r["R"] for r in rec6])
    order = np.argsort(K)
    ax.plot(K[order], R[order], "-o", color=BLUE_DARK, ms=4)
    ax.set_xlabel("concentration $K$")
    ax.set_ylabel("$R$ (phase coherence)")
    ax.set_title("(a) Coherence estimator")

    # (b) Lag lower bound: tau >= 1/f_max
    ax = fig.add_subplot(2, 2, 2)
    rec13 = d13["records"]
    lb = np.array([r["lb"] for r in rec13])
    mn = np.array([r["min_tau"] for r in rec13])
    ax.loglog(lb, mn, "o", color=BLUE_DARK, ms=4)
    x = np.logspace(np.log10(lb.min()), np.log10(lb.max()), 50)
    ax.loglog(x, x, "--", color=ACCENT_RED, lw=1.0,
              label=r"$\tau = 1/f_\mathrm{max}$")
    ax.set_xlabel(r"$1/f_\mathrm{max}$")
    ax.set_ylabel(r"$\min\,\tau$ measured")
    ax.set_title("(b) Lag lower bound")
    ax.legend()

    # (c) Coupling correlation by N
    ax = fig.add_subplot(2, 2, 3)
    rec14 = d14["records"]
    corr = np.array([r["correlation"] for r in rec14])
    Nvals = np.array([r["N"] for r in rec14])
    ax.scatter(Nvals, corr, c=BLUE_DARK, s=30, alpha=0.85)
    ax.axhline(0.5, color=ACCENT_RED, ls="--", lw=1.0, label="threshold")
    ax.set_xlabel("$N$ (decision classes)")
    ax.set_ylabel("recovery correlation")
    ax.set_title(f"(c) Coupling estimator (mean = {corr.mean():.2f})")
    ax.legend()

    # (d) 3D: cross-architecture invariance
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    rec15 = d15["records"]
    archs = sorted({r["architecture"] for r in rec15})
    arch_colours = [BLUE_DARK, BLUE_MID, BLUE_LIGHT, ACCENT_AMBER]
    for ai, arch in enumerate(archs):
        arr = [r for r in rec15 if r["architecture"] == arch]
        xs = [ai] * len(arr)
        ys = [r["trial"] for r in arr]
        zs = [r["tp_inv_predicted"] for r in arr]
        ax.scatter(xs, ys, zs, label=arch, s=34, alpha=0.85,
                   color=arch_colours[ai % len(arch_colours)])
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, fontsize=7)
    ax.set_ylabel("trial")
    ax.set_zlabel(r"$TP^{-1}$")
    ax.set_title("(d) Cross-architecture invariance")
    ax.legend(loc="upper right", fontsize=6)
    ax.view_init(elev=20, azim=-60)

    fig.suptitle("Panel 6 — Estimators and architecture invariance",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_panel(fig, "panel_06_estimators")


def main():
    paths = []
    for fn in (panel_1_universal_law, panel_2_queueing, panel_3_regimes,
               panel_4_critical, panel_5_federation, panel_6_estimators):
        p = fn()
        print(f"  saved {p}")
        paths.append(p)
    print(f"\nGenerated {len(paths)} UTL panels into {PANEL_DIR}")
    return paths


if __name__ == "__main__":
    main()
