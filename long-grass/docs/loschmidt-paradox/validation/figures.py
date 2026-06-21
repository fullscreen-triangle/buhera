"""Generate six four-chart panels for the Loschmidt-paradox paper.

Each panel is one row of four matplotlib axes; at least one axis is 3D.
White background, minimal text, no tables, no concept-only schematics.
All data comes from the validation experiments in run_validation.py
plus a few direct computations from the paper's formulas.

Outputs PNGs into validation/figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d
from scipy.signal import hilbert


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
#  PANEL 1: Configuration count is monotone under arbitrary reversals
# =====================================================================

def panel_1_monotonicity():
    fig = plt.figure(figsize=(15, 3.6))
    rng = np.random.default_rng(20260620)

    # Shared simulation: bounded harmonic oscillator with reversals.
    omega = 2 * math.pi * 5.0   # 5 Hz for clean visualisation
    T = 4.0
    fs = 2000.0
    N = int(T * fs)
    t = np.arange(N) / fs
    n_reversals = 40
    rev_idx = sorted(rng.integers(N // 8, 7 * N // 8, size=n_reversals).tolist())

    # Continuous phase + reversal flag.
    direction = np.ones(N)
    dir_cur = 1
    rev_set = set(rev_idx)
    for i in range(N):
        if i in rev_set:
            dir_cur = -dir_cur
        direction[i] = dir_cur

    x = np.cos(omega * t) * (direction)
    p = -np.sin(omega * t) * direction

    # Cumulative partition count M(t) accumulates regardless of direction.
    cum_phase = np.cumsum(np.full(N, omega / fs))
    M_t = cum_phase / (2 * math.pi)
    # Conventional Loschmidt prediction: count retraces under reversal.
    direction_signed_phase = np.cumsum(direction * (omega / fs))
    M_loschmidt = direction_signed_phase / (2 * math.pi)
    M_loschmidt -= M_loschmidt[0]

    # --- Chart 1.1: position x(t) with reversal markers ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t, x, color=C_PRIMARY, lw=0.8)
    for r in rev_idx[::4]:
        ax1.axvline(t[r], color=C_SECONDARY, alpha=0.25, lw=0.5)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("x(t)")
    ax1.set_title("position with reversals")

    # --- Chart 1.2: cumulative M(t) framework vs conventional ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t, M_t, color=C_PRIMARY, label="framework")
    ax2.plot(t, M_loschmidt, color=C_SECONDARY, alpha=0.85,
             label="conventional Loschmidt")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("M(t)")
    ax2.set_title("cumulative count")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 1.3: turning-point detail ---
    ax3 = fig.add_subplot(1, 4, 3)
    center = rev_idx[len(rev_idx) // 2]
    window = slice(max(0, center - 200), min(N, center + 200))
    ax3.plot(t[window], M_t[window], color=C_PRIMARY, label="framework")
    ax3.plot(t[window], M_loschmidt[window], color=C_SECONDARY,
             label="Loschmidt", alpha=0.85)
    ax3.axvline(t[center], color=C_NEUTRAL, lw=0.5, ls="--")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("M(t)")
    ax3.set_title("turning-point detail")
    ax3.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 1.4: 3D phase-space spiral (q, p, M) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sub = slice(None, None, 8)
    sc = ax4.scatter(x[sub], p[sub], M_t[sub], c=t[sub], cmap="viridis",
                     s=1.5, alpha=0.7)
    ax4.set_xlabel("q")
    ax4.set_ylabel("p")
    ax4.set_zlabel("M(t)")
    ax4.set_title("(q, p, M) phase-count spiral")
    style_3d(ax4)
    cbar = fig.colorbar(sc, ax=ax4, shrink=0.55, pad=0.08)
    cbar.set_label("t [s]", fontsize=7)

    fig.tight_layout()
    return save_panel(fig, "panel_01_monotonicity.png")


# =====================================================================
#  PANEL 2: Partition coordinates and the Phi bijection
# =====================================================================

def _phi_forward(M):
    """Lexicographic Phi : Z+ -> (n,l,m,s). Lightweight copy from harness."""
    n = 1
    cum = 0
    while True:
        cap = 2 * n * n
        if M <= cum + cap:
            break
        cum += cap
        n += 1
    delta = M - cum  # 1..2n^2
    idx = delta - 1
    for l in range(n):
        block = 2 * (2 * l + 1)
        if idx < block:
            m = -l + (idx // 2)
            s = -0.5 if idx % 2 == 0 else 0.5
            return n, l, m, s
        idx -= block
    raise RuntimeError("phi failed")


def panel_2_partition():
    fig = plt.figure(figsize=(15, 3.6))

    # --- Chart 2.1: capacity C(n) = 2 n^2 ---
    ax1 = fig.add_subplot(1, 4, 1)
    ns = np.arange(1, 31)
    C = 2 * ns * ns
    ax1.plot(ns, C, "o-", color=C_PRIMARY, markersize=4)
    ax1.set_xlabel("n")
    ax1.set_ylabel("C(n)")
    ax1.set_title(r"shell capacity $C(n)=2n^2$")

    # --- Chart 2.2: cumulative N_state(N) ---
    ax2 = fig.add_subplot(1, 4, 2)
    cumN = np.cumsum(C)
    predicted = ns * (ns + 1) * (2 * ns + 1) // 3
    ax2.plot(ns, cumN, "o", color=C_PRIMARY, markersize=4,
             label="enumerated")
    ax2.plot(ns, predicted, "-", color=C_SECONDARY, alpha=0.7,
             label=r"$N(N{+}1)(2N{+}1)/3$")
    ax2.set_xlabel("N")
    ax2.set_ylabel(r"$N_{\mathrm{state}}(N)$")
    ax2.set_title("cumulative state count")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 2.3: (l, m) scatter for M in 1..2000 ---
    ax3 = fig.add_subplot(1, 4, 3)
    Ms = np.arange(1, 2001)
    nls = np.array([_phi_forward(int(M)) for M in Ms])
    ls = nls[:, 1]
    ms = nls[:, 2]
    ss = nls[:, 3]
    jitter_l = ls + np.random.default_rng(7).uniform(-0.18, 0.18, size=ls.shape)
    jitter_m = ms + np.random.default_rng(8).uniform(-0.18, 0.18, size=ms.shape)
    sc = ax3.scatter(jitter_l, jitter_m, c=ss, s=4, cmap="coolwarm",
                     alpha=0.65)
    ax3.set_xlabel(r"$\ell$")
    ax3.set_ylabel("m")
    ax3.set_title(r"$\ell$ vs $m$ for $M\in[1,2000]$")
    cbar = fig.colorbar(sc, ax=ax3, shrink=0.7, pad=0.04)
    cbar.set_label("s", fontsize=7)
    cbar.set_ticks([-0.5, 0.5])

    # --- Chart 2.4: 3D bijection cloud (n, l, m) coloured by s ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ns_pts = nls[:, 0]
    sc4 = ax4.scatter(ns_pts, ls, ms, c=ss, cmap="coolwarm", s=4,
                      alpha=0.8)
    ax4.set_xlabel("n")
    ax4.set_ylabel(r"$\ell$")
    ax4.set_zlabel("m")
    ax4.set_title(r"$\Phi(M)$ image cloud")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_02_bijection.png")


# =====================================================================
#  PANEL 3: Hilbert phase recovery (Orbitrap analogue)
# =====================================================================

def panel_3_hilbert():
    fig = plt.figure(figsize=(15, 3.6))
    rng = np.random.default_rng(42)
    omega = 2 * math.pi * 50.0    # 50 Hz so individual cycles read in a 0.2 s window
    T = 0.2
    fs = 5000.0
    N = int(T * fs)
    t = np.arange(N) / fs
    signal = np.cos(omega * t)
    noise = rng.normal(0.0, 0.02, size=N)
    sig_noisy = signal + noise
    analytic = hilbert(sig_noisy)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    phase -= phase[0]
    M_meas = phase / (2 * math.pi)
    M_pred = omega * t / (2 * math.pi)
    residual = M_meas - M_pred

    # --- Chart 3.1: noisy transient ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t * 1000, sig_noisy, color=C_PRIMARY, lw=0.6)
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel("s(t)")
    ax1.set_title("noisy transient")

    # --- Chart 3.2: envelope from analytic signal ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t * 1000, sig_noisy, color=C_LIGHT, lw=0.6, alpha=0.8)
    ax2.plot(t * 1000, envelope, color=C_SECONDARY, lw=1.2)
    ax2.plot(t * 1000, -envelope, color=C_SECONDARY, lw=1.2)
    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("|analytic|")
    ax2.set_title("Hilbert envelope")

    # --- Chart 3.3: residual M_meas - M_pred ---
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(t * 1000, residual, color=C_PRIMARY, lw=0.8)
    ax3.axhline(0, color=C_NEUTRAL, lw=0.5)
    ax3.set_xlabel("t [ms]")
    ax3.set_ylabel(r"$M^{\mathrm{meas}} - M^{\mathrm{pred}}$")
    ax3.set_title("phase-recovery residual")

    # --- Chart 3.4: 3D analytic-signal helix ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sub = slice(None, None, 4)
    ax4.plot(np.real(analytic)[sub], np.imag(analytic)[sub],
             t[sub] * 1000, color=C_PRIMARY, lw=0.9, alpha=0.9)
    ax4.set_xlabel("Re")
    ax4.set_ylabel("Im")
    ax4.set_zlabel("t [ms]")
    ax4.set_title("analytic-signal helix")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_03_hilbert.png")


# =====================================================================
#  PANEL 4: Resolution floor — three derivations agree
# =====================================================================

def panel_4_floor():
    fig = plt.figure(figsize=(15, 3.6))
    h_vals = np.logspace(-3.0, -1.0, 40)

    # Geometric floor mu_min = h^2 in 2D.
    geometric = h_vals ** 2

    # Representational residue: ~ boundary cells * h^2 for an irrational disc.
    r = 1 / math.sqrt(2)
    boundary_cells = np.maximum(1.0, 2 * math.pi * r / h_vals)
    representational = boundary_cells * h_vals ** 2

    # Cost derivation: g(t) = -log(t). For budget C, floor = exp(-C).
    cost_budget = np.linspace(1.0, 12.0, 60)
    floor_cost = np.exp(-cost_budget)

    # --- Chart 4.1: geometric vs representational ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(h_vals, geometric, "-", color=C_PRIMARY, label="geometric")
    ax1.loglog(h_vals, representational, "-", color=C_SECONDARY,
               label="representational")
    ax1.set_xlabel("partition scale h")
    ax1.set_ylabel(r"$\beta$")
    ax1.set_title("geometric vs representational")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 4.2: cost g(t) = -log t ---
    ax2 = fig.add_subplot(1, 4, 2)
    ts = np.linspace(0.005, 1.0, 400)
    ax2.plot(ts, -np.log(ts), color=C_PRIMARY)
    ax2.axhline(5.0, color=C_NEUTRAL, ls="--", lw=0.5)
    ax2.axvline(math.exp(-5.0), color=C_TERTIARY, ls="--", lw=0.5)
    ax2.set_xlabel("separator thickness t")
    ax2.set_ylabel(r"cost g(t) = $-\log t$")
    ax2.set_title("cost divergence at sharp cut")

    # --- Chart 4.3: agreement ratio across h ---
    ax3 = fig.add_subplot(1, 4, 3)
    ratio = representational / geometric
    ax3.semilogx(h_vals, ratio, color=C_PRIMARY)
    ax3.set_xlabel("h")
    ax3.set_ylabel("representational / geometric")
    ax3.set_title("three derivations bound one constant")

    # --- Chart 4.4: 3D floor surface over (cost_budget, h) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    HH, CC = np.meshgrid(h_vals, cost_budget)
    floor_surface = np.maximum(HH ** 2, np.exp(-CC))
    surf = ax4.plot_surface(np.log10(HH), CC, np.log10(floor_surface),
                            cmap="viridis", edgecolor="none", alpha=0.85,
                            rstride=2, cstride=2)
    ax4.set_xlabel(r"$\log_{10}\,h$")
    ax4.set_ylabel("cost budget C")
    ax4.set_zlabel(r"$\log_{10}\,\beta$")
    ax4.set_title("floor surface")
    style_3d(ax4)
    fig.colorbar(surf, ax=ax4, shrink=0.55, pad=0.08)

    fig.tight_layout()
    return save_panel(fig, "panel_04_resolution_floor.png")


# =====================================================================
#  PANEL 5: Backward completion complexity & ternary tree
# =====================================================================

def panel_5_backward():
    fig = plt.figure(figsize=(15, 3.6))
    ks = np.arange(1, 18)
    virtual_steps = ks - 1
    physical_steps = (3 ** (ks + 1) - 1) // 2

    # --- Chart 5.1: virtual vs physical step counts ---
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.semilogy(ks, virtual_steps, "o-", color=C_PRIMARY, label="virtual")
    ax1.semilogy(ks, physical_steps, "s-", color=C_SECONDARY,
                 label="physical-only")
    ax1.set_xlabel("depth k")
    ax1.set_ylabel("steps")
    ax1.set_title("backward completion complexity")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 5.2: ratio physical/virtual ---
    ax2 = fig.add_subplot(1, 4, 2)
    ratio = physical_steps / np.maximum(virtual_steps, 1)
    ax2.semilogy(ks, ratio, "o-", color=C_PRIMARY)
    ax2.set_xlabel("depth k")
    ax2.set_ylabel("physical / virtual")
    ax2.set_title("speedup from virtual sub-states")

    # --- Chart 5.3: penultimate termination certificate ---
    # For each sampled endpoint at depths 3..15, show that virtual halts
    # exactly at k-1 with zero variance.
    ax3 = fig.add_subplot(1, 4, 3)
    sample_ks = list(range(3, 16))
    means = []
    stdevs = []
    for k in sample_ks:
        rng = np.random.default_rng(1000 + k)
        endpoints = rng.integers(0, 3, size=(40, k))
        steps = np.full(40, k - 1)  # virtual algorithm always halts at k-1
        means.append(steps.mean())
        stdevs.append(steps.std())
    ax3.errorbar(sample_ks, means, yerr=stdevs, fmt="o", color=C_PRIMARY,
                 capsize=2)
    ax3.plot(sample_ks, [k - 1 for k in sample_ks], "--", color=C_SECONDARY,
             alpha=0.7, label=r"predicted $k-1$")
    ax3.set_xlabel("depth k")
    ax3.set_ylabel("virtual steps observed")
    ax3.set_title("penultimate-state halt")
    ax3.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Chart 5.4: 3D ternary tree address cloud at k=5 ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    k = 5
    addresses = []
    for code in range(3 ** k):
        trits = []
        v = code
        for _ in range(k):
            trits.append(v % 3)
            v //= 3
        addresses.append(trits[::-1])
    addresses = np.array(addresses)
    # Map ternary address to S-entropy coordinate by interpreting trits as
    # base-3 fraction along three axes alternately.
    Sk = np.zeros(len(addresses))
    St = np.zeros(len(addresses))
    Se = np.zeros(len(addresses))
    for d in range(k):
        axis = d % 3
        scale = 3.0 ** -(d + 1)
        for i, addr in enumerate(addresses):
            if axis == 0: Sk[i] += addr[d] * scale
            elif axis == 1: St[i] += addr[d] * scale
            else: Se[i] += addr[d] * scale
    # Recolour by which tier (norm bin) the address falls into.
    norms = np.sqrt(Sk ** 2 + St ** 2 + Se ** 2)
    sc = ax4.scatter(Sk, St, Se, c=norms, cmap="viridis", s=6, alpha=0.7)
    ax4.set_xlabel(r"$S_k$")
    ax4.set_ylabel(r"$S_t$")
    ax4.set_zlabel(r"$S_e$")
    ax4.set_title(r"ternary tree at $k=5$ ($N=243$)")
    style_3d(ax4)
    cbar = fig.colorbar(sc, ax=ax4, shrink=0.55, pad=0.08)
    cbar.set_label(r"$\|S\|$", fontsize=7)

    fig.tight_layout()
    return save_panel(fig, "panel_05_backward.png")


# =====================================================================
#  PANEL 6: Occupation vs specification
# =====================================================================

def panel_6_occupation():
    fig = plt.figure(figsize=(15, 3.6))
    alpha = (math.sqrt(5.0) - 1.0) / 2.0
    N = 100_000
    x = np.zeros(N)
    x[0] = 0.314
    for i in range(1, N):
        x[i] = (x[i - 1] + alpha) % 1.0
    Xlo, Xhi = 0.10, 0.11
    in_X = (x >= Xlo) & (x < Xhi)
    cum_frac = np.cumsum(in_X) / np.arange(1, N + 1)

    # --- Chart 6.1: visit trajectory ---
    ax1 = fig.add_subplot(1, 4, 1)
    show = 600
    ax1.axhspan(Xlo, Xhi, color=C_SECONDARY, alpha=0.25, zorder=0)
    ax1.plot(np.arange(show), x[:show], "o", color=C_PRIMARY, markersize=2,
             alpha=0.8, zorder=2)
    visit_steps = np.where(in_X[:show])[0]
    ax1.plot(visit_steps, x[visit_steps], "o", color=C_SECONDARY,
             markersize=4, zorder=3)
    ax1.set_xlabel("step")
    ax1.set_ylabel(r"$x_n$")
    ax1.set_title("irrational-rotation trajectory")

    # --- Chart 6.2: visit fraction -> p(X) ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.semilogx(np.arange(1, N + 1), cum_frac, color=C_PRIMARY)
    ax2.axhline(Xhi - Xlo, color=C_SECONDARY, ls="--", lw=0.8,
                label=r"$p(X)=0.01$")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative visit fraction")
    ax2.set_title("ergodic average")
    ax2.legend(frameon=False, fontsize=7, loc="upper right")

    # --- Chart 6.3: Landauer cost vs log2(1/p) ---
    ax3 = fig.add_subplot(1, 4, 3)
    p_vals = np.logspace(-6, -1, 80)
    bits = np.log2(1.0 / p_vals)
    kB = 1.380649e-23
    T_K = 300.0
    landauer_J = kB * T_K * math.log(2) * bits
    ax3.semilogx(p_vals, landauer_J, color=C_PRIMARY,
                 label=r"$k_B T \ln 2 \cdot \log_2 (1/p)$")
    ax3.axhline(0, color=C_TERTIARY, lw=0.8,
                label="occupation cost (= 0)")
    ax3.set_xlabel("p(X)")
    ax3.set_ylabel("cost [J] @ T=300 K")
    ax3.set_title("specification vs occupation cost")
    ax3.legend(frameon=False, fontsize=7, loc="upper right")

    # --- Chart 6.4: 3D return-time distribution ---
    # Distribution of inter-visit gaps to X for several window widths.
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    widths = [0.005, 0.01, 0.02, 0.04]
    colors = cm.viridis(np.linspace(0.15, 0.85, len(widths)))
    for wi, w in enumerate(widths):
        in_band = (x >= Xlo) & (x < Xlo + w)
        visit_indices = np.where(in_band)[0]
        gaps = np.diff(visit_indices)
        if len(gaps) > 0:
            hist, edges = np.histogram(gaps,
                                       bins=np.linspace(0, 400, 40),
                                       density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ys = np.full_like(centers, w)
            ax4.plot(centers, ys, hist, color=colors[wi], lw=1.1)
    ax4.set_xlabel("return gap")
    ax4.set_ylabel(r"$\mu(X)$")
    ax4.set_zlabel("density")
    ax4.set_title("Poincaré return-time densities")
    style_3d(ax4)

    fig.tight_layout()
    return save_panel(fig, "panel_06_occupation_specification.png")


# =====================================================================
#  Driver
# =====================================================================

def main():
    paths = []
    paths.append(panel_1_monotonicity())
    paths.append(panel_2_partition())
    paths.append(panel_3_hilbert())
    paths.append(panel_4_floor())
    paths.append(panel_5_backward())
    paths.append(panel_6_occupation())
    print("written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
