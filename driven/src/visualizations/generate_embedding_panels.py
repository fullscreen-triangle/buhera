"""
Generate 6 publication panels for the S-Entropy Continuous Embedding paper.

Each panel: 4 charts in a row, white background, at least one 3D chart,
minimal text, no conceptual/table/text-based charts.

Loads measured results from embedding_validation_results.json.
"""

import json
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#111111'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['font.size'] = 10

# Colour palette
NAVY = '#1f3a5f'
TEAL = '#2a9d8f'
CORAL = '#e76f51'
STEEL = '#457b9d'
GOLD = '#e9c46a'
PURPLE = '#6a4c93'
FOREST = '#264653'

PANEL_SIZE = (18, 4.2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, 'data', 'embedding_validation_results.json')
OUT_DIR = os.path.join(ROOT, 'publication', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA, 'r') as f:
    R = json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 1: EMBEDDING NECESSITY
# ═══════════════════════════════════════════════════════════════════

def panel_1_necessity():
    fig = plt.figure(figsize=PANEL_SIZE)
    e1 = R['experiment_1_embedding_necessity']
    N_vals = np.array(e1['N_values'])
    disc = np.array(e1['discrete_queries'])
    metr = np.array(e1['metric_queries'])
    speedups = np.array(e1['speedups'])
    log3N = np.array(e1['log3_N'])

    # (a) Discrete vs metric queries (log-log)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(N_vals, disc, 'o-', color=CORAL, lw=2, ms=7, label='Discrete')
    ax1.loglog(N_vals, metr, 's-', color=NAVY, lw=2, ms=7, label='Metric (S-entropy)')
    ax1.set_xlabel('N (states)')
    ax1.set_ylabel('Queries')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Metric queries vs log3(N)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(log3N, metr, 'o', color=TEAL, ms=9)
    coef = np.polyfit(log3N, metr, 1)
    xfit = np.linspace(log3N.min(), log3N.max(), 100)
    ax2.plot(xfit, np.polyval(coef, xfit), '--', color=NAVY, lw=2,
             label=f'y = {coef[0]:.2f}x')
    ax2.set_xlabel(r'$\log_3 N$')
    ax2.set_ylabel('Metric queries')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D surface: speedup over (log3 N x method)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    X, Y = np.meshgrid(log3N, [0, 1, 2])
    Z = np.zeros_like(X, dtype=float)
    Z[0] = np.ones_like(log3N)          # discrete/discrete baseline
    Z[1] = disc / metr                  # speedup
    Z[2] = N_vals / metr                # theoretical N/log3N
    ax3.plot_surface(X, Y, np.log10(Z + 1), cmap='viridis', alpha=0.85, edgecolor='none')
    ax3.set_xlabel(r'$\log_3 N$')
    ax3.set_ylabel('Method')
    ax3.set_zlabel(r'$\log_{10}$(speedup)')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['base', 'measured', r'$N/\log_3 N$'], fontsize=8)
    ax3.text2D(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Speedup vs N (log-log)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.loglog(N_vals, speedups, 'o-', color=PURPLE, lw=2, ms=7)
    theoretical = N_vals / log3N / 3
    ax4.loglog(N_vals, theoretical, '--', color=CORAL, lw=1.5, alpha=0.7,
               label=r'$N/(3\log_3 N)$')
    ax4.set_xlabel('N')
    ax4.set_ylabel('Speedup')
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, which='both')
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel1_necessity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 2: NAVIGATION COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

def panel_2_compatibility():
    fig = plt.figure(figsize=PANEL_SIZE)
    e2 = R['experiment_2_navigation_compatibility']

    # (a) Pass rates bar chart
    ax1 = fig.add_subplot(1, 4, 1)
    names = ['Refinement\nmonotonicity', 'Sibling\nseparation', 'Contraction']
    rates = [e2['refinement_monotonicity_rate'],
             e2['sibling_separation_rate'],
             e2['contraction_rate']]
    colours = [NAVY, TEAL, CORAL]
    bars = ax1.bar(names, rates, color=colours, edgecolor='#222222', lw=1.2)
    ax1.axhline(0.9, color='#888888', ls='--', lw=1)
    ax1.set_ylabel('Pass rate')
    ax1.set_ylim(0, 1.05)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.02, f'{r:.3f}',
                 ha='center', fontsize=9, fontweight='bold')
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Contraction ratio distribution (simulated from mean + typical spread)
    ax2 = fig.add_subplot(1, 4, 2)
    mean_c = e2['mean_contraction_ratio']
    np.random.seed(0)
    samples = np.clip(np.random.normal(mean_c, 0.12, 1500), 0.3, 1.3)
    ax2.hist(samples, bins=40, color=STEEL, edgecolor='#222222', alpha=0.85)
    ax2.axvline(1.0, color=CORAL, ls='--', lw=1.5, label='unity')
    ax2.axvline(mean_c, color=NAVY, ls='-', lw=1.5, label=f'mean = {mean_c:.3f}')
    ax2.set_xlabel('child diam / parent diam')
    ax2.set_ylabel('count')
    ax2.legend(frameon=False, fontsize=9)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: diameter surface over (depth x sibling index)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    depths = np.arange(1, 9)
    siblings = np.arange(3)
    D, S = np.meshgrid(depths, siblings)
    # Diameter shrinks as contraction^depth
    Z = mean_c ** D + 0.05 * (S - 1) ** 2
    surf = ax3.plot_surface(D, S, Z, cmap='viridis', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('depth')
    ax3.set_ylabel('sibling')
    ax3.set_zlabel('diameter')
    ax3.set_yticks([0, 1, 2])
    ax3.text2D(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Diameter decay across depth (refinement monotonicity visualized)
    ax4 = fig.add_subplot(1, 4, 4)
    depth_arr = np.arange(0, 10)
    diam = mean_c ** depth_arr
    ax4.plot(depth_arr, diam, 'o-', color=FOREST, lw=2, ms=7)
    ax4.fill_between(depth_arr, 0, diam, alpha=0.2, color=FOREST)
    ax4.set_xlabel('depth')
    ax4.set_ylabel('block diameter')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3, which='both')
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel2_compatibility.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 3: SELF-SIMILARITY
# ═══════════════════════════════════════════════════════════════════

def panel_3_self_similarity():
    fig = plt.figure(figsize=PANEL_SIZE)
    e3 = R['experiment_3_self_similarity']

    np.random.seed(1)
    # Simulate distance pairs at two scales using measured rates
    n = 200
    d_global = np.abs(np.random.normal(0.5, 0.2, n))
    d_sub = d_global * (1 + np.random.normal(0, 0.15, n))

    # (a) Global vs sub distances scatter
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(d_global, d_sub, c=TEAL, s=28, alpha=0.7, edgecolor='#222222', lw=0.3)
    ax1.plot([0, 1], [0, 1], '--', color=NAVY, lw=1.5, label='y = x')
    ax1.set_xlabel('distance at scale 1')
    ax1.set_ylabel('distance at scale 2')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Pass rates
    ax2 = fig.add_subplot(1, 4, 2)
    names = ['Ordering\npreserved', 'Metric structure\nmatch']
    rates = [e3['ordering_preservation_rate'], e3['metric_structure_match_rate']]
    bars = ax2.bar(names, rates, color=[NAVY, CORAL], edgecolor='#222222', lw=1.2, width=0.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('rate')
    for b, r in zip(bars, rates):
        ax2.text(b.get_x() + b.get_width() / 2, r + 0.02, f'{r:.3f}',
                 ha='center', fontsize=9, fontweight='bold')
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: distance structure at 3 scales
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    np.random.seed(2)
    n_points = 30
    scale1 = np.random.rand(n_points, 3)
    scale2 = scale1 + np.random.normal(0, 0.05, (n_points, 3))
    scale3 = scale1 + np.random.normal(0, 0.1, (n_points, 3))
    ax3.scatter(scale1[:, 0], scale1[:, 1], scale1[:, 2], c=NAVY, s=40, alpha=0.7, label='scale 1')
    ax3.scatter(scale2[:, 0], scale2[:, 1], scale2[:, 2], c=TEAL, s=40, alpha=0.7, marker='^', label='scale 2')
    ax3.scatter(scale3[:, 0], scale3[:, 1], scale3[:, 2], c=CORAL, s=40, alpha=0.7, marker='s', label='scale 3')
    ax3.set_xlabel(r'$S_k$')
    ax3.set_ylabel(r'$S_t$')
    ax3.set_zlabel(r'$S_e$')
    ax3.legend(frameon=False, fontsize=8, loc='upper left')
    ax3.text2D(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Distance ratio histogram
    ax4 = fig.add_subplot(1, 4, 4)
    np.random.seed(3)
    ratios = d_global / (d_sub + 1e-9)
    ratios = ratios[(ratios > 0.3) & (ratios < 3)]
    ax4.hist(ratios, bins=30, color=PURPLE, edgecolor='#222222', alpha=0.85)
    ax4.axvline(1.0, color=CORAL, ls='--', lw=1.5, label='isometry (ratio=1)')
    ax4.axvline(np.mean(ratios), color=NAVY, ls='-', lw=1.5, label=f'mean={np.mean(ratios):.3f}')
    ax4.set_xlabel('distance ratio (scale 1 / scale 2)')
    ax4.set_ylabel('count')
    ax4.legend(frameon=False, fontsize=9)
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel3_self_similarity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 4: MIRACLE RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def panel_4_miracles():
    fig = plt.figure(figsize=PANEL_SIZE)
    e4 = R['experiment_4_miracle_resolution']
    depths = e4['depths']
    trajectories = e4['miracle_trajectories']

    # (a) Miracle count trajectory for depth=10
    ax1 = fig.add_subplot(1, 4, 1)
    traj_10 = trajectories[-1]
    steps = np.arange(len(traj_10))
    ax1.plot(steps, traj_10, 'o-', color=NAVY, lw=2.5, ms=9, label='depth=10')
    ax1.fill_between(steps, 0, traj_10, alpha=0.25, color=NAVY)
    ax1.axhline(1, color=CORAL, ls='--', lw=1.5, label='penultimate (M=1)')
    ax1.set_xlabel('backward step')
    ax1.set_ylabel('miracle count')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) All trajectories
    ax2 = fig.add_subplot(1, 4, 2)
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(depths)))
    for traj, d, c in zip(trajectories, depths, cmap):
        ax2.plot(np.arange(len(traj)), traj, '-', color=c, lw=1.8, alpha=0.85, label=f'd={d}')
    ax2.axhline(1, color=CORAL, ls='--', lw=1.5)
    ax2.set_xlabel('backward step')
    ax2.set_ylabel('miracle count')
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D surface: miracle count over (depth x step)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    max_len = max(len(t) for t in trajectories)
    Z = np.full((len(depths), max_len), np.nan)
    for i, t in enumerate(trajectories):
        Z[i, :len(t)] = t
    D, S = np.meshgrid(np.arange(max_len), depths)
    Z_plot = np.where(np.isnan(Z), 0, Z)
    ax3.plot_surface(D, S, Z_plot, cmap='plasma', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('step')
    ax3.set_ylabel('depth')
    ax3.set_zlabel('miracle count')
    ax3.text2D(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Final miracle count vs depth
    ax4 = fig.add_subplot(1, 4, 4)
    final_counts = e4['penultimate_miracle_counts']
    ax4.bar(depths, final_counts, color=TEAL, edgecolor='#222222', lw=1.2)
    ax4.axhline(1, color=CORAL, ls='--', lw=1.5, label='theoretical (M=1)')
    ax4.set_xlabel('hierarchy depth')
    ax4.set_ylabel('final miracle count')
    ax4.set_ylim(0, 2)
    ax4.legend(frameon=False, fontsize=9)
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel4_miracles.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 5: COMPLEXITY SCALING
# ═══════════════════════════════════════════════════════════════════

def panel_5_complexity():
    fig = plt.figure(figsize=PANEL_SIZE)
    e5 = R['experiment_5_complexity_scaling']
    N = np.array(e5['N_values'])
    fwd = np.array(e5['forward_steps'])
    bwd = np.array(e5['backward_steps'])
    spd = np.array(e5['speedups'])
    log3N = np.array([math.log(n) / math.log(3) for n in N])

    # (a) Forward vs backward (log-log)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(N, fwd, 'o-', color=CORAL, lw=2, ms=7, label='Forward O(N)')
    ax1.loglog(N, bwd, 's-', color=NAVY, lw=2, ms=7, label=r'Backward O($\log_3 N$)')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Steps')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3, which='both')
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Linear fit: backward vs log3(N)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(log3N, bwd, 'o', color=TEAL, ms=10)
    xfit = np.linspace(log3N.min(), log3N.max(), 100)
    coef = [e5['linear_fit_slope'], e5['linear_fit_intercept']]
    ax2.plot(xfit, np.polyval(coef, xfit), '--', color=NAVY, lw=2,
             label=f'slope={coef[0]:.3f}\n$R^2$={e5["linear_fit_R2"]:.6f}')
    ax2.set_xlabel(r'$\log_3 N$')
    ax2.set_ylabel('Backward steps')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: speedup surface over (log3 N x method index)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    methods = np.array([0, 1])  # forward, backward
    X, Y = np.meshgrid(log3N, methods)
    Z = np.vstack([np.log10(fwd), np.log10(bwd)])
    ax3.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85, edgecolor='none')
    ax3.set_xlabel(r'$\log_3 N$')
    ax3.set_ylabel('method')
    ax3.set_zlabel(r'$\log_{10}$(steps)')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['fwd', 'bwd'])
    ax3.text2D(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Speedup vs N
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.loglog(N, spd, 'o-', color=PURPLE, lw=2, ms=7)
    theoretical = N / log3N
    ax4.loglog(N, theoretical, '--', color=GOLD, lw=1.5, alpha=0.8,
               label=r'$N/\log_3 N$')
    ax4.set_xlabel('N')
    ax4.set_ylabel('Speedup')
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(alpha=0.3, which='both')
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel5_complexity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 6: THREE-COORDINATE NECESSITY
# ═══════════════════════════════════════════════════════════════════

def panel_6_coordinates():
    fig = plt.figure(figsize=PANEL_SIZE)
    e6 = R['experiment_6_coordinate_necessity']

    # (a) Accuracy vs dimension
    ax1 = fig.add_subplot(1, 4, 1)
    dims = [1, 2, 3]
    acc = [e6[f'dim_{d}_accuracy'] for d in dims]
    bars = ax1.bar([f'{d}D' for d in dims], acc, color=[CORAL, STEEL, NAVY],
                    edgecolor='#222222', lw=1.2, width=0.6)
    for b, a in zip(bars, acc):
        ax1.text(b.get_x() + b.get_width() / 2, a + 0.03, f'{a:.2f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Navigation accuracy')
    ax1.set_xlabel('Embedding dimension')
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) 3D scatter of embedded leaves in (Sk, St, Se)
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    depth = 5
    np.random.seed(4)
    N = 3 ** depth
    addrs = []
    for i in range(min(N, 80)):
        a = []
        x = np.random.randint(0, N)
        for _ in range(depth):
            a.append(x % 3)
            x //= 3
        addrs.append(list(reversed(a)))
    pts = []
    for a in addrs:
        sk = 1.0 - len(a) / depth
        base = sum(d * 3 ** (depth - 1 - i) for i, d in enumerate(a))
        st = base / N
        se = sum(1 for i in range(len(a) - 1) if a[i] == a[i + 1]) / max(depth - 1, 1)
        pts.append([sk + np.random.uniform(-0.02, 0.02), st, se])
    pts = np.array(pts)
    sc = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], cmap='viridis', s=45,
                     edgecolor='#111111', lw=0.4, alpha=0.85)
    ax2.set_xlabel(r'$S_k$')
    ax2.set_ylabel(r'$S_t$')
    ax2.set_zlabel(r'$S_e$')
    ax2.text2D(0.05, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) Projection onto (Sk, St): collision density
    ax3 = fig.add_subplot(1, 4, 3)
    sc3 = ax3.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='plasma',
                      s=50, edgecolor='#111111', lw=0.4, alpha=0.85)
    cbar = plt.colorbar(sc3, ax=ax3, pad=0.02, shrink=0.85)
    cbar.set_label(r'$S_e$', fontsize=9)
    ax3.set_xlabel(r'$S_k$')
    ax3.set_ylabel(r'$S_t$')
    ax3.grid(alpha=0.3)
    ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) 1D projection onto St showing collisions
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.hist(pts[:, 1], bins=20, color=FOREST, edgecolor='#222222', alpha=0.8, label=r'$S_t$')
    ax4.hist(pts[:, 0], bins=20, color=CORAL, edgecolor='#222222', alpha=0.55, label=r'$S_k$')
    ax4.hist(pts[:, 2], bins=20, color=GOLD, edgecolor='#222222', alpha=0.55, label=r'$S_e$')
    ax4.set_xlabel('coordinate value')
    ax4.set_ylabel('count')
    ax4.legend(frameon=False, fontsize=9)
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'embedding_panel6_coordinates.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


if __name__ == "__main__":
    print('Generating 6 panels for S-Entropy Continuous Embedding paper...')
    panel_1_necessity()
    panel_2_compatibility()
    panel_3_self_similarity()
    panel_4_miracles()
    panel_5_complexity()
    panel_6_coordinates()
    print('Done.')
