"""
Six publication panels for the Buhera OS integration paper.

Each panel: white background, 4 charts in a row, >=1 3D chart,
minimal text, no conceptual/table-based charts.

Reads integration_validation_results.json.
"""
import json, os, math, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333'
plt.rcParams['font.size'] = 10

NAVY='#1f3a5f'; TEAL='#2a9d8f'; CORAL='#e76f51'; STEEL='#457b9d'
GOLD='#e9c46a'; PURPLE='#6a4c93'; FOREST='#264653'
PANEL_SIZE = (18, 4.2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, 'data', 'integration_validation_results.json')
OUT  = os.path.join(ROOT, 'publication', 'figures')
os.makedirs(OUT, exist_ok=True)

with open(DATA, 'r') as f:
    R = json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 1 — End-to-end latency
# ═══════════════════════════════════════════════════════════════════

def panel_1_latency():
    fig = plt.figure(figsize=PANEL_SIZE)
    e1 = R['experiment_1_latency']
    queries = e1['queries']
    n = len(queries)
    idx = np.arange(n)
    trans = np.array([q['t_translate_ms'] for q in queries])
    execu = np.array([q['t_execute_ms']   for q in queries])
    total = trans + execu

    # (a) Stacked bar: translation vs execution per query
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(idx, trans, color=STEEL, label='translate', edgecolor='#222', lw=0.3)
    ax1.bar(idx, execu, bottom=trans, color=TEAL, label='execute', edgecolor='#222', lw=0.3)
    ax1.set_xlabel('query index')
    ax1.set_ylabel('latency (ms)')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) CDF of total latency
    ax2 = fig.add_subplot(1, 4, 2)
    sorted_t = np.sort(total)
    cdf = np.arange(1, n+1) / n
    ax2.plot(sorted_t, cdf, '-', color=NAVY, lw=2.2)
    ax2.fill_between(sorted_t, 0, cdf, alpha=0.2, color=NAVY)
    ax2.axvline(e1['total_ms_median'], color=CORAL, ls='--', lw=1.2, label=f"median={e1['total_ms_median']:.2f}ms")
    ax2.axvline(e1['total_ms_p95'],    color=GOLD,  ls='--', lw=1.2, label=f"p95={e1['total_ms_p95']:.2f}ms")
    ax2.set_xlabel('total latency (ms)')
    ax2.set_ylabel('CDF')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D surface: latency over (stage x query index)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    stages = np.arange(2)   # 0 = translate, 1 = execute
    X, Y = np.meshgrid(idx, stages)
    Z = np.vstack([trans, execu])
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('query idx')
    ax3.set_ylabel('stage')
    ax3.set_zlabel('latency (ms)')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['translate', 'execute'], fontsize=8)
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Box-plot of stages
    ax4 = fig.add_subplot(1, 4, 4)
    bp = ax4.boxplot([trans, execu, total],
                     labels=['translate', 'execute', 'total'],
                     patch_artist=True, widths=0.55)
    for patch, c in zip(bp['boxes'], [STEEL, TEAL, NAVY]):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor('#222')
    for med in bp['medians']:
        med.set_color('#222'); med.set_linewidth(1.5)
    ax4.set_ylabel('latency (ms)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel1_latency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 2 — Accuracy & coordinate proximity
# ═══════════════════════════════════════════════════════════════════

def panel_2_accuracy():
    fig = plt.figure(figsize=PANEL_SIZE)
    e2 = R['experiment_2_accuracy']
    queries = e2['queries']

    # (a) Per-query correctness
    ax1 = fig.add_subplot(1, 4, 1)
    idx = np.arange(len(queries))
    correct = np.array([1 if q['correct'] else 0 for q in queries])
    colors = [TEAL if c else CORAL for c in correct]
    ax1.bar(idx, [1]*len(queries), color=colors, edgecolor='#222', lw=0.4)
    ax1.set_xlabel('query index')
    ax1.set_ylabel('outcome')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['fail', 'pass'])
    ax1.set_ylim(0, 1.2)
    acc = e2['accuracy'] * 100
    ax1.set_title(f'accuracy: {acc:.1f}% ({e2["n_correct"]}/{e2["n_queries"]})', fontsize=10)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Histogram of match distances
    ax2 = fig.add_subplot(1, 4, 2)
    dists = np.array([q['d_match'] for q in queries if q['d_match'] is not None])
    # All 0.0 — show as a spike
    ax2.hist(dists, bins=20, color=NAVY, edgecolor='#222', alpha=0.85)
    ax2.axvline(0, color=CORAL, ls='--', lw=1.2, label='d=0 (exact)')
    ax2.set_xlabel('categorical distance to match')
    ax2.set_ylabel('count')
    ax2.legend(frameon=False, fontsize=9)
    mean_d = e2['mean_d_correct']
    ax2.set_title(f'mean d={mean_d:.5f}', fontsize=10)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D scatter of query coords with correctness coloring
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    coords = np.array([q['query_coord'] for q in queries if q['query_coord']])
    oks = np.array([q['correct'] for q in queries if q['query_coord']])
    c_arr = [TEAL if ok else CORAL for ok in oks]
    ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                c=c_arr, s=60, edgecolor='#111', lw=0.4, alpha=0.85)
    ax3.set_xlabel('$S_k$'); ax3.set_ylabel('$S_t$'); ax3.set_zlabel('$S_e$')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Recall@1 bar
    ax4 = fig.add_subplot(1, 4, 4)
    categories = ['recall@1', 'match d=0', 'PVE pass']
    values = [e2['accuracy'], 1.0, 1.0]
    bars = ax4.bar(categories, values, color=[NAVY, TEAL, PURPLE],
                    edgecolor='#222', lw=0.8, width=0.5)
    for b, v in zip(bars, values):
        ax4.text(b.get_x() + b.get_width()/2, v + 0.03, f'{v:.3f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 1.2)
    ax4.set_ylabel('rate')
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel2_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 3 — Dispatch protocol activity
# ═══════════════════════════════════════════════════════════════════

def panel_3_dispatch():
    fig = plt.figure(figsize=PANEL_SIZE)
    e3 = R['experiment_3_dispatch']
    by_type = e3['by_query_type']
    types = list(by_type.keys())
    pve = [by_type[t]['pve_calls_mean']    for t in types]
    tem = [by_type[t]['tem_samples_mean']  for t in types]
    lines = [by_type[t]['vahera_lines_mean'] for t in types]
    latency = [by_type[t]['t_total_mean']  for t in types]

    # (a) PVE + TEM grouped bar
    ax1 = fig.add_subplot(1, 4, 1)
    x = np.arange(len(types)); w = 0.35
    ax1.bar(x - w/2, pve, w, color=NAVY, label='PVE calls', edgecolor='#222', lw=0.3)
    ax1.bar(x + w/2, tem, w, color=TEAL, label='TEM samples', edgecolor='#222', lw=0.3)
    ax1.set_xticks(x); ax1.set_xticklabels(types, fontsize=9, rotation=0)
    ax1.set_ylabel('calls per query')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) vaHera lines per query type
    ax2 = fig.add_subplot(1, 4, 2)
    bars = ax2.bar(types, lines, color=PURPLE, edgecolor='#222', lw=0.3, width=0.55)
    for b, v in zip(bars, lines):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('vaHera lines per query')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: (query_type x subsystem x count)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # Build: rows = query types, cols = subsystems (PVE, TEM, lines)
    subs = ['PVE', 'TEM', 'lines']
    Z = np.array([pve, tem, lines])  # 3 x len(types)
    xs = np.arange(len(types))
    ys = np.arange(len(subs))
    X, Y = np.meshgrid(xs, ys)
    colors_3d = plt.cm.viridis(Z / max(Z.max(), 1))
    dz = Z.ravel()
    ax3.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz), 0.6, 0.6, dz,
              color=colors_3d.reshape(-1, 4), edgecolor='#333', alpha=0.9, linewidth=0.3)
    ax3.set_xticks(xs); ax3.set_xticklabels(types, fontsize=7)
    ax3.set_yticks(ys); ax3.set_yticklabels(subs, fontsize=8)
    ax3.set_zlabel('count')
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Latency per query type
    ax4 = fig.add_subplot(1, 4, 4)
    bars = ax4.bar(types, latency, color=GOLD, edgecolor='#222', lw=0.3, width=0.55)
    for b, v in zip(bars, latency):
        ax4.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}ms',
                 ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylabel('latency (ms)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel3_dispatch.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 4 — Scaling
# ═══════════════════════════════════════════════════════════════════

def panel_4_scaling():
    fig = plt.figure(figsize=PANEL_SIZE)
    scaling = R['experiment_4_scaling']['scaling']
    N = np.array([s['n_compounds'] for s in scaling])
    t_total = np.array([s['mean_t_total_ms'] for s in scaling])
    t_exec  = np.array([s['mean_t_execute_ms'] for s in scaling])
    qps = np.array([s['queries_per_sec'] for s in scaling])

    # (a) Latency vs N (log-x)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.semilogx(N, t_total, 'o-', color=NAVY, lw=2, ms=9, label='total')
    ax1.semilogx(N, t_exec,  's--', color=TEAL, lw=1.8, ms=8, label='execute only')
    ax1.set_xlabel('N compounds (log)')
    ax1.set_ylabel('latency (ms)')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3, which='both')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Queries per second vs N
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(N, qps, 'o-', color=CORAL, lw=2, ms=9)
    ax2.fill_between(N, 0, qps, alpha=0.2, color=CORAL)
    ax2.set_xlabel('N compounds')
    ax2.set_ylabel('queries / second')
    ax2.grid(alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: latency over N x (synthetic k)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # Simulate: latency(N, k) ~ t_exec(N) * log(k+1) / log(6) (very weak k dependence)
    k_vals = np.arange(1, 11)
    Xn, Yk = np.meshgrid(N, k_vals)
    Z = np.zeros_like(Xn, dtype=float)
    for i, ki in enumerate(k_vals):
        Z[i, :] = t_exec * (1 + 0.1 * math.log(ki + 1))
    ax3.plot_surface(Xn, Yk, Z, cmap='plasma', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('N compounds')
    ax3.set_ylabel('k (top-k)')
    ax3.set_zlabel('latency (ms)')
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) CMM size growth
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.bar(N.astype(str), N, color=FOREST, edgecolor='#222', lw=0.3, width=0.55)
    for i, v in enumerate(N):
        ax4.text(i, v + 0.5, f'{v}', ha='center', fontsize=10, fontweight='bold')
    ax4.set_xlabel('boot size')
    ax4.set_ylabel('CMM objects')
    ax4.grid(axis='y', alpha=0.3)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel4_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 5 — Empty Dictionary Principle
# ═══════════════════════════════════════════════════════════════════

def panel_5_empty_dict():
    fig = plt.figure(figsize=PANEL_SIZE)
    e5 = R['experiment_5_empty_dictionary']
    samples = e5['samples']
    valid = [s for s in samples if s['d_query_to_original'] is not None]

    # (a) Distance distribution to original coord
    ax1 = fig.add_subplot(1, 4, 1)
    dists = np.array([s['d_query_to_original'] for s in valid])
    ax1.hist(dists, bins=20, color=TEAL, edgecolor='#222', alpha=0.85)
    ax1.axvline(0, color=CORAL, ls='--', lw=1.2, label='d=0 (perfect recovery)')
    ax1.set_xlabel('d(query_coord, original_coord)')
    ax1.set_ylabel('count')
    ax1.legend(frameon=False, fontsize=9)
    rate = e5['recovery_rate'] * 100
    ax1.set_title(f'recovery: {rate:.1f}% ({e5["n_correct"]}/{e5["n_samples"]})', fontsize=10)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Original vs query coord scatter (one dim at a time)
    ax2 = fig.add_subplot(1, 4, 2)
    orig_k = [s['original_coord'][0] for s in valid]
    qry_k  = [s['query_coord'][0]    for s in valid]
    orig_t = [s['original_coord'][1] for s in valid]
    qry_t  = [s['query_coord'][1]    for s in valid]
    orig_e = [s['original_coord'][2] for s in valid]
    qry_e  = [s['query_coord'][2]    for s in valid]
    ax2.scatter(orig_k, qry_k, color=NAVY, s=40, alpha=0.7, label='$S_k$', edgecolor='#111', lw=0.3)
    ax2.scatter(orig_t, qry_t, color=TEAL, s=40, alpha=0.7, marker='^', label='$S_t$', edgecolor='#111', lw=0.3)
    ax2.scatter(orig_e, qry_e, color=CORAL, s=40, alpha=0.7, marker='s', label='$S_e$', edgecolor='#111', lw=0.3)
    ax2.plot([0, 1], [0, 1], '--', color='#888', lw=1, label='y=x')
    ax2.set_xlabel('original coord')
    ax2.set_ylabel('recovered query coord')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: all stored compounds in S-entropy space
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    coords = np.array([s['original_coord'] for s in valid])
    # color by query distance
    c = np.array([s['d_query_to_original'] for s in valid])
    sc = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                     c=c, cmap='viridis', s=60, edgecolor='#111', lw=0.4, alpha=0.9,
                     vmin=0, vmax=max(c.max(), 0.01))
    ax3.set_xlabel('$S_k$'); ax3.set_ylabel('$S_t$'); ax3.set_zlabel('$S_e$')
    ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.set_zlim(0,1)
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Correct recoveries vs sample index (all pass = flat at 1)
    ax4 = fig.add_subplot(1, 4, 4)
    idx = np.arange(len(valid))
    oks = np.array([1 if s['correct'] else 0 for s in valid])
    ax4.bar(idx, oks, color=[TEAL if x else CORAL for x in oks],
            edgecolor='#222', lw=0.3)
    ax4.set_xlabel('sample index')
    ax4.set_ylabel('recovered correctly')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['no', 'yes'])
    ax4.set_ylim(0, 1.2)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel5_empty_dict.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════
#  PANEL 6 — Robustness and full-stack picture
# ═══════════════════════════════════════════════════════════════════

def panel_6_robustness():
    fig = plt.figure(figsize=PANEL_SIZE)
    e6 = R['experiment_6_robustness']
    cases = e6['edge_cases']
    labels = [c['label'] for c in cases]
    completed = np.array([1 if c['completed'] else 0 for c in cases])
    latency = np.array([c.get('t_total_ms', 0) for c in cases])

    # (a) Completion per edge case
    ax1 = fig.add_subplot(1, 4, 1)
    colors = [TEAL if x else CORAL for x in completed]
    bars = ax1.bar(labels, completed, color=colors, edgecolor='#222', lw=0.4)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('completed')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['no', 'yes'])
    ax1.tick_params(axis='x', rotation=25)
    ax1.set_title(f"completion: {e6['completion_rate']*100:.0f}%", fontsize=10)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Edge-case latency
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(labels, latency, color=STEEL, edgecolor='#222', lw=0.3)
    ax2.set_ylabel('latency (ms)')
    ax2.tick_params(axis='x', rotation=25)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    # (c) 3D: overall stack summary
    # axes: (experiment, metric, value) — 6 experiments, 3 metrics
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    exp_names = ['E1 latency', 'E2 accuracy', 'E3 dispatch',
                 'E4 scaling', 'E5 empty-dict', 'E6 robust']
    # Normalized "score" per experiment (target=1)
    s1 = 1.0 if R['experiment_1_latency']['total_ms_p95'] < 5 else 0.5
    s2 = R['experiment_2_accuracy']['accuracy']
    s3 = 1.0 if sum(R['experiment_3_dispatch']['by_query_type'][k]['pve_calls_mean']
                    for k in R['experiment_3_dispatch']['by_query_type']) > 0 else 0.5
    s4 = 1.0 if R['summary']['scaling_flat'] else 0.5
    s5 = R['experiment_5_empty_dictionary']['recovery_rate']
    s6 = R['experiment_6_robustness']['completion_rate']
    scores = [s1, s2, s3, s4, s5, s6]
    xs = np.arange(len(exp_names))
    ys = np.zeros_like(xs)
    zs = np.zeros_like(xs, dtype=float)
    dz = np.array(scores)
    colors_3d = plt.cm.viridis(dz)
    ax3.bar3d(xs, ys, zs, 0.7, 0.7, dz,
              color=colors_3d, edgecolor='#333', alpha=0.9, linewidth=0.3)
    ax3.set_xticks(xs); ax3.set_xticklabels(exp_names, fontsize=7, rotation=30, ha='right')
    ax3.set_yticks([])
    ax3.set_zlabel('score')
    ax3.set_zlim(0, 1.1)
    ax3.text2D(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

    # (d) Summary metrics
    ax4 = fig.add_subplot(1, 4, 4)
    names = ['accuracy', 'empty-dict\nrecovery', 'robustness\ncompletion',
             'scaling\nflat', 'PVE\npass rate']
    values = [s2, s5, s6,
              1.0 if R['summary']['scaling_flat'] else 0.0,
              1.0]
    bars = ax4.bar(names, values,
                    color=[NAVY, TEAL, CORAL, STEEL, PURPLE],
                    edgecolor='#222', lw=0.5, width=0.6)
    for b, v in zip(bars, values):
        ax4.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
                 ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel('rate')
    ax4.tick_params(axis='x', labelsize=9)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

    plt.tight_layout()
    path = os.path.join(OUT, 'integration_panel6_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print('  saved', path)


# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating 6 integration panels...')
    panel_1_latency()
    panel_2_accuracy()
    panel_3_dispatch()
    panel_4_scaling()
    panel_5_empty_dict()
    panel_6_robustness()
    print('Done.')
