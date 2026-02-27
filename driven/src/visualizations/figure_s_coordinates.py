"""
Generate Figure 4: S-Entropy Coordinate Addressing

4-panel figure showing:
1. S-coordinate space visualization (2D scatter with regions)
2. Address resolution trajectory (2D path plot)
3. 3D S-space with partition cells (3D scatter)
4. Entropy component correlation (2D heatmap/correlation matrix)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.collections import PatchCollection
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def generate_s_coordinate_figure(save_path: str = "driven/figures/figure_s_coordinates.pdf"):
    """Generate 4-panel S-coordinate addressing figure."""

    fig = plt.figure(figsize=(16, 3.5))

    np.random.seed(42)

    # ========== Panel 1: S-Coordinate Space (2D projection) ==========
    ax1 = fig.add_subplot(141)

    # Generate sample data points in S-space
    n_points = 300

    # Create clusters representing different partition regions
    centers = np.array([
        [0.2, 0.2], [0.5, 0.5], [0.8, 0.8],
        [0.2, 0.8], [0.8, 0.2]
    ])

    points_s_k = []
    points_s_t = []
    colors_list = []

    for i, center in enumerate(centers):
        n_cluster = n_points // len(centers)
        cluster_s_k = np.random.normal(center[0], 0.08, n_cluster)
        cluster_s_t = np.random.normal(center[1], 0.08, n_cluster)

        points_s_k.extend(cluster_s_k)
        points_s_t.extend(cluster_s_t)
        colors_list.extend([i] * n_cluster)

    points_s_k = np.array(points_s_k)
    points_s_t = np.array(points_s_t)
    colors_list = np.array(colors_list)

    # Clip to [0,1]
    points_s_k = np.clip(points_s_k, 0, 1)
    points_s_t = np.clip(points_s_t, 0, 1)

    # Plot points
    scatter = ax1.scatter(points_s_k, points_s_t, c=colors_list,
                         cmap='tab10', s=30, alpha=0.6,
                         edgecolors='black', linewidths=0.5)

    # Draw partition grid (ternary divisions)
    for i in [1/3, 2/3]:
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
        ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

    # Label regions
    region_labels = [
        (1/6, 1/6, '(0,0)'),
        (0.5, 1/6, '(1,0)'),
        (5/6, 1/6, '(2,0)'),
        (1/6, 0.5, '(0,1)'),
        (0.5, 0.5, '(1,1)'),
        (5/6, 0.5, '(2,1)'),
        (1/6, 5/6, '(0,2)'),
        (0.5, 5/6, '(1,2)'),
        (5/6, 5/6, '(2,2)')
    ]

    for x, y, label in region_labels:
        ax1.text(x, y, label, ha='center', va='center',
                fontsize=8, color='gray', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.7, edgecolor='none'))

    ax1.set_xlabel('$S_k$ (Kinetic Entropy)')
    ax1.set_ylabel('$S_t$ (Thermal Entropy)')
    ax1.set_title('(a) S-Coordinate Space')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle=':')

    # ========== Panel 2: Address Resolution Trajectory ==========
    ax2 = fig.add_subplot(142)

    # Generate a path showing hierarchical resolution
    # Start from broad partition, refine iteratively
    path_x = [0.5]  # Start at center
    path_y = [0.5]

    # Simulate navigation to target (0.75, 0.25)
    target_x, target_y = 0.75, 0.25

    current_x, current_y = 0.5, 0.5
    partition_size = 1.0

    for iteration in range(5):
        # Determine which ternary cell contains target
        dx = (target_x - current_x) / partition_size
        dy = (target_y - current_y) / partition_size

        # Move to appropriate sub-partition
        if dx < -1/3:
            step_x = -partition_size / 3
        elif dx > 1/3:
            step_x = partition_size / 3
        else:
            step_x = 0

        if dy < -1/3:
            step_y = -partition_size / 3
        elif dy > 1/3:
            step_y = partition_size / 3
        else:
            step_y = 0

        current_x += step_x
        current_y += step_y
        partition_size /= 3

        path_x.append(current_x)
        path_y.append(current_y)

    # Plot trajectory
    ax2.plot(path_x, path_y, 'o-', color='#2E86AB', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='Resolution Path', zorder=3)

    # Mark start and end
    ax2.scatter([path_x[0]], [path_y[0]], c='#2E86AB', s=200,
               marker='s', edgecolors='black', linewidths=2,
               label='Start', zorder=4)
    ax2.scatter([path_x[-1]], [path_y[-1]], c='#E63946', s=200,
               marker='*', edgecolors='black', linewidths=2,
               label='Target', zorder=4)

    # Show refinement regions
    for i in range(1, len(path_x)):
        size = (1/3)**(i-1)
        rect = Rectangle((path_x[i] - size/2, path_y[i] - size/2),
                        size, size, fill=False, edgecolor='#F18F01',
                        linewidth=2, linestyle='--', alpha=0.5)
        ax2.add_patch(rect)

    # Draw partition grid
    for i in [1/3, 2/3]:
        ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        ax2.axhline(y=i, color='gray', linestyle=':', alpha=0.3)

    ax2.set_xlabel('$S_k$')
    ax2.set_ylabel('$S_t$')
    ax2.set_title('(b) Hierarchical Resolution')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Annotate steps
    ax2.text(0.85, 0.9, f'{len(path_x)-1} steps',
            fontsize=11, fontweight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='#2E86AB', linewidth=2))

    # ========== Panel 3: 3D S-Space (with all three coordinates) ==========
    ax3 = fig.add_subplot(143, projection='3d')

    # Generate 3D S-coordinates
    n_3d_points = 400
    s_k_3d = np.random.rand(n_3d_points)
    s_t_3d = np.random.rand(n_3d_points)
    s_e_3d = np.random.rand(n_3d_points)

    # Color by partition index (ternary encoding)
    partition_idx = (np.floor(s_k_3d * 3) * 9 +
                    np.floor(s_t_3d * 3) * 3 +
                    np.floor(s_e_3d * 3))

    scatter_3d = ax3.scatter(s_k_3d, s_t_3d, s_e_3d,
                            c=partition_idx, cmap='tab20',
                            s=25, alpha=0.6, edgecolors='black',
                            linewidths=0.3, depthshade=True)

    # Draw partition boundaries
    for i in [1/3, 2/3]:
        # XY planes
        ax3.plot([i, i], [0, 1], [0, 0], 'k--', alpha=0.2, linewidth=0.8)
        ax3.plot([0, 1], [i, i], [0, 0], 'k--', alpha=0.2, linewidth=0.8)
        # XZ planes
        ax3.plot([i, i], [0, 0], [0, 1], 'k--', alpha=0.2, linewidth=0.8)
        ax3.plot([0, 1], [0, 0], [i, i], 'k--', alpha=0.2, linewidth=0.8)
        # YZ planes
        ax3.plot([0, 0], [i, i], [0, 1], 'k--', alpha=0.2, linewidth=0.8)
        ax3.plot([0, 0], [0, 1], [i, i], 'k--', alpha=0.2, linewidth=0.8)

    ax3.set_xlabel('$S_k$\\n(Kinetic)', labelpad=8, fontsize=9)
    ax3.set_ylabel('$S_t$\\n(Thermal)', labelpad=8, fontsize=9)
    ax3.set_zlabel('$S_e$\\n(Exchange)', labelpad=8, fontsize=9)
    ax3.set_title('(c) 3D S-Entropy Space')
    ax3.view_init(elev=20, azim=45)

    # Set limits
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)

    # ========== Panel 4: Entropy Component Correlations ==========
    ax4 = fig.add_subplot(144)

    # Generate correlation data for different system types
    systems = ['Sorted Array', 'Random Data', 'Hot Gas', 'Cold Crystal']
    n_systems = len(systems)

    # Correlation matrix (S_k, S_t, S_e correlations)
    # Each system has different entropy structure
    correlation_data = np.array([
        [0.1, 0.05, 0.2],   # Sorted: low kinetic, low thermal, some exchange
        [0.5, 0.5, 0.5],    # Random: balanced
        [0.9, 0.95, 0.7],   # Hot gas: high kinetic, high thermal
        [0.2, 0.1, 0.8]     # Cold crystal: low kinetic/thermal, high exchange
    ])

    # Create stacked bar chart
    x = np.arange(n_systems)
    width = 0.6

    colors = ['#2E86AB', '#F18F01', '#A23B72']
    labels = ['$S_k$', '$S_t$', '$S_e$']

    bottom = np.zeros(n_systems)
    for i in range(3):
        ax4.bar(x, correlation_data[:, i], width, bottom=bottom,
               label=labels[i], color=colors[i], edgecolor='black',
               linewidth=1.5, alpha=0.8)
        bottom += correlation_data[:, i]

    ax4.set_ylabel('Normalized Entropy')
    ax4.set_title('(d) Entropy Composition')
    ax4.set_xticks(x)
    ax4.set_xticklabels(systems, rotation=15, ha='right')
    ax4.legend(loc='upper left', framealpha=0.9, ncol=3)
    ax4.set_ylim(0, 1.1)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')

    fig.suptitle('S-Entropy Coordinate Addressing',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"[OK] S-coordinate figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    fig = generate_s_coordinate_figure()
    plt.show()
