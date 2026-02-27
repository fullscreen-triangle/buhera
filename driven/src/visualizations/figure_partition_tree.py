"""
Generate Figure 3: Ternary Partition Trees

4-panel figure showing:
1. Tree structure visualization (2D hierarchical layout)
2. Navigation complexity (2D log plot)
3. 3D partition space (3D scatter with cells)
4. Capacity scaling (2D line plot)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def draw_ternary_tree(ax, depth=3):
    """Draw a ternary tree structure."""

    def draw_node(x, y, level, max_level, parent_x=None, parent_y=None):
        if level > max_level:
            return

        # Draw node
        if level == 0:
            circle = Circle((x, y), 0.15, facecolor='#2E86AB',
                          edgecolor='black', linewidth=2, zorder=3)
        else:
            circle = Circle((x, y), 0.12, facecolor='white',
                          edgecolor='#2E86AB', linewidth=1.5, zorder=3)
        ax.add_patch(circle)

        # Draw connection to parent
        if parent_x is not None:
            ax.plot([parent_x, x], [parent_y, y], 'k-',
                   linewidth=1.5, alpha=0.5, zorder=1)

        if level < max_level:
            # Calculate child positions
            y_next = y - 1
            spacing = (3.0**(max_level - level - 1)) * 0.8

            # Draw three children
            for i, offset in enumerate([-spacing, 0, spacing]):
                x_child = x + offset
                draw_node(x_child, y_next, level + 1, max_level, x, y)

                # Label branches
                if level == 0:
                    mid_x = (x + x_child) / 2
                    mid_y = (y + y_next) / 2
                    ax.text(mid_x, mid_y, str(i), fontsize=9,
                           ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor='white',
                                   edgecolor='#F18F01', linewidth=1.5))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw tree
    draw_node(0, 0, 0, depth)

    # Add depth labels
    for i in range(depth + 1):
        ax.text(-5.5, -i, f'$d={i}$', fontsize=10,
               ha='right', va='center', fontweight='bold')

    # Add capacity annotation
    capacity_text = f'Capacity: $3^{depth} = {3**depth}$ leaves'
    ax.text(0, -depth - 0.8, capacity_text, fontsize=10,
           ha='center', fontweight='bold', color='#2E86AB')


def generate_partition_tree_figure(save_path: str = "driven/figures/figure_partition_tree.pdf"):
    """Generate 4-panel ternary partition tree figure."""

    fig = plt.figure(figsize=(16, 3.5))

    # ========== Panel 1: Tree Structure ==========
    ax1 = fig.add_subplot(141)
    draw_ternary_tree(ax1, depth=3)
    ax1.set_title('(a) Ternary Tree Structure', pad=10)

    # ========== Panel 2: Navigation Complexity ==========
    ax2 = fig.add_subplot(142)

    N = np.logspace(1, 6, 50)

    # O(log_3 N) - Categorical
    log3_n = np.log(N) / np.log(3)

    # O(log_2 N) - Binary search
    log2_n = np.log2(N)

    # O(N) - Linear search
    linear_n = N

    ax2.plot(N, log3_n, '-', color='#2E86AB', linewidth=3,
            label='$O(\\log_3 N)$ - Categorical')
    ax2.plot(N, log2_n, '--', color='#F18F01', linewidth=2.5,
            label='$O(\\log_2 N)$ - Binary')
    ax2.plot(N, linear_n, ':', color='#E63946', linewidth=2,
            label='$O(N)$ - Linear')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Elements $N$')
    ax2.set_ylabel('Navigation Steps')
    ax2.set_title('(b) Complexity Comparison')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':', which='both')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Annotate advantage
    n_point = 1000
    log3_point = np.log(n_point) / np.log(3)
    log2_point = np.log2(n_point)
    improvement = (log2_point - log3_point) / log2_point * 100

    ax2.annotate(f'{improvement:.0f}% fewer\nsteps',
                xy=(n_point, log3_point),
                xytext=(n_point * 10, log3_point * 3),
                fontsize=9, color='#2E86AB', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

    # ========== Panel 3: 3D Partition Space ==========
    ax3 = fig.add_subplot(143, projection='3d')

    # Create hierarchical 3D partition
    np.random.seed(42)

    # Generate points in ternary partitions
    n_points = 500
    depth = 4

    # Assign to ternary cells
    ternary_coords = np.random.randint(0, 3, size=(n_points, depth))

    # Convert to 3D positions
    x = np.sum(ternary_coords[:, ::2] * (3.0 ** -np.arange(0, depth, 2)[:, np.newaxis]).T, axis=1)
    y = np.sum(ternary_coords[:, 1::2] * (3.0 ** -np.arange(0, depth, 2)[:, np.newaxis]).T, axis=1)
    z = np.random.rand(n_points)  # Random depth

    # Color by partition
    colors = ternary_coords[:, 0]  # Color by first branch

    scatter = ax3.scatter(x, y, z, c=colors, cmap='viridis',
                         s=30, alpha=0.6, edgecolors='black',
                         linewidths=0.5, depthshade=True)

    # Draw partition boundaries (major divisions)
    for i in [1/3, 2/3]:
        ax3.plot([i, i], [0, 1], [0, 0], 'k--', alpha=0.3, linewidth=1)
        ax3.plot([0, 1], [i, i], [0, 0], 'k--', alpha=0.3, linewidth=1)

    # Draw 3D partition cells
    for i in range(3):
        for j in range(3):
            x_base = i/3
            y_base = j/3
            # Draw cube edges
            vertices = [
                [x_base, y_base, 0],
                [x_base + 1/3, y_base, 0],
                [x_base + 1/3, y_base + 1/3, 0],
                [x_base, y_base + 1/3, 0]
            ]
            for k in range(4):
                v1 = vertices[k]
                v2 = vertices[(k+1) % 4]
                ax3.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                        'k-', alpha=0.2, linewidth=0.8)

    ax3.set_xlabel('Partition X', labelpad=8)
    ax3.set_ylabel('Partition Y', labelpad=8)
    ax3.set_zlabel('Depth', labelpad=8)
    ax3.set_title('(c) 3D Partition Space')
    ax3.view_init(elev=20, azim=45)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.6, pad=0.1)
    cbar.set_label('Branch ID', rotation=90, labelpad=10)

    # ========== Panel 4: Capacity Scaling ==========
    ax4 = fig.add_subplot(144)

    depths = np.arange(1, 11)

    # Number of leaves at each depth
    ternary_capacity = 3 ** depths
    binary_capacity = 2 ** depths

    ax4.fill_between(depths, ternary_capacity * 0.9, ternary_capacity * 1.1,
                    alpha=0.2, color='#2E86AB')
    ax4.plot(depths, ternary_capacity, 'o-', color='#2E86AB',
            linewidth=2.5, markersize=9, markerfacecolor='white',
            markeredgewidth=2, label='Ternary: $3^d$')

    ax4.plot(depths, binary_capacity, 's-', color='#F18F01',
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgewidth=2, label='Binary: $2^d$')

    # Show C(n) = 2n² for quantum states
    n_vals = depths
    quantum_capacity = 2 * n_vals**2
    ax4.plot(depths, quantum_capacity, '^-', color='#A23B72',
            linewidth=2, markersize=7, markerfacecolor='white',
            markeredgewidth=2, label='Quantum: $2n^2$')

    ax4.set_xlabel('Tree Depth $d$')
    ax4.set_ylabel('Capacity (nodes)')
    ax4.set_title('(d) Storage Capacity')
    ax4.set_yscale('log')
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.suptitle('Ternary Partition Tree Architecture',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"[OK] Partition tree figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    fig = generate_partition_tree_figure()
    plt.show()
