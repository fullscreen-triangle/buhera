"""
Generate Figure 5: Categorical Processor Architecture

4-panel figure showing:
1. Processor pipeline (2D flow diagram)
2. Trajectory completion vs forward simulation (2D comparison)
3. 3D state space navigation (3D trajectory plot)
4. Penultimate state detection (2D distance plot)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.patches import Rectangle, Arrow
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def generate_processor_figure(save_path: str = "driven/figures/figure_processor.pdf"):
    """Generate 4-panel categorical processor figure."""

    fig = plt.figure(figsize=(16, 3.5))

    # ========== Panel 1: Processor Pipeline ==========
    ax1 = fig.add_subplot(141)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')

    # Define pipeline stages
    stages = [
        {'name': 'Input', 'y': 7, 'color': '#2E86AB'},
        {'name': 'Resolve\nAddress', 'y': 5.5, 'color': '#F18F01'},
        {'name': 'Navigate to\nPenultimate', 'y': 4, 'color': '#A23B72'},
        {'name': 'Complete\nTrajectory', 'y': 2.5, 'color': '#E63946'},
        {'name': 'Output', 'y': 1, 'color': '#2E86AB'}
    ]

    for i, stage in enumerate(stages):
        # Draw box
        box = FancyBboxPatch((2, stage['y']-0.3), 6, 0.6,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=stage['color'],
                            linewidth=2, alpha=0.7)
        ax1.add_patch(box)

        # Add text
        ax1.text(5, stage['y'], stage['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Add arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((5, stage['y']-0.35),
                                   (5, stages[i+1]['y']+0.35),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=2.5, color='black', zorder=0)
            ax1.add_patch(arrow)

    # Add side annotations
    ax1.text(9, 5.5, 'O(log₃ N)', ha='left', va='center',
            fontsize=9, style='italic', color='#F18F01')
    ax1.text(9, 4, 'O(log₃ N)', ha='left', va='center',
            fontsize=9, style='italic', color='#A23B72')
    ax1.text(9, 2.5, 'O(1)', ha='left', va='center',
            fontsize=9, style='italic', color='#E63946',
            fontweight='bold')

    ax1.set_title('(a) Processing Pipeline', pad=10)

    # ========== Panel 2: Trajectory Completion vs Forward Simulation ==========
    ax2 = fig.add_subplot(142)

    # Time steps
    steps = np.arange(0, 11)

    # Forward simulation: O(N) operations
    forward_ops = steps * 10
    forward_time = steps

    # Trajectory completion: Navigate to penultimate (O(log N)), then complete (O(1))
    navigate_time = np.log(steps + 1) / np.log(3)
    complete_time = navigate_time[-1] + 1
    trajectory_time = np.full_like(steps, complete_time, dtype=float)
    trajectory_time[steps <= np.log(10)/np.log(3)] = navigate_time[steps <= np.log(10)/np.log(3)]

    # Plot
    ax2.fill_between(steps, 0, forward_time, alpha=0.2, color='#E63946',
                    label='Forward Simulation')
    ax2.plot(steps, forward_time, 's-', color='#E63946',
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgewidth=2, label='Forward: O(N)')

    ax2.fill_between(steps, 0, trajectory_time, alpha=0.2, color='#2E86AB')
    ax2.plot(steps, trajectory_time, 'o-', color='#2E86AB',
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgewidth=2, label='Categorical: O(log N) + O(1)')

    # Mark penultimate state
    penultimate_step = int(np.log(10)/np.log(3))
    ax2.axvline(x=penultimate_step, color='#F18F01', linestyle='--',
               linewidth=2, alpha=0.7, label='Penultimate State')

    ax2.set_xlabel('Computation Progress')
    ax2.set_ylabel('Time / Operations')
    ax2.set_title('(b) Completion vs Simulation')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Annotate speedup
    speedup = forward_time[-1] / trajectory_time[-1]
    ax2.annotate(f'{speedup:.1f}× faster',
                xy=(steps[-1], trajectory_time[-1]),
                xytext=(steps[-2], trajectory_time[-1] + 2),
                fontsize=10, color='#2E86AB', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

    # ========== Panel 3: 3D State Space Navigation ==========
    ax3 = fig.add_subplot(143, projection='3d')

    # Generate trajectory in state space
    t = np.linspace(0, 2*np.pi, 100)

    # Initial state
    x_init = np.ones_like(t) * 0
    y_init = np.ones_like(t) * 0
    z_init = np.ones_like(t) * 0

    # Final state
    x_final = 5
    y_final = 5
    z_final = 5

    # Forward simulation (spiral from initial to final)
    x_forward = t / (2*np.pi) * x_final + 0.5 * np.sin(4*t)
    y_forward = t / (2*np.pi) * y_final + 0.5 * np.cos(4*t)
    z_forward = t / (2*np.pi) * z_final

    # Categorical navigation (direct path to penultimate)
    t_cat = np.linspace(0, 1, 20)
    x_cat = t_cat * (x_final - 0.5)
    y_cat = t_cat * (y_final - 0.5)
    z_cat = t_cat * (z_final - 0.5)

    # Completion morphism (last step)
    x_complete = np.array([x_cat[-1], x_final])
    y_complete = np.array([y_cat[-1], y_final])
    z_complete = np.array([z_cat[-1], z_final])

    # Plot trajectories
    ax3.plot(x_forward, y_forward, z_forward, '-', color='#E63946',
            linewidth=2, alpha=0.5, label='Forward Simulation')

    ax3.plot(x_cat, y_cat, z_cat, 'o-', color='#2E86AB',
            linewidth=2.5, markersize=6, markerfacecolor='white',
            markeredgewidth=1.5, label='Categorical Navigation')

    ax3.plot(x_complete, y_complete, z_complete, '-', color='#F18F01',
            linewidth=4, label='Completion Morphism')

    # Mark states
    ax3.scatter([0], [0], [0], c='#2E86AB', s=200, marker='s',
               edgecolors='black', linewidths=2, label='Initial', zorder=10)
    ax3.scatter([x_cat[-1]], [y_cat[-1]], [z_cat[-1]], c='#F18F01', s=200,
               marker='D', edgecolors='black', linewidths=2,
               label='Penultimate', zorder=10)
    ax3.scatter([x_final], [y_final], [z_final], c='#E63946', s=200,
               marker='*', edgecolors='black', linewidths=2,
               label='Final', zorder=10)

    ax3.set_xlabel('State Dim 1', labelpad=8)
    ax3.set_ylabel('State Dim 2', labelpad=8)
    ax3.set_zlabel('State Dim 3', labelpad=8)
    ax3.set_title('(c) State Space Trajectories')
    ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax3.view_init(elev=20, azim=45)

    # ========== Panel 4: Penultimate State Detection ==========
    ax4 = fig.add_subplot(144)

    # Generate distance to final state over navigation steps
    nav_steps = np.arange(0, 15)

    # Distance decreases logarithmically until penultimate
    distance = 10 * (3 ** (-nav_steps/3))

    # Mark penultimate (one step before final)
    penultimate_idx = 12

    # After penultimate, distance is constant (can't get closer without completion)
    distance[penultimate_idx:] = distance[penultimate_idx]

    ax4.fill_between(nav_steps, 0, distance, alpha=0.2, color='#2E86AB')
    ax4.plot(nav_steps, distance, 'o-', color='#2E86AB',
            linewidth=2.5, markersize=9, markerfacecolor='white',
            markeredgewidth=2)

    # Highlight penultimate state
    ax4.scatter([penultimate_idx], [distance[penultimate_idx]],
               c='#F18F01', s=300, marker='D', edgecolors='black',
               linewidths=2, zorder=10, label='Penultimate State')

    # Mark completion region
    ax4.axvspan(penultimate_idx, nav_steps[-1], alpha=0.1,
               color='#F18F01', label='Completion Region')

    # Add threshold line
    completion_threshold = distance[penultimate_idx] * 1.1
    ax4.axhline(y=completion_threshold, color='#A23B72',
               linestyle='--', linewidth=2, alpha=0.7,
               label='Detection Threshold')

    ax4.set_xlabel('Navigation Steps')
    ax4.set_ylabel('Distance to Final State')
    ax4.set_title('(d) Penultimate Detection')
    ax4.set_yscale('log')
    ax4.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle=':', which='both')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Annotate
    ax4.annotate('Apply single\ncompletion morphism',
                xy=(penultimate_idx + 1, distance[penultimate_idx]),
                xytext=(penultimate_idx - 2, distance[penultimate_idx] * 3),
                fontsize=9, color='#F18F01', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#F18F01', lw=1.5))

    fig.suptitle('Categorical Processor Operation',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"[OK] Processor figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    fig = generate_processor_figure()
    plt.show()
