from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from antmaze_experiments.progress_reward import maze_geometry


ROOT = Path(__file__).resolve().parent
WALLS, GOALS, BOUNDS = maze_geometry('v1')
MODES = [
    ('native-natural', 'Random-z μ-only', '#1468a2'),
    ('policy-natural', 'Direct sampled policy (z + σ)', '#e46b20'),
]


def decorate(ax, zoom):
    for x0, y0, x1, y1 in WALLS:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                               facecolor='#b8bec6', edgecolor='#77808a', lw=.65, zorder=1))
    ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, ls='--', lw=1.2,
                           edgecolor='#706e70', zorder=3))
    ax.scatter(GOALS[:, 0], GOALS[:, 1], marker='*', s=140, color='#f1bd36',
               edgecolor='black', lw=.6, zorder=8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim((-3.5, 3.1) if zoom else (BOUNDS[0], BOUNDS[2]))
    ax.set_ylim((-4.5, 4.5) if zoom else (BOUNDS[1], BOUNDS[3]))
    ax.set_xlabel('Ant x (m)')
    ax.set_ylabel('Ant y (m)')
    ax.grid(alpha=.13)


fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.1), constrained_layout=True)
for col, (mode, title, color) in enumerate(MODES):
    data = np.load(ROOT / mode / 'rollouts.npz', allow_pickle=False)
    xy, lengths, goals = data['xy'], data['lengths'], data['goals']
    assert xy.shape == (40, 501, 2) and np.all(lengths == 500)
    assert np.all(goals == 0)
    ends = np.stack([xy[i, n] for i, n in enumerate(lengths)])
    for row in range(2):
        ax = axes[row, col]
        decorate(ax, zoom=row == 1)
        for i, n in enumerate(lengths):
            ax.plot(xy[i, :n+1, 0], xy[i, :n+1, 1], color=color,
                    alpha=.29 if row == 1 else .34, lw=.72, zorder=4)
        ax.scatter(xy[:, 0, 0], xy[:, 0, 1], marker='o', s=11,
                   facecolor='black', alpha=.58, zorder=6)
        ax.scatter(ends[:, 0], ends[:, 1], marker='x', s=20,
                   color='#bd2530', lw=.8, alpha=.75, zorder=7)
        ax.set_title(f'{title} · {"start-area zoom" if row else "full maze"} · 0/40 success',
                     fontsize=10.5)

fig.suptitle('AntMaze v1 · 50,176 environment steps · OptiQ UTD=1, T=1 · 40 rollouts per mode',
             fontsize=13, fontweight='bold')
axes[0, 0].legend(handles=[
    Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=5,
           label='Randomized start'),
    Line2D([0], [0], marker='x', color='#bd2530', linestyle='none', markersize=6,
           label='500-step timeout'),
    Line2D([0], [0], marker='*', color='#f1bd36', markeredgecolor='black',
           linestyle='none', markersize=10, label='Goal'),
], loc='upper right', fontsize=8, framealpha=.9)
fig.savefig(ROOT / 'v1_50176_trajectories.png', dpi=180, bbox_inches='tight')
fig.savefig(ROOT / 'v1_50176_trajectories.pdf', bbox_inches='tight')
