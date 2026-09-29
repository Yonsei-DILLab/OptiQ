"""Equal-step, direct-policy trajectory comparison for AntMaze v1/v3/v4."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from antmaze_experiments.progress_reward import maze_geometry


ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1] if len(sys.argv) > 1 else 'policy'
assert MODE in ('policy', 'native')
CONTROL_V1 = ROOT.parents[1] / 'antmaze_utd256_euclidean_20260925' / 'v1-50176' / f'{MODE}-natural' / 'rollouts.npz'
TASKS = ('v1', 'v3', 'v4')
VARIANTS = (
    ('control', '100 × distance decrease', '#17689c'),
    ('progress10', '10 × distance decrease', '#d87528'),
    ('dense', '−next distance', '#7d4b9a'),
)
LIMITS = {
    'v1': ((-12, 4), (-8, 8)),
    'v3': ((-18, 18), (-18, 18)),
    'v4': ((-22, 6), (-14, 14)),
}


def load(task, variant):
    if task == 'v1' and variant == 'control':
        path = CONTROL_V1
    elif MODE == 'native':
        path = ROOT / (f'{task}_native_50176.npz' if variant == 'control'
                       else f'{task}_{variant}_native_50176.npz')
    else:
        path = ROOT / f'{task}_{variant}_50176.npz'
    a = np.load(path, allow_pickle=False)
    xy, lengths = a['xy'], a['lengths']
    assert xy.shape[0] == len(lengths) == 40 and int(a['env_steps']) == 50176
    assert not np.any(a['goals'] > 0)
    return [xy[i, :n+1] for i, n in enumerate(lengths)]


fig, axes = plt.subplots(3, 3, figsize=(15.8, 15.4), constrained_layout=True)
for row, task in enumerate(TASKS):
    walls, goals, _ = maze_geometry(task)
    for col, (variant, name, color) in enumerate(VARIANTS):
        ax = axes[row, col]
        trajectories = load(task, variant)
        for x0, y0, x1, y1 in walls:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                   facecolor='#c1c8ce', edgecolor='#87919a', lw=.45, zorder=1))
        for xy in trajectories:
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=.75, alpha=.34, zorder=3)
        starts = np.array([xy[0] for xy in trajectories])
        ends = np.array([xy[-1] for xy in trajectories])
        ax.scatter(starts[:, 0], starts[:, 1], s=10, color='black', alpha=.7, zorder=4)
        ax.scatter(ends[:, 0], ends[:, 1], s=18, marker='x', color='#ba2630',
                   lw=.75, alpha=.8, zorder=5)
        ax.scatter(goals[:, 0], goals[:, 1], s=145, marker='*', color='#f1bd36',
                   edgecolor='black', lw=.5, zorder=6)
        if task == 'v1':
            ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, ls='--', lw=1.1,
                                   edgecolor='#555', zorder=2))
        if task in ('v1', 'v4'):
            upper = sum(((xy[:, 0] < -2) & (xy[:, 1] > 2)).any() for xy in trajectories)
            lower = sum(((xy[:, 0] < -2) & (xy[:, 1] < -2)).any() for xy in trajectories)
            route_note = f'upper/lower gate: {upper}/{lower}'
        else:
            left = sum((xy[:, 0] < -2).any() for xy in trajectories)
            right = sum((xy[:, 0] > 2).any() for xy in trajectories)
            route_note = f'left/right entry: {left}/{right}'
        ax.text(.02, .03, f'0/40 success · {route_note}', transform=ax.transAxes,
                va='bottom', ha='left', fontsize=8.3,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
        ax.set_title(f'{task} · {name}', fontsize=10.4)
        ax.set_xlim(*LIMITS[task][0]); ax.set_ylim(*LIMITS[task][1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.1)
        ax.set_xlabel('x (m)', fontsize=9); ax.set_ylabel('y (m)', fontsize=9)

description = ('direct sampled policy (random z + conditional σ)' if MODE == 'policy'
               else 'native evaluation (random z, μ-only; no conditional σ)')
fig.suptitle(f'OptiQ AntMaze · same 50,176-step checkpoint · {description} · 40 rollouts per panel',
             fontsize=12.6, fontweight='bold')
axes[0, 0].legend(handles=[
    Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=5, label='Start'),
    Line2D([0], [0], marker='x', color='#ba2630', linestyle='none', markersize=6, label='Episode end'),
    Line2D([0], [0], marker='*', color='#f1bd36', markeredgecolor='black',
           linestyle='none', markersize=10, label='Goal'),
], loc='upper right', fontsize=7.8, framealpha=.9)
stem = 'v1_v3_v4_50k_direct_trajectories' if MODE == 'policy' else 'v1_v3_v4_50k_mu_only_trajectories'
fig.savefig(ROOT / f'{stem}.png', dpi=180, bbox_inches='tight')
fig.savefig(ROOT / f'{stem}.pdf', bbox_inches='tight')
