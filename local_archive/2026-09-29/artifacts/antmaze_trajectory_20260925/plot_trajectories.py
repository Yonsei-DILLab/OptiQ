"""Render verified 100k AntMaze trajectory snapshots from archived raw NPZ files."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from antmaze_experiments.progress_reward import maze_geometry


ROOT = Path(__file__).resolve().parent
STEP = '0000100096'
COLORS = {'left': '#2475a9', 'upper': '#2475a9',
          'right': '#db7725', 'lower': '#db7725',
          'both': '#7764a5', 'uncommitted': '#979ca4'}


def route(task: str, xy: np.ndarray) -> str:
    if task == 'v3':
        left = np.any(xy[:, 0] < -8)
        right = np.any(xy[:, 0] > 8)
        return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
    crossing = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
    if not len(crossing):
        return 'uncommitted'
    i = int(crossing[0])
    y = xy[i, 1] + (-4 - xy[i, 0]) / (xy[i + 1, 0] - xy[i, 0]) * (xy[i + 1, 1] - xy[i, 1])
    return 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted'


def load(task: str, mode: str, random_start: bool):
    if random_start:
        ident = f'{task}-optiq-utd1-basic-euclidean-random-starts-s0'
        base = ROOT / 'raw' / 'random-start' / task
        suffix = 'natural'
    else:
        ident = f'{task}-optiq-utd1-euclidean-NM128-s0'
        base = ROOT / 'raw' / 'fixed-nm128'
        suffix = 'fixed'
    directory = base / 'runs' / ident / 'evaluations' / STEP / f'{mode}-{suffix}'
    npz = directory / 'rollouts.npz'
    summary = directory / 'summary.json'
    with np.load(npz, allow_pickle=False) as data:
        xy = np.asarray(data['xy'], dtype=np.float32)
        goals = np.asarray(data['goals'])
        returns = np.asarray(data['returns'])
        initial = np.asarray(data['initial_full_state'])
        assert int(data['env_steps']) == 100096
        assert len(xy) == len(goals) == len(returns) == len(initial) == 40
        paths = [track[np.isfinite(track).all(axis=1)] for track in xy]
        assert all(len(track) >= 2 and np.isfinite(track).all() for track in paths)
        np.testing.assert_allclose(np.stack([p[0] for p in paths]), initial[:, :2], atol=1e-5)
    info = json.loads(summary.read_text())
    assert info['episodes'] == 40
    assert np.isclose(info['success_rate'], np.mean(goals > 0))
    assert np.isclose(info['mean_return'], returns.mean())
    assert bool(info['fixed']) != random_start
    labels = [route(task, p) for p in paths]
    counts = Counter(labels)
    return paths, counts, info, npz


def render(random_start: bool):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=180)
    fig.patch.set_facecolor('white')
    record = {'step': 100096, 'random_start': random_start, 'panels': {}}
    for row, task in enumerate(('v3', 'v4')):
        walls, goals, bounds = maze_geometry(task)
        for col, mode in enumerate(('native', 'policy')):
            ax = axes[row, col]
            paths, counts, info, npz = load(task, mode, random_start)
            for x0, y0, x1, y1 in walls:
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                       facecolor='#e3e5e8', edgecolor='#b7bbc1', linewidth=.45, zorder=1))
            if random_start:
                ax.add_patch(Rectangle((-2, -2), 4, 4, facecolor='none',
                                       edgecolor='#50545a', linestyle='--', linewidth=1, zorder=2))
            for path in paths:
                color = COLORS[route(task, path)]
                ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.1, alpha=.62, zorder=3)
            starts = np.stack([p[0] for p in paths])
            ends = np.stack([p[-1] for p in paths])
            ax.scatter(starts[:, 0], starts[:, 1], marker='o', s=18,
                       facecolors='white', edgecolors='#141a21', linewidths=.65, zorder=5)
            ax.scatter(ends[:, 0], ends[:, 1], marker='x', s=12,
                       c='#242a32', linewidths=.65, alpha=.75, zorder=5)
            ax.scatter(goals[:, 0], goals[:, 1], marker='*', s=130,
                       c='#b32e40', edgecolors='white', linewidths=.5, zorder=7)
            for i, (gx, gy) in enumerate(goals):
                ax.annotate(f'G{i+1}', (gx,gy), xytext=(4,5), textcoords='offset points',
                            fontsize=8, weight='bold', color='#84202f', zorder=8)
            ax.set_xlim(bounds[0]-1, bounds[2]+1)
            ax.set_ylim(bounds[1]-1, bounds[3]+1)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(color='#e1e3e6', linewidth=.5, alpha=.7, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.tick_params(labelsize=8)
            route_order = ('left','right','uncommitted') if task == 'v3' else ('upper','lower','uncommitted')
            numbers = ' · '.join(f'{name} {counts.get(name,0)}' for name in route_order)
            label = 'random-z μ-only' if mode == 'native' else 'direct policy (σ included)'
            ax.set_title(f'{task} · {label}\n{numbers} · success {round(info["success_rate"]*40)}/40',
                         loc='left', fontsize=10, weight='medium', pad=8)
            record['panels'][f'{task}-{mode}'] = dict(routes=dict(counts),
                success_rate=info['success_rate'], mean_return=info['mean_return'],
                identical_initial_full_state=info['identical_initial_full_state'],
                raw_rollouts=str(npz.relative_to(ROOT)),
                raw_sha256=hashlib.sha256(npz.read_bytes()).hexdigest())
    title = ('Random starts: N=M64, T=1' if random_start else
             'Fixed starts: N=M128, T=1')
    fig.suptitle(f'OptiQ AntMaze trajectories after 100k transitions — {title}',
                 fontsize=15, weight='medium', y=.987)
    legend = [Line2D([0],[0],color=COLORS['left'],lw=2,label='v3 left / v4 upper'),
              Line2D([0],[0],color=COLORS['right'],lw=2,label='v3 right / v4 lower'),
              Line2D([0],[0],color=COLORS['uncommitted'],lw=2,label='no gate entry'),
              Line2D([0],[0],marker='o',color='none',markeredgecolor='#141a21',
                     markerfacecolor='white',markersize=5,label='start'),
              Line2D([0],[0],marker='*',color='none',markerfacecolor='#b32e40',
                     markeredgecolor='white',markersize=9,label='goal')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.015),
               ncol=5,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,bottom=.075,top=.93,wspace=.20,hspace=.22)
    filename = 'random_start_nm64_100k.png' if random_start else 'fixed_start_nm128_100k.png'
    fig.savefig(ROOT / filename, dpi=180)
    plt.close(fig)
    return filename, record


if __name__ == '__main__':
    name_random, random_record = render(True)
    name_fixed, fixed_record = render(False)
    report = {'random_start': random_record, 'fixed_start': fixed_record,
              'figures': [name_random, name_fixed],
              'report_source': 'post-hoc figure script; no frozen training source modified'}
    (ROOT/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name_random, name_fixed)
