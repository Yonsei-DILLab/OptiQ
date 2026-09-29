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
RAW = ROOT / 'raw' / 'latest'
CAMPAIGN_PAIR = 'antmaze-optiq-utd1-random-reward-pair-v34-s0-20260925'
CAMPAIGN_BASE = 'antmaze-optiq-utd1-random-progress100-baseline-replacement-v34-s0-20260925'
CONDITIONS = {
    'progress100_no_step': {
        'label': '100Δd · no step cost', 'campaign': CAMPAIGN_BASE,
        'run': {'v3': 'v3-optiq-utd1-basic-euclidean-random-starts-s0-replacement',
                'v4': 'v4-optiq-utd1-basic-euclidean-random-starts-s0-replacement'},
    },
    'progress100_step01': {
        'label': '100Δd − 0.1', 'campaign': CAMPAIGN_PAIR,
        'run': {'v3': 'v3-optiq-utd1-random-progress100_cost01-nm64-s0',
                'v4': 'v4-optiq-utd1-random-progress100_cost01-nm64-s0'},
    },
    'negative_distance': {
        'label': '−d(next)', 'campaign': CAMPAIGN_PAIR,
        'run': {'v3': 'v3-optiq-utd1-random-negative_distance-nm64-s0',
                'v4': 'v4-optiq-utd1-random-negative_distance-nm64-s0'},
    },
}
MODES = {'native-natural': 'random-z μ-only', 'policy-natural': 'direct policy (σ included)'}
ROUTE_COLORS = {
    'left': '#2475a9', 'upper': '#2475a9', 'right': '#db7725', 'lower': '#db7725',
    'both': '#7764a5', 'uncommitted': '#979ca4',
}


def classify_route(task: str, xy: np.ndarray) -> str:
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


def data_for(task: str, condition: str, mode: str):
    meta = CONDITIONS[condition]
    run = meta['run'][task]
    run_dir = RAW / meta['campaign'] / 'runs' / run
    evaluations = sorted((run_dir / 'evaluations').glob('*/' + mode), key=lambda p: p.parent.name)
    if not evaluations:
        raise FileNotFoundError(f'No evaluation {task} {condition} {mode}')
    eval_dir = evaluations[-1]
    info = json.loads((eval_dir / 'summary.json').read_text())
    npz = eval_dir / 'rollouts.npz'
    with np.load(npz, allow_pickle=False) as z:
        xy = np.asarray(z['xy'], dtype=np.float32)
        returns = np.asarray(z['returns'], dtype=np.float64)
        goals = np.asarray(z['goals'])
        lengths = np.asarray(z['lengths'])
        initial = np.asarray(z['initial_full_state'])
        step = int(z['env_steps'])
        paths = [track[np.isfinite(track).all(axis=1)] for track in xy]
        assert len(paths) == info['episodes'] == 40
        assert step == info['step']
        assert all(len(path) >= 2 and np.isfinite(path).all() for path in paths)
        np.testing.assert_allclose(np.stack([path[0] for path in paths]), initial[:, :2], atol=1e-5)
        assert np.isfinite(returns).all()
    routes = [classify_route(task, path) for path in paths]
    route_counts = Counter(routes)
    route_by_goal = {}
    for goal in sorted(set(int(value) for value in goals)):
        route_by_goal[str(goal)] = dict(Counter(label for value, label in zip(goals, routes)
                                               if int(value) == goal))
    assert np.isclose(float(np.mean(goals > 0)), info['success_rate'])
    assert np.isclose(float(np.mean(returns)), info['mean_return'])
    p = run_dir / 'progress.json'
    progress = json.loads(p.read_text()) if p.exists() else {}
    return dict(paths=paths, info=info, step=step, routes=routes, route_counts=dict(route_counts),
                return_mean=float(np.mean(returns)), success_count=int(np.sum(goals > 0)),
                goal_counts={str(k): int(v) for k, v in Counter(goals.tolist()).items()},
                route_by_goal=route_by_goal,
                episode_lengths=lengths.tolist(), progress=progress,
                sha256=hashlib.sha256(npz.read_bytes()).hexdigest(), raw=str(npz.relative_to(ROOT)))


def draw(task: str, ax, result, title: str):
    walls, goals, bounds = maze_geometry(task)
    for x0, y0, x1, y1 in walls:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor='#e3e5e8',
                               edgecolor='#b7bbc1', linewidth=.45, zorder=1))
    ax.add_patch(Rectangle((-2, -2), 4, 4, facecolor='none', edgecolor='#50545a',
                           linestyle='--', linewidth=.9, zorder=2))
    for path, label in zip(result['paths'], result['routes']):
        ax.plot(path[:, 0], path[:, 1], color=ROUTE_COLORS[label], linewidth=1.05,
                alpha=.55, zorder=3)
    starts = np.stack([p[0] for p in result['paths']])
    ends = np.stack([p[-1] for p in result['paths']])
    ax.scatter(starts[:, 0], starts[:, 1], marker='o', s=13, facecolors='white',
               edgecolors='#141a21', linewidths=.45, zorder=5)
    ax.scatter(ends[:, 0], ends[:, 1], marker='x', s=10, c='#242a32', linewidths=.55,
               alpha=.7, zorder=5)
    ax.scatter(goals[:, 0], goals[:, 1], marker='*', s=115, c='#b32e40',
               edgecolors='white', linewidths=.4, zorder=7)
    for i, (gx, gy) in enumerate(goals):
        ax.annotate(f'G{i+1}', (gx, gy), xytext=(4, 5), textcoords='offset points',
                    fontsize=7, weight='bold', color='#84202f', zorder=8)
    ax.set_xlim(bounds[0]-1, bounds[2]+1)
    ax.set_ylim(bounds[1]-1, bounds[3]+1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#e1e3e6', linewidth=.45, alpha=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    if task == 'v3':
        route_line = 'left {left} · right {right} · both {both} · no gate {uncommitted}'.format(
            **{k: result['route_counts'].get(k, 0) for k in ('left','right','both','uncommitted')})
    else:
        route_line = 'upper {upper} · lower {lower} · no gate {uncommitted}'.format(
            **{k: result['route_counts'].get(k, 0) for k in ('upper','lower','uncommitted')})
    ax.set_title(f'{task} · {result["step"]:,} transitions · success {result["success_count"]}/40\n'
                 f'{route_line}', loc='left', fontsize=8.4, pad=5)


def render(mode: str):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.3), dpi=190)
    fig.patch.set_facecolor('white')
    report = {}
    for col, (condition, metadata) in enumerate(CONDITIONS.items()):
        report[condition] = {}
        fig.text(.055 + .935 * (col + .5) / 3, .91, metadata['label'],
                 ha='center', va='bottom', fontsize=11, weight='bold')
        for row, task in enumerate(('v3', 'v4')):
            result = data_for(task, condition, mode)
            draw(task, axes[row, col], result, metadata['label'])
            report[condition][task] = {k: v for k, v in result.items() if k != 'paths'}
    mode_label = MODES[mode]
    fig.suptitle(f'AntMaze OptiQ trajectories · {mode_label}\n'
                 '40 fresh random starts per checkpoint · seed 0',
                 fontsize=15, weight='bold', y=.99)
    handles = [
        Line2D([0], [0], color=ROUTE_COLORS['left'], lw=2, label='v3 left / v4 upper route'),
        Line2D([0], [0], color=ROUTE_COLORS['right'], lw=2, label='v3 right / v4 lower route'),
        Line2D([0], [0], color=ROUTE_COLORS['uncommitted'], lw=2, label='did not enter a route gate'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='#141a21', markerfacecolor='white',
               markersize=5, label='random start'),
        Line2D([0], [0], marker='*', color='none', markerfacecolor='#b32e40', markeredgecolor='white',
               markersize=9, label='goal'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005), ncol=5,
               frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=.055, right=.99, bottom=.075, top=.865, wspace=.20, hspace=.24)
    out = ROOT / ('trajectory_mu_only_latest.png' if mode == 'native-natural' else 'trajectory_sigma_included_latest.png')
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return report, out


if __name__ == '__main__':
    report = {'source_commits': {'reward_pair': '75874066c432c48501053eda055c1b71e61a85da',
                                 'baseline_replacement': '914cddcdd1e210c02dd333c195d888035beb173f'},
              'interpretation': 'Seed 0; random initial XY per episode. Route counts combine different starts and do not establish same-state multimodality.'}
    for mode in MODES:
        data, image = render(mode)
        report[mode] = data
        print(image)
    (ROOT / 'latest_report.json').write_text(json.dumps(report, indent=2) + '\n')
