from __future__ import annotations

from collections import Counter, defaultdict
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
RAW = ROOT / 'fixed-start' / 'raw'
OUT = ROOT / 'fixed-start'
MODES = ('native-fixed', 'policy-fixed')
MODE_TITLES = {
    'native-fixed': 'μ-only action, fresh random z (primary OptiQ eval)',
    'policy-fixed': 'conditional σ included (supplementary)',
}
RUNS = {
    ('v3', '100Δd'): ('v3-100delta-no-cost-step300032', 300032),
    ('v3', '100Δd − 0.1'): ('v3-100delta-cost01-step300032', 300032),
    ('v3', '−d(next)'): ('v3-negative-distance-step100096', 100096),
    ('v4', '100Δd'): ('v4-100delta-no-cost-step200192', 200192),
    ('v4', '100Δd − 0.1'): ('v4-100delta-cost01-step250112', 250112),
    ('v4', '−d(next)'): ('v4-negative-distance-step250112', 250112),
}
COLORS = {
    'left': '#2475a9', 'upper': '#2475a9',
    'right': '#db7725', 'lower': '#db7725',
    'both': '#8064a2', 'uncommitted': '#a0a4aa',
}


def classify_route(task: str, xy: np.ndarray) -> str:
    if task == 'v3':
        left = bool(np.any(xy[:, 0] < -8))
        right = bool(np.any(xy[:, 0] > 8))
        return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
    crossing = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
    if not len(crossing):
        return 'uncommitted'
    i = int(crossing[0])
    dy = xy[i + 1, 1] - xy[i, 1]
    dx = xy[i + 1, 0] - xy[i, 0]
    y = xy[i, 1] + (-4 - xy[i, 0]) / dx * dy
    return 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted'


def load_panel(task: str, reward: str, mode: str):
    folder, step = RUNS[(task, reward)]
    directory = RAW / folder / 'evaluations' / f'{step:010d}' / mode
    summary = json.loads((directory / 'summary.json').read_text())
    verification = json.loads((RAW / folder / 'verification.json').read_text())
    assert verification['passed'] and verification['identical_initial_full_state']
    assert verification['training_checkpoint_unchanged'] and verification['restored_policy_exact']
    with np.load(directory / 'rollouts.npz', allow_pickle=False) as data:
        xy = np.asarray(data['xy'], dtype=np.float32)
        goals = np.asarray(data['goals'], dtype=np.int64)
        returns = np.asarray(data['returns'], dtype=np.float64)
        lengths = np.asarray(data['lengths'], dtype=np.int64)
        initial = np.asarray(data['initial_full_state'], dtype=np.float64)
        assert bool(data['fixed']) and data['mode'].item() == ('native' if mode == 'native-fixed' else 'policy')
        assert int(data['env_steps']) == step
    rel_npz = (directory / 'rollouts.npz').relative_to(RAW / folder).as_posix()
    actual_sha = hashlib.sha256((directory / 'rollouts.npz').read_bytes()).hexdigest()
    assert verification['raw_sha256'][rel_npz] == actual_sha
    np.testing.assert_allclose(initial, np.broadcast_to(initial[:1], initial.shape), atol=0, rtol=0)
    np.testing.assert_allclose(initial[:, :2], 0., atol=1e-8, rtol=0)
    assert len(xy) == len(goals) == len(returns) == len(lengths) == 40
    paths = [track[np.isfinite(track).all(axis=1)] for track in xy]
    labels = [classify_route(task, track) for track in paths]
    # A terminal success is shorter than the fixed 700-step time limit.
    success = lengths < 700
    assert int(success.sum()) == int(round(summary['success_rate'] * 40))
    assert np.isclose(returns.mean(), summary['mean_return'])
    assert summary['identical_initial_full_state'] is True
    route_counts = Counter(labels)
    success_by_route = defaultdict(lambda: {'success': 0, 'episodes': 0})
    for route, ok in zip(labels, success):
        success_by_route[route]['episodes'] += 1
        success_by_route[route]['success'] += int(ok)
    info = {
        'task': task, 'reward': reward, 'mode': mode, 'step': step,
        'episodes': len(paths), 'successes': int(success.sum()),
        'success_rate': float(success.mean()), 'mean_return': float(returns.mean()),
        'mean_length': float(lengths.mean()),
        'route_counts': dict(route_counts),
        'success_by_route': dict(success_by_route),
        'goal_id_counts': {str(k): int(v) for k, v in zip(*np.unique(goals, return_counts=True))},
        'raw_npz': str(directory.relative_to(ROOT)),
        'raw_sha256': actual_sha,
        'verification': verification,
    }
    return paths, labels, info


def draw_maze(ax, task: str):
    walls, goals, bounds = maze_geometry(task)
    for x0, y0, x1, y1 in walls:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                               facecolor='#e3e5e8', edgecolor='#b7bbc1',
                               linewidth=.45, zorder=1))
    ax.scatter(goals[:, 0], goals[:, 1], marker='*', s=150,
               c='#b32e40', edgecolors='white', linewidths=.5, zorder=8)
    for i, (gx, gy) in enumerate(goals):
        ax.annotate(f'G{i+1}', (gx, gy), xytext=(4, 5), textcoords='offset points',
                    fontsize=8, weight='bold', color='#84202f', zorder=9)
    ax.scatter([0], [0], marker='o', s=40, facecolors='white',
               edgecolors='#121820', linewidths=1.1, zorder=10)
    ax.set_xlim(bounds[0]-1, bounds[2]+1)
    ax.set_ylim(bounds[1]-1, bounds[3]+1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#e1e3e6', linewidth=.5, alpha=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.tick_params(labelsize=8)


def make_figure(mode: str):
    rewards = ('100Δd', '100Δd − 0.1', '−d(next)')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=180)
    fig.patch.set_facecolor('white')
    record = {'mode': mode, 'panels': {}}
    for row, task in enumerate(('v3', 'v4')):
        for col, reward in enumerate(rewards):
            ax = axes[row, col]
            draw_maze(ax, task)
            paths, labels, info = load_panel(task, reward, mode)
            for path, label in zip(paths, labels):
                ax.plot(path[:, 0], path[:, 1], color=COLORS[label], linewidth=.85,
                        alpha=.42, zorder=3)
                ax.scatter(path[-1, 0], path[-1, 1], s=8, c=COLORS[label],
                           alpha=.45, linewidths=0, zorder=4)
            routes = ('left', 'right', 'both', 'uncommitted') if task == 'v3' else ('upper', 'lower', 'uncommitted')
            route_text = ' · '.join(f'{name} {info["route_counts"].get(name, 0)}' for name in routes)
            ax.set_title(f'{task} · {reward} · step {info["step"]:,}\n'
                         f'success {info["successes"]}/40 · {route_text}',
                         loc='left', fontsize=9.5, weight='medium', pad=7)
            record['panels'][f'{task}-{reward}'] = info
    label = MODE_TITLES[mode]
    fig.suptitle('OptiQ AntMaze · same fixed origin/full state · 40 rollouts per panel\n' + label,
                 fontsize=15, weight='medium', y=.99)
    legend = [Line2D([0], [0], color=COLORS['left'], lw=2, label='v3 left / v4 upper branch'),
              Line2D([0], [0], color=COLORS['right'], lw=2, label='v3 right / v4 lower branch'),
              Line2D([0], [0], color=COLORS['both'], lw=2, label='v3 visited both sides'),
              Line2D([0], [0], color=COLORS['uncommitted'], lw=2, label='no classified gate entry'),
              Line2D([0], [0], marker='o', color='none', markeredgecolor='#121820',
                     markerfacecolor='white', markersize=6, label='fixed start (0,0)'),
              Line2D([0], [0], marker='*', color='none', markeredgecolor='white',
                     markerfacecolor='#b32e40', markersize=10, label='goal')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .015),
               ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, bottom=.075, top=.91, wspace=.16, hspace=.21)
    filename = 'trajectory_fixed_mu_only.png' if mode == 'native-fixed' else 'trajectory_fixed_sigma_included.png'
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)
    return filename, record


if __name__ == '__main__':
    files, panels = zip(*(make_figure(mode) for mode in MODES))
    report = {
        'title': 'Fixed-start OptiQ AntMaze v3/v4 reward comparison',
        'fixed_initial_xy': [0.0, 0.0],
        'full_initial_state_identical_within_each_panel': True,
        'episodes_per_panel': 40,
        'evaluation_only': True,
        'training_checkpoints_unchanged': True,
        'modes': {p['mode']: p['panels'] for p in panels},
        'figures': list(files),
        'route_definition': {
            'v3': 'left if any x < -8m; right if any x > 8m; both if either; else uncommitted',
            'v4': 'first crossing x=-4m; upper if crossing y > 2m, lower if y < -2m, else uncommitted',
        },
    }
    (OUT / 'fixed_start_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(files)
