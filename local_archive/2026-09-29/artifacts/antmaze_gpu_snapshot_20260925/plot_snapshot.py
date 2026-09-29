"""Compare the latest raw trajectories for each live AntMaze GPU job."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from antmaze_experiments.progress_reward import maze_geometry
COLORS = {'left': '#226ca3', 'upper': '#226ca3',
          'right': '#e07b39', 'lower': '#e07b39',
          'both': '#8956a0', 'uncommitted': '#9299a1'}
REWARDS = {
    0: '100 Δd − 0.1, γ=.99',
    1: '−d(next), γ=.99',
    2: '100 Δd, γ=.99',
    3: '−d(next), γ=.9',
}


def route(task, xy):
    if task == 'v3':
        left = bool(np.any(xy[:, 0] < -8))
        right = bool(np.any(xy[:, 0] > 8))
        return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
    crossing = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
    if len(crossing) == 0:
        return 'uncommitted'
    i = int(crossing[0])
    denominator = xy[i + 1, 0] - xy[i, 0]
    if abs(denominator) < 1e-12:
        return 'uncommitted'
    y = xy[i, 1] + (-4 - xy[i, 0]) / denominator * (xy[i + 1, 1] - xy[i, 1])
    return 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted'


def load_records():
    snapshot = json.loads((ROOT / 'collection.json').read_text())
    records = {}
    for host in snapshot:
        host_name = host['host']
        for job in host['runs']:
            task = job['job'][:2]
            gpu = int(job['gpu'])
            base = ROOT / 'raw' / host_name / job['campaign'] / 'runs' / job['job']
            config = json.loads((base / 'config.json').read_text())
            progress = json.loads((base / 'progress.json').read_text())
            item = dict(host=host_name, campaign=job['campaign'], job=job['job'],
                        task=task, gpu=gpu, config=config, progress=progress,
                        eval_step=job['eval_step'], modes={})
            if job['eval_step'] is not None:
                evaluation = base / 'evaluations' / f"{job['eval_step']:010d}"
                for kind in ('native', 'policy'):
                    modes = [p for p in evaluation.iterdir() if p.is_dir() and p.name.startswith(kind + '-')]
                    if len(modes) != 1:
                        raise ValueError((job['job'], kind, modes))
                    directory = modes[0]
                    summary = json.loads((directory / 'summary.json').read_text())
                    raw = directory / 'rollouts.npz'
                    with np.load(raw, allow_pickle=False) as data:
                        xy = np.asarray(data['xy'])
                        lengths = np.asarray(data['lengths'])
                        goals = np.asarray(data['goals'])
                        returns = np.asarray(data['returns'])
                        initial = np.asarray(data['initial_full_state'])
                        assert int(data['env_steps']) == job['eval_step']
                        assert bool(data['fixed']) == bool(summary['fixed'])
                    paths = [xy[i, :int(lengths[i]) + 1] for i in range(len(lengths))]
                    assert len(paths) == summary['episodes'] == len(goals) == len(returns)
                    assert all(len(path) > 1 and np.isfinite(path).all() for path in paths)
                    np.testing.assert_allclose(np.array([p[0] for p in paths]), initial[:, :2], atol=1e-4)
                    assert np.isclose(summary['success_rate'], np.mean(goals > 0))
                    assert np.isclose(summary['mean_return'], returns.mean())
                    labels = [route(task, path) for path in paths]
                    counts = Counter(labels)
                    item['modes'][kind] = dict(name=directory.name, summary=summary,
                                               counts=dict(counts), paths=paths,
                                               goals=goals, raw=str(raw.relative_to(ROOT)),
                                               sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            records[(task, gpu)] = item
    assert len(records) == 8, records.keys()
    return records


def draw(records, kind):
    fig, axes = plt.subplots(2, 4, figsize=(25, 11.5), dpi=170)
    fig.patch.set_facecolor('white')
    for row, task in enumerate(('v3', 'v4')):
        walls, goals, bounds = maze_geometry(task)
        for gpu in range(4):
            item = records[(task, gpu)]
            ax = axes[row, gpu]
            for x0, y0, x1, y1 in walls:
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                       facecolor='#e4e6e9', edgecolor='#b9bdc3',
                                       linewidth=.35, zorder=1))
            mode = item['modes'].get(kind)
            if mode:
                for path in mode['paths']:
                    ax.plot(path[:, 0], path[:, 1], c=COLORS[route(task, path)],
                            lw=.75, alpha=.53, zorder=3)
                starts = np.array([p[0] for p in mode['paths']])
                ends = np.array([p[-1] for p in mode['paths']])
                ax.scatter(starts[:, 0], starts[:, 1], s=10, marker='o',
                           facecolors='white', edgecolors='#21252a', lw=.45, zorder=5)
                ax.scatter(ends[:, 0], ends[:, 1], s=8, marker='x',
                           c='#30353c', lw=.4, zorder=5)
                counts = mode['counts']
                names = ('left', 'right', 'both', 'uncommitted') if task == 'v3' else (
                    'upper', 'lower', 'uncommitted')
                route_text = ' / '.join(f'{name[0].upper()} {counts.get(name,0)}' for name in names)
                success = round(mode['summary']['success_rate'] * mode['summary']['episodes'])
                count = mode['summary']['episodes']
                detail = f"{item['eval_step']:,} steps · {route_text}\nsuccess {success}/{count}"
            else:
                detail = 'First 50k evaluation pending'
                ax.text(.5, .52, detail, ha='center', va='center', transform=ax.transAxes,
                        color='#5c646d', fontsize=10)
            ax.scatter(goals[:, 0], goals[:, 1], marker='*', s=105,
                       c='#bd394d', edgecolors='white', linewidths=.45, zorder=7)
            for i, (gx, gy) in enumerate(goals):
                ax.annotate(f'G{i+1}', (gx, gy), xytext=(4, 5), textcoords='offset points',
                            fontsize=8, color='#81212e', fontweight='bold')
            ax.set_xlim(bounds[0] - 1, bounds[2] + 1)
            ax.set_ylim(bounds[1] - 1, bounds[3] + 1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_axisbelow(True)
            ax.grid(color='#e1e3e6', lw=.45, alpha=.7)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('x (m)', fontsize=9)
            ax.set_ylabel('y (m)', fontsize=9)
            start = 'fixed start' if item['config'].get('fixed_full_state_start') else 'random starts'
            ax.set_title(f'{task} · GPU {gpu} · {REWARDS[gpu]} · {start}\n{detail}',
                         fontsize=9.5, loc='left', pad=8)
    label = 'random-z μ-only' if kind == 'native' else 'random-z + conditional σ'
    fig.suptitle(f'Live OptiQ AntMaze: latest 40-rollout trajectories · {label}',
                 fontsize=16, y=.99)
    legend = [Line2D([0], [0], color=COLORS['left'], lw=2, label='v3 left / v4 upper'),
              Line2D([0], [0], color=COLORS['right'], lw=2, label='v3 right / v4 lower'),
              Line2D([0], [0], color=COLORS['both'], lw=2, label='v3 both'),
              Line2D([0], [0], color=COLORS['uncommitted'], lw=2, label='no classified gate'),
              Line2D([0], [0], marker='o', color='none', markeredgecolor='#21252a',
                     markerfacecolor='white', label='start'),
              Line2D([0], [0], marker='*', color='none', markerfacecolor='#bd394d',
                     label='goal')]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(.5, .005), frameon=False)
    fig.subplots_adjust(left=.055, right=.99, bottom=.08, top=.89, wspace=.16, hspace=.32)
    destination = ROOT / f'latest_trajectories_{kind}.png'
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    return destination


def main():
    records = load_records()
    rows = []
    provenance = []
    for task in ('v3', 'v4'):
        for gpu in range(4):
            item = records[(task, gpu)]
            config = item['config']
            for kind in ('native', 'policy'):
                mode = item['modes'].get(kind)
                rows.append(dict(host=item['host'], task=task, gpu=gpu, job=item['job'],
                                 reward=REWARDS[gpu], gamma=config['discount'],
                                 train_starts=config['train_starts'],
                                 eval_starts='fixed' if config.get('fixed_full_state_start') else 'random',
                                 current_step=item['progress'].get('step'),
                                 eval_step=item['eval_step'], mode=kind,
                                 episodes=mode['summary']['episodes'] if mode else None,
                                 success=round(mode['summary']['success_rate'] * mode['summary']['episodes']) if mode else None,
                                 mean_return=mode['summary']['mean_return'] if mode else None,
                                 routes=json.dumps(mode['counts'], sort_keys=True) if mode else None))
                if mode:
                    provenance.append(dict(host=item['host'], gpu=gpu, task=task, mode=kind,
                                           eval_step=item['eval_step'], raw=mode['raw'],
                                           sha256=mode['sha256'], routes=mode['counts'],
                                           summary=mode['summary']))
    with (ROOT / 'latest_metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for kind in ('native', 'policy'):
        print(draw(records, kind))
    for row in rows:
        print(row['task'], row['gpu'], row['mode'], row['current_step'], row['eval_step'],
              row['success'], row['routes'])


if __name__ == '__main__':
    main()
