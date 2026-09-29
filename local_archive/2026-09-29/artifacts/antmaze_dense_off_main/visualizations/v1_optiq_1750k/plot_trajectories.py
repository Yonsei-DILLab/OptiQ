"""Read-only visualization of the saved 1,750,016-step evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from antmaze_experiments.dense_off_report import geometry

COLORS = {'upper': '#1976b6', 'lower': '#e08425', 'both': '#8c55a5', 'neither': '#7e8994'}
GOAL = np.array([-8., 0.])
STEP = 1750016


def read(mode):
    path = ROOT / 'raw' / mode / 'rollouts.npz'
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    summary = json.loads((path.parent / 'summary.json').read_text())
    xy, lengths, goals = d['xy'], d['lengths'], d['goals']
    assert int(d['env_steps']) == STEP and not bool(d['fixed'])
    assert len(xy) == len(lengths) == len(goals) == summary['episodes'] == 20
    for line, length in zip(xy, lengths):
        assert np.isfinite(line[:length+1]).all() and np.isnan(line[length+1:]).all()
    distance = np.linalg.norm(xy - GOAL, axis=-1)
    assert np.array_equal(np.nanmin(distance, axis=1) <= .50002, goals > 0)
    assert np.allclose(-np.nansum(distance[:, 1:], axis=1), d['returns'], rtol=2e-6, atol=.003)
    assert len(np.unique(d['initial_full_state'], axis=0)) == 20
    upper = np.any((xy[:, :, 0] < -2) & (xy[:, :, 1] < -2), axis=1)
    lower = np.any((xy[:, :, 0] < -2) & (xy[:, :, 1] > 2), axis=1)
    categories = np.where(upper & lower, 'both', np.where(upper, 'upper', np.where(lower, 'lower', 'neither')))
    metrics = dict(step=STEP, episodes=len(xy), success_count=int((goals > 0).sum()),
                   upper_only=int((categories == 'upper').sum()), lower_only=int((categories == 'lower').sum()),
                   both=int((categories == 'both').sum()), neither=int((categories == 'neither').sum()),
                   closest_goal_m=float(np.nanmin(distance)), mean_return=float(d['returns'].mean()),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   route_definition='Entry: x < -2 and y < -2 (upper) or y > 2 (lower); entry is not a successful route.')
    return d, categories, metrics


def background(ax):
    maze, rr, cc = geometry('v1')
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle(((j-cc)*4-2, (i-rr)*4-2), 4, 4,
                                      facecolor='#e7ebef', edgecolor='#bcc5ce', lw=.7, zorder=0))
    ax.add_patch(Rectangle((-2, -2), 4, 4, facecolor='#e7f3ec', alpha=.48, lw=0, zorder=.5))
    ax.add_patch(Circle(GOAL, .5, facecolor='#f2cb65', edgecolor='#ae850e', alpha=.5, lw=1, zorder=5))
    ax.scatter(*GOAL, marker='*', s=230, facecolor='#d6a719', edgecolor='#806016', linewidth=.7, zorder=6)
    ax.text(-8, 1.10, 'Goal (r = 0.5 m)', ha='center', fontsize=10, color='#796017')
    ax.text(-4, -5.6, 'Upper corridor', ha='center', fontsize=10, color='#426174')
    ax.text(-4, 5.55, 'Lower corridor', ha='center', fontsize=10, color='#946128')
    ax.set_xlim(-10.25, 2.75)
    ax.set_ylim(6.1, -6.1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m); upper route = negative y')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.grid(alpha=.13, lw=.6)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2))
    fig.subplots_adjust(left=.06, right=.98, bottom=.21, top=.79, wspace=.18)
    all_metrics = {}
    starts = []
    for ax, mode, title in zip(axes, ['policy-natural', 'native-natural'],
                              ['Direct policy: random z + conditional sigma', 'Supplement: random z, mu-only']):
        d, categories, metrics = read(mode)
        starts.append(d['initial_full_state'])
        all_metrics[mode] = metrics
        background(ax)
        for line, length, goal, cat in zip(d['xy'], d['lengths'], d['goals'], categories):
            line = line[:length+1]
            ax.plot(line[:, 0], line[:, 1], color=COLORS[cat], alpha=.66, lw=1.0, zorder=2)
            ax.scatter(*line[-1], marker='o' if goal else 'x', s=32 if goal else 27,
                       c='#278851' if goal else '#c83943', lw=1.2, alpha=.85, zorder=7)
        ax.scatter(d['xy'][:, 0, 0], d['xy'][:, 0, 1], marker='^', s=29,
                   c='#233444', edgecolor='white', linewidth=.45, zorder=8)
        ax.set_title(title + f"\nSuccess {metrics['success_count']}/20 | nearest goal {metrics['closest_goal_m']:.2f} m",
                     fontsize=12, pad=13)
        ax.text(.5, -.17, f"Corridor entries: upper {metrics['upper_only']}, lower {metrics['lower_only']}, "
                f"neither {metrics['neither']} / 20", ha='center', transform=ax.transAxes, fontsize=11)
    assert np.array_equal(*starts), 'Both modes must use the same sampled starting states.'
    fig.suptitle('OptiQ | AntMaze v1 | latest saved evaluation at capture\n'
                 '1,750,016 environment steps (54,432 learner updates) | dense reward + NovelD OFF | seed 0',
                 fontsize=15, y=.965, linespacing=1.5)
    handles = [Line2D([], [], color=COLORS['upper'], lw=2, label='Upper entry'),
               Line2D([], [], color=COLORS['lower'], lw=2, label='Lower entry'),
               Line2D([], [], color=COLORS['neither'], lw=2, label='Neither'),
               Line2D([], [], color='#233444', marker='^', ls='', label='Random start'),
               Line2D([], [], color='#c83943', marker='x', ls='', label='Failed endpoint'),
               Line2D([], [], color='#d6a719', marker='*', markersize=12, ls='', label='Goal')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .07), ncol=6, frameon=False, fontsize=10)
    fig.text(.5, .025, '20 rollouts per mode; 500-step limit. Starts sampled in [-2, 2]^2; all failures shown.\n'
             'No external DACER exploration noise. Corridor entry counts do not imply successful routes.',
             ha='center', fontsize=10, color='#48545e', linespacing=1.4)
    fig.savefig(ROOT / 'trajectories_1750k.png', dpi=180, facecolor='white')
    fig.savefig(ROOT / 'trajectories_1750k.pdf', facecolor='white')
    plt.close(fig)
    config = json.loads((ROOT / 'raw' / 'config.json').read_text())
    output = dict(source_commit=config['source_commit'],task='v1',method='optiq',seed=0,
                  step=STEP,learner_updates=(STEP-8192)//32,primary='policy-natural',
                  same_start_samples_across_modes=True,metrics=all_metrics)
    (ROOT / 'metrics.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
