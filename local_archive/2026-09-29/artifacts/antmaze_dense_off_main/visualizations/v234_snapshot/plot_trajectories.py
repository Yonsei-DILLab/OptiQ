"""Plot recorded OptiQ evaluations; never run or modify training."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from antmaze_experiments.dense_off_report import GOALS, geometry

TASKS = ['v2', 'v3', 'v4']
STEPS = {'v2': 1250048, 'v3': 2000128, 'v4': 1250048}


def read(task, mode):
    path = ROOT / 'raw' / task / mode / 'rollouts.npz'
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    s = json.loads((path.parent / 'summary.json').read_text())
    xy, lengths, goals = d['xy'], d['lengths'], d['goals']
    assert int(d['env_steps']) == STEPS[task] and not bool(d['fixed'])
    assert len(xy) == len(lengths) == len(goals) == s['episodes'] == 20
    assert len(np.unique(d['initial_full_state'], axis=0)) == 20
    for line, n in zip(xy, lengths):
        assert np.isfinite(line[:n+1]).all() and np.isnan(line[n+1:]).all()
    dist = np.linalg.norm(xy[:, :, None, :] - np.array(GOALS[task])[None, None], axis=-1)
    reached = np.nanmin(dist, axis=(1, 2)) <= .50002
    assert np.array_equal(reached, goals > 0)
    assert np.allclose(-np.nansum(dist[:, 1:].min(axis=-1), axis=1), d['returns'], rtol=2e-6, atol=.003)
    for i, goal in enumerate(goals):
        if goal:
            assert dist[i, lengths[i], goal-1] <= .50002
    left = np.any(xy[:, :, 0] < -4, axis=1)
    right = np.any(xy[:, :, 0] > 4, axis=1)
    gate_x = -6 if task == 'v3' else -2
    upper = np.any((xy[:, :, 0] < gate_x) & (xy[:, :, 1] < -2), axis=1)
    lower = np.any((xy[:, :, 0] < gate_x) & (xy[:, :, 1] > 2), axis=1)
    m = dict(step=STEPS[task], episodes=20, success_count=int(reached.sum()),
             goal_counts={str(g): int((goals == g).sum()) for g in range(3)},
             left4_entries=int(left.sum()), right4_entries=int(right.sum()),
             upper_entries=int(upper.sum()), lower_entries=int(lower.sum()),
             both_entries=int((upper & lower).sum()), neither_entries=int((~upper & ~lower).sum()),
             upper_lower_gate_x=gate_x,
             nearest_each_goal_m=np.nanmin(dist, axis=(0, 1)).tolist(),
             maximum_left_x=float(np.nanmin(xy[:, :, 0])),
             mean_return=float(d['returns'].mean()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return d, m


def draw_maze(ax, task):
    maze, rr, cc = geometry(task)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle(((j-cc)*4-2, (i-rr)*4-2), 4, 4,
                                      facecolor='#e3e8ed', edgecolor='#c1cbd3', lw=.6, zorder=0))
    ax.add_patch(Rectangle((-2, -2), 4, 4, facecolor='#dceee3', alpha=.6, lw=0, zorder=1))
    for i, goal in enumerate(GOALS[task], 1):
        ax.add_patch(Circle(goal, .5, facecolor='#edc763', alpha=.65, edgecolor='#967210', zorder=6))
        ax.scatter(*goal, marker='*', s=170, c='#dca817', edgecolors='#8b6600', lw=.6, zorder=7)
        ax.annotate(f'G{i}', goal, xytext=(9, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#8b6600', zorder=9)
    ax.set_xlim(-cc*4-2, (len(maze[0])-1-cc)*4+2)
    ax.set_ylim((len(maze)-1-rr)*4+2, -rr*4-2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(alpha=.12, lw=.5)


def draw(mode, name, title):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2))
    fig.subplots_adjust(left=.045, right=.987, top=.79, bottom=.26, wspace=.2)
    results = {}
    for ax, task in zip(axes, TASKS):
        d, m = read(task, mode)
        results[task] = m
        draw_maze(ax, task)
        for line, n, goal in zip(d['xy'], d['lengths'], d['goals']):
            line = line[:n+1]
            ax.plot(line[:, 0], line[:, 1], color='#19845a' if goal else '#287db1',
                    lw=.95, alpha=.65, zorder=3)
            ax.scatter(*line[-1], marker='o' if goal else 'x', s=19 if goal else 24,
                       color='#167745' if goal else '#ca3745', alpha=.85, lw=1, zorder=8)
        ax.scatter(d['xy'][:, 0, 0], d['xy'][:, 0, 1], marker='^', s=21, c='#233444',
                   edgecolor='white', lw=.4, zorder=9)
        ax.set_title(f"{task.upper()} | {STEPS[task]/1000:,.0f}k steps | success {m['success_count']}/20\n"
                     f"Goal G1: {m['goal_counts']['1']}   |   Goal G2: {m['goal_counts']['2']}",
                     fontsize=12, pad=12)
        note = (f"Right branch {m['right4_entries']}/20; left branch {m['left4_entries']}/20" if task == 'v2' else
                f"Left branch {m['left4_entries']}/20; right branch {m['right4_entries']}/20" if task == 'v3' else
                f"Upper entry {m['upper_entries']}; lower entry {m['lower_entries']}; neither {m['neither_entries']}")
        ax.text(.5, -.18, note, transform=ax.transAxes, ha='center', fontsize=10.5)
    fig.suptitle('OptiQ | AntMaze v2 / v3 / v4 | latest saved evaluations at capture\n'
                 f'{title} | dense reward + NovelD OFF | seed 0', fontsize=15, y=.967, linespacing=1.5)
    handles = [Line2D([], [], color='#19845a', lw=2, label='Successful rollout'),
               Line2D([], [], color='#287db1', lw=2, label='Failed rollout'),
               Line2D([], [], marker='^', color='#233444', ls='', label='Random start'),
               Line2D([], [], marker='x', color='#ca3745', ls='', label='Failed endpoint'),
               Line2D([], [], marker='*', color='#dca817', ls='', markersize=12, label='Goal')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .079), ncol=5, frameon=False, fontsize=10)
    fig.text(.5, .025, '20 rollouts per panel, random starts in [-2, 2]^2; all failures included. No external DACER noise.\n'
             'Snapshots have different training budgets. Branch/corridor entry does not imply successful paths or state-conditional multimodality.',
             ha='center', fontsize=9.5, color='#48545e', linespacing=1.4)
    fig.savefig(ROOT / (name+'.png'), dpi=180, facecolor='white')
    fig.savefig(ROOT / (name+'.pdf'), facecolor='white')
    plt.close(fig)
    return results


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    result = dict(source_commit='26336810f7ea4ca61210ea70c6aeeae9f7acaed0',single_training_seed=0,
                  snapshot_utc='2026-09-23T07:06:47Z',
                  route_definition='v2/v3 branch: any x<-4 or x>4. v4 entry: x<-2 and y<-2 or y>2. Counts can overlap.',
                  primary=draw('policy-natural', 'trajectories_direct_policy', 'Direct policy: random z + conditional sigma'),
                  supplementary=draw('native-natural', 'trajectories_mu_only', 'Supplement: random z, mu-only'))
    for task in TASKS:
        c=json.loads((ROOT/'raw'/task/'config.json').read_text())
        assert c['source_commit']==result['source_commit'] and c['reward_profile']=='dense' and not c['noveld_enabled']
        a,_=read(task,'policy-natural');b,_=read(task,'native-natural')
        assert np.array_equal(a['initial_full_state'],b['initial_full_state'])
    (ROOT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
