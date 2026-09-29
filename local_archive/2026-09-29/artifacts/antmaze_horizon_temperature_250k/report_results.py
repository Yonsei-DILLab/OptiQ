"""Post-hoc route validation for each gamma/T condition; never train or evaluate."""
from collections import Counter
from datetime import datetime, timezone
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
ARTIFACTS = ROOT.parent
SOURCE = ARTIFACTS.parent / 'tmp/reward-progress-worktree'
sys.path.insert(0, str(SOURCE))
from antmaze_experiments.progress_reward import maze_geometry
from antmaze_experiments.critic_diagnostics import route_label

NEW_SHA = 'eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45'
BASE_SHA = '2564b59faa0d319eece496b93eff0f19359efc37'
CONDITIONS = {'control': (.99, 1.), 'gamma999': (.999, 1.),
              'temp3': (.99, 3.), 'gamma999_temp3': (.999, 3.)}
NAMES = {'control': 'Control: gamma .99 / T1', 'gamma999': 'gamma .999 / T1',
         'temp3': 'gamma .99 / T3', 'gamma999_temp3': 'gamma .999 / T3'}
LINE_COLORS = {'control': '#4f5964', 'gamma999': '#df8823',
               'temp3': '#277db5', 'gamma999_temp3': '#29926b'}
ROUTE_COLORS = {'upper': '#277db5', 'lower': '#e58a27', 'left': '#277db5',
                'right': '#e58a27', 'uncommitted': '#8d959e'}
SHORT = {'upper': 'U', 'lower': 'D', 'left': 'L', 'right': 'R', 'uncommitted': 'none'}
TASKS = ('v1', 'v3', 'v4')
MODES = ('policy', 'native')
WARMUP = 8192
SCREEN_TOTAL = 258304
OUT = ROOT / 'report'


def read(path):
    return json.loads(path.read_text())


def first_gate(task, xy):
    if task in ('v1', 'v4'):
        return route_label(task, xy)[0]
    crossed = np.flatnonzero(np.abs(xy[:, 0]) > 8)
    return ('left' if xy[crossed[0], 0] < 0 else 'right') if len(crossed) else 'uncommitted'


def evaluate_saved(summary_path, cfg, condition, mode):
    summary = read(summary_path)
    raw = summary_path.with_name('rollouts.npz')
    with np.load(raw, allow_pickle=False) as z:
        assert str(z['mode']) == mode and int(z['env_steps']) == summary['step']
        points = [xy[:int(k) + 1].copy() for xy, k in zip(z['xy'], z['lengths'])]
        lengths, goals = z['lengths'].copy(), z['goals'].copy()
        starts, returns = z['initial_full_state'].copy(), z['returns'].copy()
    task, n = cfg['task'], len(points)
    assert n == summary['episodes'] == len(goals) == len(returns)
    assert np.isclose((goals > 0).mean(), summary['success_rate'])
    assert np.isclose(returns.mean(), summary['mean_return'])
    assert all(np.isfinite(p).all() and len(p) > 1 for p in points)
    if task == 'v1':
        assert not summary['fixed'] and not summary['identical_initial_full_state']
    else:
        assert summary['original_origin_fixed'] and summary['identical_initial_full_state']
        np.testing.assert_array_equal(starts, np.repeat(starts[:1], n, axis=0))
        assert np.all(starts[:, :2] == 0)
    _, goal_xy, _ = maze_geometry(task)
    distances = [np.linalg.norm(p[:, None] - goal_xy, axis=-1).min(-1) for p in points]
    np.testing.assert_allclose(returns, [100 * (d[0] - d[-1]) for d in distances],
                               atol=.002, rtol=2e-5)
    for p, goal in zip(points, goals):
        if goal > 0:
            assert np.linalg.norm(p[-1] - goal_xy[int(goal) - 1]) < .501
    labels = [first_gate(task, p) for p in points]
    entries = dict(Counter(labels))
    successes = dict(Counter(label for label, goal in zip(labels, goals) if goal > 0))
    sides = ('upper', 'lower') if task in ('v1', 'v4') else ('left', 'right')
    minority_entry = min(entries.get(side, 0) for side in sides)
    minority_success = min(successes.get(side, 0) for side in sides)
    gamma = cfg['native']['alg']['gamma']
    # Finite rollout reward sum only: no timeout value bootstrap. It is not an exact Q target.
    mc = [float(np.dot(100 * (d[:-1] - d[1:]), gamma ** np.arange(len(d) - 1))) for d in distances]
    row = dict(id=summary_path.parents[3].name, task=task, condition=condition, mode=mode,
               source_commit=cfg['source_commit'], gamma=gamma, temperature=cfg['temperature'],
               step=int(summary['step']), global_step=int(summary['step']) - WARMUP,
               episodes=n, success_rate=float((goals > 0).mean()), mean_return=float(returns.mean()),
               route_counts=entries, successful_route_counts=successes,
               successful_goal_routes=dict(Counter(f'{label}/G{goal}' for label, goal in zip(labels, goals) if goal > 0)),
               minority_entry_rate=minority_entry / n, minority_success_rate=minority_success / n,
               screen_both_routes_ge10pct=minority_success >= max(4, .1 * n),
               both_successful_routes_observed=minority_success > 0,
               mean_closest_goal_distance=float(np.mean([d.min() for d in distances])),
               mean_final_goal_distance=float(np.mean([d[-1] for d in distances])),
               mean_episode_length=float(lengths.mean()),
               mean_finite_horizon_discounted_return=float(np.mean(mc)),
               route_finite_horizon_discounted_returns={side: float(np.mean([v for v, label in zip(mc, labels) if label == side]))
                    for side in sides if side in labels},
               fixed_original_start=task != 'v1',
               raw_path=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    return dict(row=row, points=points, goals=goals, labels=labels)


def collect():
    runs = {}
    locations = ((ARTIFACTS / 'antmaze_dacer_positive_500k', BASE_SHA, True),
                 (ROOT, NEW_SHA, False))
    for campaign, source, is_control in locations:
        for host in sorted((campaign / 'results').glob('vast-heechan-*')):
            manifest = read(host / 'manifest.json')
            assert manifest['source_commit'] == source
            for job in manifest['jobs']:
                if job['task'] not in TASKS or job['dacer_target_entropy_per_dim'] != .7:
                    continue
                condition = 'control' if is_control else job['hypothesis']
                key = (job['task'], condition)
                assert key not in runs, f'Duplicate policy: {key}'
                folder = host / 'runs' / job['id']
                run = dict(id=job['id'], host=host.name, condition=condition, task=job['task'],
                           cfg=None, evaluations={mode: [] for mode in MODES})
                runs[key] = run
                if not (folder / 'config.json').exists():
                    continue
                cfg = read(folder / 'config.json'); run['cfg'] = cfg
                assert cfg['source_commit'] == source and cfg['dacer_interval_updates'] == 500
                assert cfg['dacer_target_entropy_per_dim'] == .7 and cfg['dacer_enabled']
                assert not cfg['noveld_enabled'] and cfg['seed'] == 0
                assert cfg['reward_specification']['formula'] == '100*(d(current)-d(next))'
                assert (cfg['native']['alg']['gamma'], cfg['temperature']) == CONDITIONS[condition]
                assert cfg['eval_starts'] == 'upstream'
                reset = 'natural' if cfg['task'] == 'v1' else 'fixed'
                for mode in MODES:
                    for p in sorted((folder / 'evaluations').glob(f'*/{mode}-{reset}/summary.json')):
                        if read(p)['step'] > SCREEN_TOTAL or not p.with_name('rollouts.npz').exists():
                            continue
                        run['evaluations'][mode].append(evaluate_saved(p, cfg, condition, mode))
    return runs


def plot_trajectories(runs, mode):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), layout='constrained')
    for i, task in enumerate(TASKS):
        walls, goals, bounds = maze_geometry(task)
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i, j]; run = runs.get((task, condition))
            for x0, y0, x1, y1 in walls:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      facecolor='#e2e6e9', edgecolor='#bcc3c9', lw=.4))
            saved = run['evaluations'][mode] if run else []
            subtitle = 'Not in registered grid' if run is None else 'No saved evaluation yet'
            if saved:
                evaluation = saved[-1]; row = evaluation['row']
                for index in np.argsort(evaluation['goals'] > 0):
                    p, goal = evaluation['points'][index], evaluation['goals'][index]
                    color = ROUTE_COLORS[evaluation['labels'][index]]
                    ax.plot(p[:, 0], p[:, 1], color=color, alpha=.4 if goal > 0 else .17,
                            lw=1 if goal > 0 else .6)
                    ax.scatter(*p[-1], s=6, color=color, alpha=.4)
                start = np.asarray([p[0] for p in evaluation['points']])
                ax.scatter(*start.T, s=6, c='black', alpha=.3)
                counts = ' '.join(f'{SHORT[k]}:{v}' for k, v in row['route_counts'].items())
                sc = ' '.join(f'{SHORT[k]}:{v}' for k, v in row['successful_route_counts'].items()) or 'none'
                subtitle = f"{row['step']/1000:.1f}k | entry {counts}\nsuccess {sum(row['successful_route_counts'].values())}/{row['episodes']}: {sc}"
            ax.set_title(f'{task} | {NAMES[condition]}\n{subtitle}', fontsize=9)
            ax.scatter(*goals.T, marker='*', s=90, c='#38a35f', edgecolor='white', lw=.5, zorder=5)
            ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect('equal')
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    policy = 'direct policy: random z + conditional sigma' if mode == 'policy' else 'mu-only: random z; no conditional sigma'
    fig.suptitle(f'OptiQ gamma / temperature screen | {policy} | seed0\n'
                 'Latest saved step <=258304. Controls may have earlier checkpoints; see panel steps.\n'
                 'No external DACER noise. v1 random start; v3/v4 identical original full state.', fontsize=12)
    fig.savefig(OUT / f'latest_trajectories_{mode}.png', dpi=150); plt.close(fig)


def plot_matched(runs, task):
    control = runs.get((task, 'control'))
    if not control:
        return None
    controls = {e['row']['step']: e for e in control['evaluations']['policy']}
    pairs = []
    for condition in CONDITIONS:
        if condition == 'control' or (task, condition) not in runs:
            continue
        experiments = {e['row']['step']: e for e in runs[task, condition]['evaluations']['policy']}
        common = sorted(controls.keys() & experiments.keys())
        if common:
            step = common[-1]
            pairs.append((condition, controls[step], experiments[step]))
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 2, figsize=(9, 4 * len(pairs)),
                             squeeze=False, layout='constrained')
    walls, goals, bounds = maze_geometry(task)
    for i, (condition, baseline, experiment) in enumerate(pairs):
        for col, (name, evaluation) in enumerate((('control', baseline), (condition, experiment))):
            ax = axes[i, col]; row = evaluation['row']
            for x0, y0, x1, y1 in walls:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      facecolor='#e2e6e9', edgecolor='#bcc3c9', lw=.4))
            for index in np.argsort(evaluation['goals'] > 0):
                p = evaluation['points'][index]; success = evaluation['goals'][index] > 0
                ax.plot(p[:, 0], p[:, 1], color=ROUTE_COLORS[evaluation['labels'][index]],
                        alpha=.45 if success else .2, lw=1 if success else .7)
            counts = ' '.join(f'{SHORT[k]}:{v}' for k, v in row['route_counts'].items())
            sc = ' '.join(f'{SHORT[k]}:{v}' for k, v in row['successful_route_counts'].items()) or 'none'
            ax.set_title(f"{NAMES[name]} | {row['step']/1000:.1f}k\nentry {counts} | successful routes {sc}", fontsize=9)
            ax.scatter(*goals.T, marker='*', s=90, c='#38a35f', edgecolor='white', lw=.5)
            ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect('equal')
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.suptitle(f'{task}: exact-step matched direct-policy trajectories | seed0\n'
                 'Each row compares equal interaction budgets. Failures included; no external DACER noise.', fontsize=11)
    filename = f'matched_trajectories_{task}.png'
    fig.savefig(OUT / filename, dpi=150); plt.close(fig)
    return filename


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    runs = collect()
    records = [e['row'] for run in runs.values() for mode in MODES for e in run['evaluations'][mode]]
    latest = [run['evaluations'][mode][-1]['row'] for run in runs.values() for mode in MODES if run['evaluations'][mode]]
    matched = []
    for (task, condition), run in runs.items():
        if condition == 'control' or (task, 'control') not in runs:
            continue
        for mode in MODES:
            base = {e['row']['step']: e['row'] for e in runs[task, 'control']['evaluations'][mode]}
            for e in run['evaluations'][mode]:
                if e['row']['step'] in base:
                    matched.append(dict(task=task, condition=condition, mode=mode, step=e['row']['step'],
                                        control=base[e['row']['step']], experiment=e['row']))
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), training_source=NEW_SHA,
                   control_source=BASE_SHA, controller_source='7278a0ed0f37e536ae81663fbd1e08a49ae561b8',
                   report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   latest=latest, history=records, exact_step_matched_comparisons=matched,
                   registered_new_policies=8, observed_new_configs=sum(r['cfg'] is not None for r in runs.values() if r['condition'] != 'control'),
                   validation='Raw NPZ episode count, starts, goal endpoint, success and reward telescope verified.',
                   limitations=['One seed per condition; do not combine different policies or checkpoints.',
                                'Corridor entry is not goal success; >=10% both-success screen is not proof of retention.',
                                'v1 random starts do not alone establish same-state multimodality.',
                                'Finite discounted rollout returns exclude timeout bootstraps; do not equate them to exact Q.',
                                'Use exact_step_matched_comparisons for budget-matched comparisons.'])
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    fields = ['task', 'condition', 'mode', 'step', 'episodes', 'gamma', 'temperature', 'success_rate',
              'minority_entry_rate', 'minority_success_rate', 'mean_closest_goal_distance',
              'mean_final_goal_distance', 'route_counts', 'successful_route_counts']
    with (OUT / 'history.csv').open('w') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], dict) else row[k] for k in fields})
    for mode in MODES:
        plot_trajectories(runs, mode)
        fig, axes = plt.subplots(4, 3, figsize=(13, 11), layout='constrained')
        for col, task in enumerate(TASKS):
            for condition in CONDITIONS:
                run = runs.get((task, condition)); saved = run['evaluations'][mode] if run else []
                for row, key in enumerate(('success_rate', 'minority_entry_rate', 'minority_success_rate', 'mean_closest_goal_distance')):
                    axes[row, col].plot([e['row']['step']/1000 for e in saved], [e['row'][key] for e in saved],
                                       color=LINE_COLORS[condition], marker='o', ms=3)
            for row, title in enumerate(('Goal success fraction', 'Minority entry / all episodes',
                                         'Minority successful route / all episodes', 'Closest goal distance, mean (m)')):
                ax = axes[row, col]; ax.set_title(f'{task} | {title}', fontsize=9)
                if row < 3: ax.set_ylim(-.02, 1.02 if row == 0 else .52)
                ax.set_xlabel('Total transitions (k)'); ax.grid(alpha=.2)
        axes[0, 0].legend(handles=[Line2D([0], [0], color=LINE_COLORS[c], label=NAMES[c]) for c in CONDITIONS], fontsize=7)
        fig.suptitle(f'Checkpoint retention and goal progress | {mode} | separate trained policies', fontsize=12)
        fig.savefig(OUT / f'learning_curves_{mode}.png', dpi=150); plt.close(fig)
    matched_figures = [f for task in TASKS if (f := plot_matched(runs, task))]
    lines = ['# Gamma/temperature 단기 실험', '',
             '현재 수집된 정책별 결과입니다. 기존 H/d +0.7 대조군을 포함하며, 표의 step이 다르면 최종 성능을 직접 비교하지 않습니다. '
             '정확히 같은 step의 비교만 results.json의 exact_step_matched_comparisons에 보관합니다.', '',
             '| 환경 | 조건 | 평가 | total step | 횟수 | 첫 통로 | 성공 통로 | 성공률 |',
             '|---|---|---|---:|---:|---|---|---:|']
    for r in sorted(latest, key=lambda x: (x['task'], x['condition'], x['mode'])):
        lines.append(f"| {r['task']} | {r['condition']} | {r['mode']} | {r['step']} | {r['episodes']} | {r['route_counts']} | {r['successful_route_counts']} | {r['success_rate']:.1%} |")
    lines += ['', *[f'![동일 step 비교]({f})' for f in matched_figures], '',
              'policy는 random-z와 conditional sigma를 모두 사용하는 직접 정책입니다. native는 random-z μ-only 보조 결과입니다. '
              '두 평가 모두 외부 DACER 행동잡음을 더하지 않습니다. v1은 원래 랜덤 시작이고 v3/v4는 원래 고정 full state입니다.', '',
              '통로 진입·목표 도달·후기 체크포인트에서의 유지 여부를 따로 판단합니다. 단일 seed의 일시적인 양쪽 방문으로 목표 달성을 선언하지 않습니다.', '',
              '![직접 정책](latest_trajectories_policy.png)', '![직접 정책 곡선](learning_curves_policy.png)',
              '![mu-only 보조](latest_trajectories_native.png)', '![mu-only 곡선](learning_curves_native.png)', '',
              f'학습 source {NEW_SHA}, 대조군 {BASE_SHA}. 별도 후처리이며 학습·원시 자료를 변경하지 않았습니다.']
    (OUT / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(OUT), evaluated_snapshots=len(records), matched=len(matched),
                          latest=[r for r in latest if r['mode'] == 'policy']), indent=2))


if __name__ == '__main__':
    main()
