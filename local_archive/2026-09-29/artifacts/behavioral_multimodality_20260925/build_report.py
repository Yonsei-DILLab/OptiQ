"""Read-only behavioral diversity audit of saved AntMaze final evaluations.

This does not restore a checkpoint, perform new rollouts, or change training.
The primary comparison uses 100 direct-policy rollouts from one identical full
initial state per maze and method. Historical random-start evaluations are
retained as separate performance context.
"""
import hashlib
import json
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SNAPSHOT = REPO / 'artifacts/antmaze_anneal_baselines/reports/20260924T050707Z'
sys.path.insert(0, str(REPO / 'artifacts/antmaze_dense_multimodality_audit_20260924'))
import analyze as route_audit

METHODS = [('SAC', 'Gaussian SAC'), ('MFPO', 'MFPO flow'),
           ('DIPO', 'DIPO diffusion'), ('T=1', 'OptiQ/iBOLT')]
TASKS = ('v1', 'v2', 'v3', 'v4')
COLORS = {'SAC': '#617181', 'MFPO': '#a2753d', 'DIPO': '#14847b', 'T=1': '#2175bb'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bins_05m(xy):
    points = xy[np.isfinite(xy).all(axis=-1)]
    return np.unique(np.floor(points / 0.5).astype(np.int32), axis=0)


def rollout_path(row, mode):
    candidates = list(SNAPSHOT.glob(
        f"{row['host']}/*/runs/{row['run_id']}/evaluations/{row['steps']:010d}/{mode}/rollouts.npz"))
    matches = [path for path in candidates if json.loads(
        (path.parents[3] / 'config.json').read_text())['source_commit'] == row['source']]
    assert len(matches) == 1, (row['run_id'], mode, matches)
    return matches[0]


def load_rollout(path, task, fixed):
    data, metric = route_audit.read_rollout(path, task, fixed=fixed)
    assert metric['episodes'] == 100
    metric['mean_return'] = float(data['returns'].mean())
    metric['return_sd'] = float(data['returns'].std(ddof=1))
    metric['visited_half_meter_bins'] = int(len(bins_05m(data['xy'])))
    metric['input_sha256'] = sha256(path)
    assert metric['input_sha256'] == route_audit.INPUTS[str(path.relative_to(REPO))]
    return data, metric


def main():
    ROOT.mkdir(exist_ok=True)
    source = json.loads((SNAPSHOT / 'analysis.json').read_text())
    assert source['snapshot']['verified'] and source['raw_validation_passed']
    rows, arrays = {}, {}
    for task in TASKS:
        starts = []
        for key, label in METHODS:
            row = source['runs'][key][task]
            assert row['method'] == ('optiq' if key == 'T=1' else key.lower())
            assert row['checkpoint_verification']['readback_verified']
            record = dict(method=label, maze=task, run_id=row['run_id'], host=row['host'],
                          source_commit=row['source'], steps=row['steps'],
                          learner_updates=row['updates'], modes={})
            for mode in ('policy-fixed', 'policy-natural'):
                path = rollout_path(row, mode)
                assert source['input_sha256'][str(path.relative_to(REPO))] == sha256(path)
                data, metric = load_rollout(path, task, fixed=mode.endswith('fixed'))
                assert metric['successes'] == row['modes'][mode]['successes']
                record['modes'][mode] = metric
                arrays[task, key, mode] = data
                if mode == 'policy-fixed':
                    states = data['initial_full_state']
                    assert np.array_equal(states, np.broadcast_to(states[0], states.shape))
                    starts.append(states[0])
            rows[task, key] = record
        assert all(np.array_equal(starts[0], state) for state in starts)
        rows[task, 'common_start_xy'] = starts[0][:2].tolist()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for i, task in enumerate(TASKS):
        for j, (key, label) in enumerate(METHODS):
            ax = axes[i, j]
            data = arrays[task, key, 'policy-fixed']
            metric = rows[task, key]['modes']['policy-fixed']
            route_audit.draw(ax, task, data, metric)
            ax.set_title(f"{task.upper()} · {label}\nsuccess {metric['successes']}/100 · "
                         f"successful routes {len(metric['successful_routes'])}")
    fig.suptitle('Saved AntMaze policies: 100 stochastic rollouts from one identical full state',
                 fontsize=16)
    fig.supxlabel('All successful and failed trajectories shown. Blue/orange: successful routes; gray: failures. '
                  'Direct policy sampling, training seed 0; dense reward, NovelD OFF. '
                  'Start (x,y) ≈ (1.99,-1.45) for this archived diagnostic.', fontsize=9)
    fig.savefig(ROOT / 'same_state_trajectories.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    for i, task in enumerate(('v1', 'v3')):
        for j, (key, label) in enumerate(METHODS):
            ax = axes[i, j]
            data = arrays[task, key, 'policy-fixed']
            points = data['xy'][np.isfinite(data['xy']).all(axis=-1)]
            maze, rr, cc = route_audit.geometry(task)
            xedges = np.arange(-cc*4-2, (len(maze[0])-1-cc)*4+2.51, .5)
            yedges = np.arange(-rr*4-2, (len(maze)-1-rr)*4+2.51, .5)
            counts, _, _ = np.histogram2d(points[:,0], points[:,1], bins=(xedges, yedges))
            capped = np.minimum(counts.T, 100)
            heat = ax.pcolormesh(xedges, yedges, np.ma.masked_where(capped == 0, capped),
                                 norm=LogNorm(vmin=1, vmax=100), cmap='magma', zorder=1)
            route_audit.decorate(ax, task)
            n = rows[task, key]['modes']['policy-fixed']['visited_half_meter_bins']
            ax.set_title(f'{task.upper()} · {label}\nvisited 0.5 m bins: {n}')
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=.65,
                 label='Visits per 0.5 m cell (clipped at 100)')
    fig.suptitle('Saved policy evaluation visitation; density clipped at 100 visits', fontsize=16)
    fig.supxlabel('DDiffPG-style density display and binary cell coverage, computed from 100 same-state '
                  'policy rollouts. These are evaluation visits, not training exploration.', fontsize=9)
    fig.savefig(ROOT / 'evaluation_coverage.png', dpi=160)
    plt.close(fig)

    output = dict(snapshot=str(SNAPSHOT.relative_to(REPO)), training_seed=0,
                  methods={key: label for key, label in METHODS},
                  reset_note='policy-fixed is a shared sampled full state near (1.99,-1.45); '
                             'v2-v4 upstream training reset is the origin, so this is a diagnostic.',
                  sampling='direct stochastic policy; OptiQ random latent plus conditional sigma; '
                           'no external DACER noise',
                  density='0.5 m visited cells and per-cell counts capped at 100; evaluation rollouts only',
                  route_rule='saved geometry-based successful corridor/goal labels from analyze.py',
                  rows={task: {key: rows[task, key] for key, _ in METHODS} |
                        {'common_start_xy': rows[task, 'common_start_xy']} for task in TASKS})
    (ROOT / 'results.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'maze', 'method', 'evaluation', 'steps', 'episodes', 'successes',
            'success_rate', 'discovered_successful_routes', 'route_counts',
            'visited_half_meter_bins', 'mean_return', 'return_sd',
            'source_commit', 'input_sha256'])
        writer.writeheader()
        for task in TASKS:
            for key, label in METHODS:
                record = rows[task, key]
                for mode in ('policy-fixed', 'policy-natural'):
                    m = record['modes'][mode]
                    writer.writerow(dict(maze=task, method=label, evaluation=mode,
                        steps=record['steps'], episodes=m['episodes'],
                        successes=m['successes'],
                        success_rate=m['successes'] / m['episodes'],
                        discovered_successful_routes=len(m['successful_routes']),
                        route_counts=json.dumps(m['routes'], ensure_ascii=False),
                        visited_half_meter_bins=m['visited_half_meter_bins'],
                        mean_return=m['mean_return'], return_sd=m['return_sd'],
                        source_commit=record['source_commit'], input_sha256=m['input_sha256']))

    lines = ['# 저장된 AntMaze 결과의 동일 초기 상태 행동 다양성', '',
             '최종 체크포인트마다 정책을 직접 샘플링한 저장 rollout 100회를 재집계했다. '
             '네 방법의 각 미로 내 100회는 동일한 qpos/qvel에서 시작한다. 학습 seed는 각 1개다.', '',
             '## 동일 상태 진단', '',
             '|미로|방법|성공/100|성공 행동 종류|성공 경로|방문 0.5m 격자|평균 return|',
             '|---|---|---:|---:|---|---:|---:|']
    for task in TASKS:
        for key, label in METHODS:
            m = rows[task, key]['modes']['policy-fixed']
            lines.append(f"|{task}|{label}|{m['successes']}|{len(m['successful_routes'])}|"
                         f"{json.dumps(m['routes'], ensure_ascii=False)}|"
                         f"{m['visited_half_meter_bins']}|{m['mean_return']:.1f}|")
    lines += ['', '성공 행동 종류는 목표 ID와 미로 통로 교차를 결합한 사전 정의 경로 범주 수다. '
              '성공이 없는 경우 0으로 기록한다. 방문 격자는 학습 exploration이 아닌 저장 평가 궤적에서 센 것이다.',
              '', '## 랜덤 시작 저장 평가', '',
              '|미로|방법|성공/100|평균 return|', '|---|---|---:|---:|']
    for task in TASKS:
        for key, label in METHODS:
            m = rows[task, key]['modes']['policy-natural']
            lines.append(f"|{task}|{label}|{m['successes']}|{m['mean_return']:.1f}|")
    lines += ['', '이 랜덤 시작 평가는 당시의 역사적 평가 override다. v2-v4의 원래 '
              '학습 reset은 고정 원점이므로 현재 matched-start primary 성능으로 인용하지 않는다.',
              '', '## 해석', '',
              '이 네 방법의 동일 상태 최종 정책에서 관찰된 성공 경로는 방법·미로별 최대 1개다. '
              '따라서 이 비교군만으로 iBOLT의 성공 행동 다봉성을 주장할 수 없다. '
              '별도 과거 OptiQ v1 dense+NovelD 0.01 저장 정책의 동일 상태 100회에서는 '
              '위/아래 성공 경로가 71/29로 확인되었다. 같은 모델의 mu-only '
              '동일 상태 평가도 75/21 성공, 4회 실패여서 conditional sigma 잡음만의 '
              '효과로 설명되지 않는다. 다만 자체 환경 포트·학습 설정이 달라 '
              '이 표의 비교군과 합산하거나 인과 비교하지 않는다.',
              '', 'GMM40의 target fitting 결과와 이 AntMaze 정책은 동일한 target 또는 '
              '체크포인트 쌍이 아니다. 따라서 target fitting → 행동 → RL 성능의 '
              '세 단계를 하나의 인과 실험으로 연결했다고 주장할 수 없다.',
              '', 'Gaussian SAC는 Gaussian 정책으로 Boltzmann 대상에 대한 reverse-KL형 '
              '목적을 사용한다. 이를 별도의 동일 actor reverse-KL ablation으로 세지 않는다. '
              'MFPO는 flow matching 방법이며 reverse-KL 대조군으로 표기하지 않는다. '
              '따라서 요청한 네 범주 중 독립된 reverse-KL actor 대조군의 AntMaze '
              '체크포인트/rollout은 이 저장 비교군에 없다.',
              '', '그림은 DDiffPG 선행연구의 궤적 오버레이·목표/경로 구분·방문 밀도(100회 상한)·'
              '이진 방문격자 방식에 맞췄다. DTW 계층 clustering은 성공 경로가 여러 개 '
              '나왔을 때 보조 검증으로 적용할 수 있으며, 현재 표에서 1개 경로를 '
              '인위적으로 더 나누지 않았다.',
              '', '선행연구: https://supersglzc.github.io/projects/ddiffpg/ '
              '(trajectory modes, density maps, binary state coverage, DTW 계층 군집).',
              '', '## 검증', '',
              '입력은 수집 당시 SHA256 및 현재 SHA256와 일치한다. 원시 좌표 유한성, '
              'padding, 목표 반경, dense return의 거리합, 체크포인트 readback sidecar, '
              '동일 전체 초기 상태를 재확인했다. 모델·원본 자료·진행 중 작업은 변경하지 않았다. '
              '정확한 run/source SHA와 입력 해시는 results.json에 기록했다.']
    (ROOT / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({task: {key: rows[task,key]['modes']['policy-fixed']['routes']
                             for key,_ in METHODS} for task in TASKS}, indent=2))


if __name__ == '__main__':
    main()
