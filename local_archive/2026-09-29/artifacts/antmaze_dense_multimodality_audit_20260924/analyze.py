"""Read-only audit of historical OptiQ dense-reward policies and saved rollouts.

Run from the OptiQ workspace with the Python 3.12 report environment.
No training, model mutation, resampling, or selection of successful episodes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from antmaze_experiments.dense_off_report import GOALS, geometry

LEGACY = REPO / 'artifacts/antmaze_dense_noveld_1m/runs'
OFF = ROOT / 'upstream_dense_off'
POS, NEG, FAIL = '#177eb2', '#e18c25', '#b2b7bf'
INPUTS = {}


def load(path):
    return json.loads(path.read_text())


def crossing(path, x, side=-1, last=False):
    old, new = path[:-1], path[1:]
    mask = ((old[:, 0] > x) & (new[:, 0] <= x) if side < 0 else
            (old[:, 0] < x) & (new[:, 0] >= x))
    ix = np.flatnonzero(mask)
    if not len(ix):
        return None
    i = ix[-1] if last else ix[0]
    t = (x - old[i, 0]) / (new[i, 0] - old[i, 0])
    return float(old[i, 1] + t * (new[i, 1] - old[i, 1]))


def route(task, path, goal, goals):
    if not goal:
        return 'failure'
    if task == 'v1':
        y = crossing(path, -4, last=True)
        return 'G1/' + ('upper' if y is not None and y > 2 else
                        'lower' if y is not None and y < -2 else 'unclassified')
    if task == 'v2':
        # The two historical implementations use opposite goal-ID ordering.
        gx, gy = goals[goal - 1]
        y = crossing(path, 4 if gx > 0 else -4, side=1 if gx > 0 else -1)
        return f'goal({gx:g},{gy:g})/' + ('central-corridor' if y is not None and abs(y) < 2
                                         else 'unclassified')
    if task == 'v4':
        entry, late = crossing(path, -4), crossing(path, -12, last=True)
        first = ('upper' if entry is not None and entry > 2 else
                 'lower' if entry is not None and entry < -2 else 'unclassified')
        final = ('upper-outer' if late is not None and late > 6 else
                 'lower-outer' if late is not None and late < -6 else
                 'middle' if late is not None and abs(late) < 2 else 'unclassified')
        return f'G{goal}/{first}-entry/{final}'
    y = crossing(path, -8 if goal == 1 else 8, side=-1 if goal == 1 else 1, last=True)
    return f'G{goal}/passage-y{round(y / 4) * 4:+d}' if y is not None else f'G{goal}/unclassified'


def read_rollout(path, task, legacy=False, fixed=False):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(REPO))] = hashlib.sha256(raw).hexdigest()
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    goal_ids = d['goal_ids' if legacy else 'goals']
    goals = (load(LEGACY / f'{task}-optiq-s0/config.json')['environment']['goals']
             if legacy else GOALS[task])
    goals = np.array(goals, dtype=float)
    routes = []
    for xy, n, g, ret in zip(d['xy'], d['lengths'], goal_ids, d['returns']):
        line = xy[:n + 1]
        assert np.isfinite(line).all() and np.isnan(xy[n + 1:]).all()
        dist = np.linalg.norm(line[:, None, :] - goals[None, :, :], axis=-1)
        assert bool(g) == bool(dist.min() <= .50002), path
        if g:
            assert dist[:, int(g) - 1].min() <= .50002
        assert np.isclose(-dist[1:].min(axis=-1).sum(), ret, rtol=2e-6, atol=.003), path
        routes.append(route(task, line, int(g), goals))
    if fixed:
        key = 'initial_simulator_state' if legacy else 'initial_full_state'
        assert np.array_equal(d[key], np.broadcast_to(d[key][0], d[key].shape)), path
    counts = Counter(routes)
    success = int((goal_ids > 0).sum())
    selected = {k: v for k, v in counts.items() if k != 'failure' and 'unclassified' not in k}
    p = np.array(list(selected.values()), dtype=float)
    if len(p):
        p /= p.sum()
    m = dict(episodes=len(goal_ids), successes=success, failures=len(goal_ids) - success,
             routes=dict(sorted(counts.items())), successful_routes=selected,
             dominant_successful_route_fraction=float(p.max()) if len(p) else None,
             effective_successful_routes=float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0,
             goal_coordinates=goals.tolist(),
             goals={str(int(g)): int((goal_ids == g).sum()) for g in np.unique(goal_ids)},
             fixed_full_state_verified=fixed, input=str(path.relative_to(REPO)))
    d['route_labels'] = routes
    return d, m


def decorate(ax, task):
    maze, rr, cc = geometry(task)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle(((j - cc) * 4 - 2, (i - rr) * 4 - 2), 4, 4,
                                       facecolor='#e8ebef', edgecolor='#c9cfd5', lw=.5, zorder=0))
    for g in GOALS[task]:
        ax.add_patch(Circle(g, .5, facecolor='#eee09b', zorder=5))
        ax.scatter(*g, marker='*', s=110, color='#a77e00', edgecolor='white', lw=.6, zorder=6)
    ax.set_xlim(-cc * 4 - 2, (len(maze[0]) - 1 - cc) * 4 + 2)
    ax.set_ylim(-rr * 4 - 2, (len(maze) - 1 - rr) * 4 + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.tick_params(labelsize=8)


def draw(ax, task, d, m):
    decorate(ax, task)
    # Failures first; all episodes retained, no cherry-picked trajectories.
    order = sorted(range(m['episodes']), key=lambda i: d['route_labels'][i] != 'failure')
    for i in order:
        line = d['xy'][i, :d['lengths'][i] + 1]
        label = d['route_labels'][i]
        color = FAIL if label == 'failure' else NEG if label == 'G1/lower' else POS
        ax.plot(line[:, 0], line[:, 1], color=color, lw=.65,
                alpha=.38 if label == 'failure' else .28, zorder=2)
        if label == 'failure':
            ax.scatter(*line[-1], color='#b44b4b', s=7, alpha=.5, zorder=3)
    ax.scatter(d['xy'][:, 0, 0], d['xy'][:, 0, 1], marker='^', color='#303743', s=9, alpha=.5, zorder=7)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False})
    legacy, off, arrays = {}, {}, {}
    for task in ['v1', 'v2', 'v3', 'v4']:
        legacy[task] = {}
        for mode in ['policy', 'native', 'zero_z']:
            for reset in ['natural', 'fixed']:
                label = f'{mode}-{reset}'
                path = LEGACY / f'{task}-optiq-s0/rollouts/1000000-{label}.npz'
                d, m = read_rollout(path, task, legacy=True, fixed=reset == 'fixed')
                legacy[task][label] = m
                arrays[('legacy', task, label)] = d
                if task == 'v1':
                    original = load(path.with_suffix('.json'))
                    assert Counter(original['routes']) == Counter(d['route_labels'])
        run = OFF / f'{task}-optiq-s0'
        latest = sorted((run / 'evaluations').iterdir())[-1]
        off[task] = dict(step=int(latest.name), completed=(run / 'result.json').exists(), modes={})
        for folder in sorted(latest.iterdir()):
            path = folder / 'rollouts.npz'
            if not path.exists():
                continue
            d, m = read_rollout(path, task, fixed=folder.name.endswith('-fixed'))
            off[task]['modes'][folder.name] = m
            arrays[('off', task, folder.name)] = d

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 10.6))
    fig.subplots_adjust(top=.82, bottom=.15, left=.055, right=.985, hspace=.68, wspace=.23)
    cols = [('policy-natural', 'Direct policy | random starts'),
            ('policy-fixed', 'Direct policy | same full state'),
            ('native-fixed', 'Mu-only, random z | same full state')]
    for row, cohort in enumerate(['legacy', 'off']):
        for col, (label, title) in enumerate(cols):
            m = legacy['v1'][label] if cohort == 'legacy' else off['v1']['modes'][label]
            draw(axes[row, col], 'v1', arrays[(cohort, 'v1', label)], m)
            counts = m['routes']
            axes[row, col].set_title(title + '\n' +
                f'+y route {counts.get("G1/upper", 0)} | -y route {counts.get("G1/lower", 0)} | fail {m["failures"]}',
                fontsize=10, pad=9)
    fig.suptitle('OptiQ: do multiple successful routes survive training?\nAntMaze v1 | 100 rollouts per panel | one training seed',
                 fontsize=17, y=.98)
    fig.text(.5, .883, 'Earlier custom port | dense + NovelD 0.01 | 1.000M transitions / 995,000 updates',
             ha='center', fontsize=12, weight='bold')
    fig.text(.5, .493, 'Official environment | dense + NovelD OFF | 3.008M transitions / 93,752 updates',
             ha='center', fontsize=12, weight='bold')
    handles = [Line2D([0], [0], color=POS, lw=2, label='Successful +y route'),
               Line2D([0], [0], color=NEG, lw=2, label='Successful -y route'),
               Line2D([0], [0], color=FAIL, lw=2, label='Failure (red endpoint)'),
               Line2D([0], [0], marker='^', color='#303743', lw=0, label='Initial position')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .067), ncol=4, frameon=False)
    fig.text(.5, .035, 'Direct policy = fresh random z + conditional sigma. Mu-only removes conditional sigma; z remains random.\n'
             'Rows differ in environment, batch, update schedule and training budget. Fixed starts are identical within a row, not between rows.\n'
             'All failures included. No external DACER noise in evaluation. Cartesian coordinates: +y is upward.',
             ha='center', va='center', fontsize=9, color='#4c5661')
    fig.savefig(ROOT / 'v1_policy_comparison.png', dpi=170)
    plt.close(fig)

    legacy_hist = load(LEGACY / 'v1-optiq-s0/history-policy-natural.json')
    old_series = [dict(step=r['step'], episodes=r['episodes'], routes=dict(Counter(r['routes']))) for r in legacy_hist]
    new_series = []
    for stepdir in sorted((OFF / 'v1-optiq-s0/evaluations').iterdir()):
        _, m = read_rollout(stepdir / 'policy-natural/rollouts.npz', 'v1')
        new_series.append(dict(step=int(stepdir.name), episodes=m['episodes'], routes=m['routes']))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), constrained_layout=True)
    for ax, rows, title in zip(axes, [old_series, new_series],
                              ['Custom port + NovelD 0.01', 'Official environment + NovelD OFF']):
        x = [r['step'] / 1e6 for r in rows]
        for key, color, name in [('G1/upper', POS, '+y successful route'), ('G1/lower', NEG, '-y successful route'),
                                 ('failure', FAIL, 'Failure')]:
            ax.plot(x, [r['routes'].get(key, 0) / r['episodes'] for r in rows], color=color,
                    label=name, lw=1.7, marker='o', ms=3)
        ax.set_ylim(-.04, 1.04)
        ax.set_title(title)
        ax.set_xlabel('Environment transitions (millions)')
        ax.set_ylabel('Fraction of evaluated episodes')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9, loc='center left')
    fig.suptitle('v1 route persistence across saved policy evaluations | random starts', fontsize=14)
    fig.supxlabel('Each point evaluates the policy at that step: left 10 episodes; right 20, with 100 at final. One training seed.', fontsize=9)
    fig.savefig(ROOT / 'v1_route_persistence.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.8), constrained_layout=True)
    for ax, task in zip(axes, ['v1', 'v2', 'v3', 'v4']):
        r = off[task]
        m = r['modes']['policy-natural']
        draw(ax, task, arrays[('off', task, 'policy-natural')], m)
        ax.set_title(f'{task.upper()} | {r["step"] / 1e6:.3f}M transitions\n'
                     f'success {m["successes"]}/{m["episodes"]} | ' + ('completed' if r['completed'] else 'interrupted'), fontsize=11)
    fig.suptitle('Historical official-environment OptiQ | dense reward + NovelD OFF\nLatest SAVED direct-policy evaluation; random starts, seed 0', fontsize=15)
    fig.supxlabel('All episodes, including failures. v2-v4 were stopped by user before completion; only their saved trajectories are available.\n'
                  'Blue/orange: successful routes. Gray: failures; red: failed endpoints; triangles: starts; stars: goals. No external exploration noise.', fontsize=10)
    fig.savefig(ROOT / 'upstream_dense_off_latest.png', dpi=170)
    plt.close(fig)

    late = [r for r in old_series if r['step'] >= 500000]
    retained = sum(r['routes'].get('G1/upper', 0) > 0 and r['routes'].get('G1/lower', 0) > 0 for r in late)
    results = dict(analysis_date='2026-09-24 KST', single_training_seed=0,
                   sources=dict(legacy='19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5',
                                official_dense_off='26336810f7ea4ca61210ea70c6aeeae9f7acaed0'),
                   validation='Raw finite trajectories, true goal hits, summed dense returns, and fixed initial states rechecked',
                   route_rule='Geometric corridor crossing; successful episodes only; v1 last leftward crossing of x=-4, y>2 or y<-2',
                   legacy=legacy, official_dense_off=off,
                   v1_history=dict(legacy=old_series, official_dense_off=new_series,
                                   legacy_late_both_routes=dict(evaluations=retained, total=len(late))),
                   input_sha256=INPUTS,
                   limitations=['Different environments, update ratios, batch sizes, and budgets; not a controlled NovelD ablation',
                                'One seed per policy; evaluated route diversity is not proof of multimodal action densities',
                                'No new rollouts in this audit; raw saved rollouts and prior fresh-seed rerollout validation inspected',
                                'Official dense-OFF v2-v4 interrupted and lack latest full policy checkpoints'])
    (ROOT / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')

    lines = ['# 기존 OptiQ dense reward 결과의 경로 다양성 검증', '',
             '2026-09-24. 저장된 원시 rollout을 다시 집계했으며 신규 학습·평가 rollout은 실행하지 않았다. 각 실험 단일 seed 0.', '',
             '**결론: v1의 이전 자체 포트 + NovelD 0.01 정책에는 두 성공 경로가 유지된다. 공식 환경 + NovelD OFF v1은 성공률이 높지만 거의 한 경로로 집중된다. 모든 maze에서 다양성을 유지한다고 말할 근거는 없다.**', '',
             '## v1 최종 정책: 성공 경로와 실패 모두 집계', '',
             '|실험|평가|+y 성공 경로|−y 성공 경로|실패|', '|---|---|---:|---:|---:|']
    for name, modes in [('이전 포트 dense + NovelD 0.01, 1M', legacy['v1']),
                        ('공식 환경 dense + NovelD OFF, 3.008M', off['v1']['modes'])]:
        for label in ['policy-natural', 'policy-fixed', 'native-fixed', 'zero_z-fixed']:
            m = modes[label]
            lines.append(f'|{name}|{label}|{m["routes"].get("G1/upper",0)}|{m["routes"].get("G1/lower",0)}|{m["failures"]}|')
    lines += ['', 'policy는 fresh random z + conditional sigma, native는 random z의 mu-only, zero_z는 z=0 mu-only다. 평가에 외부 DACER 잡음이나 NovelD 보상을 더하지 않았다.',
              'natural은 v1 랜덤 시작. fixed는 각 실험 내부에서 qpos/qvel 및 저장된 초기 시뮬레이터 상태를 동일하게 복원했다. 두 실험의 fixed 시작은 서로 다르다: 이전 포트 (0,0), 공식 환경 약 (1.9873,−1.4523). 그림은 +y가 위로 향하도록 통일했다.',
              f'이전 포트 v1은 500k부터 1M까지 {retained}/{len(late)}회의 평가에서 두 성공 경로가 모두 관찰됐다(각 10회). 한 번만 우연히 나타난 결과가 아니다.',
              '이전 포트 v1의 고정 시작 mu-only에서도 75/21로 나뉘고 z=0에서는 100/0이다. 시작점 차이나 conditional sigma 잡음만으로 설명되는 현상이 아니며, latent를 사용하는 정책의 경로 다양성에 대한 근거다. 다만 동일 state의 action density 자체가 다봉이라는 직접 증명은 아니다.',
              '별도 과거 모델 복원 감사에서도 새 평가 난수로 v1 policy 100회를 실행해 +y 71, −y 28, 실패1을 얻었다. 현재 감사에서는 그 저장 결과와 model-unchanged 검증서를 확인했다.', '',
              '## 공식 환경 dense + NovelD OFF의 미로별 마지막 저장 평가', '',
              '|미로|저장 평가 step|상태|성공/평가수|성공 경로 집계|', '|---|---:|---|---:|---|']
    for task, r in off.items():
        m = r['modes']['policy-natural']
        desc = ', '.join(f'{k}: {v}' for k, v in m['successful_routes'].items()) or '성공 없음'
        lines.append(f'|{task}|{r["step"]:,}|{"완료" if r["completed"] else "사용자 중단 전 마지막 평가"}|{m["successes"]}/{m["episodes"]}|{desc}|')
    lines += ['', 'v2-v4는 완료 결과가 아니고 마지막 평가는 20회뿐이다. v4는 성공 경로가 없어 모드 붕괴를 판정할 수 없다. 한 목표에 도달했다는 사실과 한 경로를 사용했다는 사실은 구분하고 실제 통로 통과 좌표로 분류했다.', '',
              '## 이전 포트 dense + NovelD 0.01: 1M 최종 고정 상태 direct-policy', '',
              '|미로|성공/100|성공 경로 집계|', '|---|---:|---|']
    for task in ['v1', 'v2', 'v3', 'v4']:
        m = legacy[task]['policy-fixed']
        lines.append(f'|{task}|{m["successes"]}|'+', '.join(f'{k}: {v}' for k,v in m['successful_routes'].items())+'|')
    lines += ['', 'v2의 goal ID는 과거 포트와 공식 환경에서 반대로 부여되므로 실제 좌표로 비교했다. v2-v4 이전 포트는 natural과 fixed의 시작 상태 분포가 같으므로 200개 독립 시작 상태처럼 합치지 않는다.', '',
              '## 모델과 실험 조건', '',
              '|항목|이전 포트 dense + 0.01|공식 환경 dense OFF v1|현재 실행 중 dense OFF|',
              '|---|---|---|---|', '|actor/critic|256×2|256×2|256×3|',
              '|temperature|0.25|0.25|0.01|', '|LR actor / critic|3e-4 / 3e-4|3e-4 / 3e-4|3e-4 / 5e-4|',
              '|환경 수 / batch|1 / 256|256 / 4096|256 / 4096|',
              '|업데이트 비율|1 / transition|8 / 256 transitions|8 / 256 transitions|',
              '|최종 transitions / learner updates|1M / 995,000|3,008,256 / 93,752|v1 예정 3,008,256 / 93,752|',
              '|NovelD|0.01|OFF, RND updates 0|OFF|',
              '','모든 세대의 OptiQ는 여기서 random z, mean-init1, log sigma[-5,-1], DACER behavior exploration을 사용한다. 본 비교의 차이는 NovelD 하나가 아니므로 ON/OFF 인과효과나 현재 설정의 최종 성능을 단정하지 않는다.',
              '이전 포트 v1-v4 full state를 CPU에서 읽어 원본 SHA256, actor/critic 실제 256×2 kernel, optimizer/model 유한성을 검증했다. 공식 환경 v1의 307MB final checkpoint도 서버 CPU에서 SHA256·93,752 updates·RND OFF·실제 256×2·유한한 정책/optimizer를 확인했다. 공식 환경 v2-v4는 final-only 저장 설정으로 중단 당시 full checkpoint가 없으며 원시 평가 궤적을 사용했다.',
              '저장 궤적 전부에서 유한한 좌표, 실제 goal 반경 도달, dense return 거리합, 고정 상태 동일성을 다시 검사했다. 입력 SHA256은 results.json에 기록했다.',
              '', '## 출처와 파일', '',
              '- 이전 포트 학습 source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`',
              '- 공식 환경 dense OFF 학습 source: `26336810f7ea4ca61210ea70c6aeeae9f7acaed0`',
              '- 현재 실행 중 dense OFF source: `a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3` (이번 과거 분석의 결과에 섞지 않음)',
              '- 이번 재집계: `analyze.py`, `results.json`, `legacy-model-verification.json`, `upstream-model-verification.json`',
              '- 이전 새 난수 모델 복원 확인: `../antmaze_route_audit/rerollout-verification.json`',
              '- 그림: `v1_policy_comparison.png`, `v1_route_persistence.png`, `upstream_dense_off_latest.png`',
              '', '현재 16개 실험은 이 분석을 위해 설정을 바꾸거나 중단하지 않았다.']
    (ROOT / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(legacy_v1=legacy['v1'], official_dense_off=off,
                          late_both_routes=[retained,len(late)], outputs=str(ROOT)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
