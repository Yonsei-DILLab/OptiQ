"""Summarize archived historical AntMaze reset and action-sampling audits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[1] / 'tmp' / 'reward-progress-worktree'
sys.path.insert(0, str(WORKTREE))
from antmaze_experiments.critic_diagnostics import route_label
from antmaze_experiments.progress_reward import maze_geometry


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def original(task, step):
    host = 'vast-heechan-180' if task == 'v3' else 'vast-heechan-199'
    return ROOT.parent / 'antmaze_critic_dynamics' / 'results' / host / 'runs' / (
        task + '-optiq-control-500k-s0') / 'evaluations' / f'{step:010d}' / 'policy-fixed' / 'rollouts.npz'


fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
data = {}
for row, task in enumerate(('v3', 'v4')):
    early = 300032 if task == 'v3' else 200192
    audit = json.loads((ROOT / 'data' / task / 'evaluation-audit.json').read_text())
    first = json.loads((ROOT / 'data' / task / 'first-actions-final.json').read_text())
    assert first['initial_observations_unique'] == 1
    assert first['first_actions_unique'] == 100
    assert len(first['first_actions']) == 100
    final_file = ROOT / 'data' / task / 'rollouts-200.npz'
    assert sha(final_file) == audit['rollout_sha256']
    walls, goals, bounds = maze_geometry(task)
    chosen, opposite = ('right', 'left') if task == 'v3' else ('upper', 'lower')
    stages = [('Early, two branches', original(task, early)),
              ('Final, 200 direct-policy draws', final_file)]
    stage_records = []
    for col, (title, file) in enumerate(stages):
        ax = axes[row, col]
        for xmin, ymin, xmax, ymax in walls:
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   facecolor='#d9dfe6', edgecolor='#bcc5ce', lw=.4, zorder=0))
        with np.load(file) as rollout:
            starts = rollout['initial_full_state']
            assert len(np.unique(starts, axis=0)) == 1
            labels = []
            for xy, length in zip(rollout['xy'], rollout['lengths']):
                path = xy[:int(length) + 1]
                assert np.isfinite(path).all()
                label = route_label(task, path)[0]
                labels.append(label)
                color = '#d87632' if label == chosen else '#258db2' if label == opposite else '#666b70'
                ax.plot(path[:, 0], path[:, 1], color=color,
                        alpha=.27 if col == 0 else .065, lw=.65, zorder=1)
            counts = dict(Counter(labels))
            if col == 1:
                x, y = rollout['xy'][:, :, 0], rollout['xy'][:, :, 1]
                opposite_excursions = int(np.any(x < -2, axis=1).sum()) if task == 'v3' else int(np.any(y < -2, axis=1).sum())
                assert opposite_excursions == 0
            else:
                opposite_excursions = None
        for xg, yg in goals:
            ax.scatter(xg, yg, marker='*', s=100, c='#151719', edgecolors='white', linewidths=.6, zorder=4)
        ax.scatter(0, 0, marker='o', s=50, c='#d12845', edgecolors='white', linewidths=.6, zorder=5)
        ax.set_xlim(bounds[0] - 1, bounds[2] + 1)
        ax.set_ylim(bounds[1] - 1, bounds[3] + 1)
        ax.set_aspect('equal')
        ax.grid(alpha=.12)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_title(f'{task.upper()} | {title}\n{opposite}: {counts.get(opposite, 0)} · {chosen}: {counts.get(chosen, 0)} · other: {counts.get("uncommitted", 0)}', fontsize=11)
        stage_records.append(dict(checkpoint_step=early if col == 0 else audit['checkpoint_step'],
                                  episodes=sum(counts.values()), routes=counts, file_sha256=sha(file),
                                  opposite_shallow_excursions=opposite_excursions))
    with np.load(ROOT / 'data' / task / 'teacher-508416.npz') as proposal:
        mask = proposal['reference_landmarks'].astype(str) == '0'
        assert len(np.unique(proposal['observations'][mask], axis=0)) == 1
        means = proposal['mu'][mask].reshape(-1, 8)
        latent_mean_action_spread = float(np.linalg.norm(means.std(axis=0)))
    data[task] = dict(training_source=audit['training_source'], checkpoint_sha256=audit['checkpoint_sha256'],
                      audit_source='82a42ea', fixed_full_state_count=audit['unique_full_starts'],
                      direct_policy_first_actions_unique=first['first_actions_unique'],
                      first_action_std=first['first_action_std'],
                      first_action_std_norm=float(np.linalg.norm(first['first_action_std'])),
                      latent_mean_action_spread=latent_mean_action_spread,
                      reference_first_100=audit['reference_first_100'],
                      stages=stage_records)

fig.suptitle('AntMaze: fresh z and action noise still produce one route at the final checkpoint', fontsize=15)
fig.supxlabel('Same full start state; direct policy resamples latent and conditional action noise every action.\nOrange = final chosen side; blue = opposite side. Gate entry is not goal success.', fontsize=10)
fig.savefig(ROOT / 'route_eval_audit.png', dpi=180)
fig.savefig(ROOT / 'route_eval_audit.pdf')
plt.close(fig)
(ROOT / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

report = ['# AntMaze 경로 평가 방식 감사', '',
          'v3/v4 frozen control source 23603a7, 학습 보상100×Euclidean progress, T=1, DACER/NovelD OFF, seed0, 약500k post-warmup. 학습/정책 가중치는 수정하지 않았다.', '',
          '## 평가 시작 상태', '',
          '원본 AntMaze reset_model은 random_init=false일 때 init_qpos/init_qvel을 그대로 사용한다. 학습 v3/v4와 기존 평가 모두 XY·자세·속도가 완전히 동일하다. 두 episode의 native-reset 사전확인에서 full state 고유값은1개였고, 원본 평가 및 추가200회 자료에서도 고유값1개다. 따라서 평가만 자세/속도를 지나치게 고정했다는 앞선 추정은 틀렸다.', '',
          '## 실제 샘플링과 경로', '',
          'Frozen JaxLearner.act는 매 행동 reset_noise()로 새 key를 만들고, policy mode에서 random latent와 conditional sigma를 모두 샘플링한다. zero_z·mu-only는 별도 모드다. 최종 원본 평가의 동일한 초기 관측100개에서 실제 첫 행동100개는 모두 고유했다.', '',
          '|환경|이른 시점 직접정책 평가|추가 최종 직접정책 평가|반대 방향의 얕은 진입|첫 행동 std의 8D norm|',
          '|---|---|---|---:|---:|']
for task in ('v3', 'v4'):
    d = data[task]
    early, final = d['stages']
    side = 'left/right' if task == 'v3' else 'lower/upper'
    report.append(f"|{task} ({side})|{early['routes']} / {early['episodes']}|{final['routes']} / {final['episodes']}|{final['opposite_shallow_excursions']}|{d['first_action_std_norm']:.3f}|")
report += ['',
           'v3의 얕은 진입은 x<−2, v4는 y<−2 기준이다. 각 환경200회에서 이마저0이었다. 기존 통로 판정 문턱(x<−8 등)만 지나치게 엄격해서 다른 쪽 경로가 누락된 것이 아니다.', '',
           '같은 초기상태의 teacher probe에서도 latent별 conditional mean의 8차원 분산 합 제곱근은 v3 0.902, v4 1.405다. z가 완전히 무시된 것은 아니다. 다만 행동의 분산은 양쪽 경로를 끝까지 수행하는 확률과 다르며, 이 수치만으로 action 분포가 두 모드라고 말할 수 없다.', '',
           '추가 CPU 평가와 원본 GPU 평가의 초기 한 스텝 XY 차이는 v3 최대1.71e−5, v4 최대6.01e−6. 장기 동역학의 수치 차이로 첫100 경로라벨 일치율은 각각92/100,98/100이었다. 두 실행 모두 반대 방향 진입0이라는 결론은 같다.', '',
           '![초기/최종 경로 비교](route_eval_audit.png)', '',
           '## 해석 범위', '',
           '이 결과는 해당 체크포인트에서 초기상태를 고정한 direct 정책의 관측된 반대경로 비율이 낮다는 사실을 확인한다. 0/200은 정확한 확률0의 증명이 아니다. 버퍼에 남은 과거 전이와 과거 경로는 현재 actor π(a|s)의 두 경로 능력을 보장하지 않는다. 정책이 각 시점에 다양한 8D 행동을 내더라도, 복수 시점의 일관된 좌·우/상·하 제어가 경로를 결정한다. 어느 학습 동역학이 그 일관성을 없앴는지는 이 평가만으로 확정하지 않는다.', '',
           '추가평가 원자료·해시와 첫 행동 자료는 data/, 재현 스크립트는 direct-gmm-trg-antmaze commit82a42ea에 있다.']
(ROOT / 'REPORT_KO.md').write_text('\n'.join(report) + '\n')
print(json.dumps({task: dict(stages=d['stages'], first_action_std_norm=d['first_action_std_norm']) for task,d in data.items()}, indent=2))
