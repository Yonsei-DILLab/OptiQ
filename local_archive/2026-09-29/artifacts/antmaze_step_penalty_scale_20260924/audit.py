"""Read-only counterfactual reward audit of preserved AntMaze trajectories."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'antmaze_dacer_off_t1/reports/20260924T085941Z'
GAMMA = .99
COEFFICIENTS = [.03, .05, .1, .3, 1.]
GOALS = {'v1': [[-8, 0]], 'v3': [[-12, 12], [12, -12]],
         'v4': [[-16, 4], [-16, -4]]}

def stats(a):
    a = np.asarray(a)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(a.mean()),
                p10=float(np.quantile(a, .1)), p50=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p99=float(np.quantile(a, .99)))

def summarize(a):
    # Columns: literal progress, discount offset, potential increment, old dense,
    # xy displacement, episode success, true goal terminal.
    subsets = {'all_nonterminal': ~a[:, 6].astype(bool),
               'successful_episode_nonterminal': (a[:, 5] == 1) & (a[:, 6] == 0),
               'failed_episode_nonterminal': (a[:, 5] == 0) & (a[:, 6] == 0),
               'towards_goal_nonterminal': (a[:, 0] > .001) & (a[:, 6] == 0),
               'away_from_goal_nonterminal': (a[:, 0] < -.001) & (a[:, 6] == 0),
               'xy_motion_under_1mm_nonterminal': (a[:, 4] <= .001) & (a[:, 6] == 0),
               'goal_terminal': a[:, 6] == 1}
    result = {}
    for name, mask in subsets.items():
        b = a[mask]
        if not len(b):
            continue
        result[name] = dict(raw_progress=stats(b[:, 0]), discount_offset=stats(b[:, 1]),
                            potential_increment=stats(b[:, 2]), old_dense_reward=stats(b[:, 3]),
                            candidate_rewards={str(c): dict(
                                mean_potential_reward=float((b[:, 2]-c).mean()),
                                positive_potential_reward_fraction=float((b[:, 2] > c).mean()),
                                positive_literal_progress_reward_fraction=float((b[:, 0] > c).mean()))
                                for c in COEFFICIENTS})
    return result

data = {}; groups = {}; episodes = []; inputs = []; checks = []
for task, goals in GOALS.items():
    files = sorted(ROOT.glob(f'**/runs/{task}-optiq-dacer-off-T1-s0/evaluations/*/policy-natural/rollouts.npz'))
    arrays = []; summaries = []
    for path in files:
        z = np.load(path)
        assert str(z['mode']) == 'policy' and not bool(z['fixed'])
        config = json.loads((path.parents[3]/'config.json').read_text())
        assert config['source_commit'] == '484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
        assert config['temperature'] == 1 and not config['dacer_enabled'] and not config['noveld_enabled']
        file_arrays = []; max_old_error = 0.; max_identity_error = 0.
        for ix, (line, length, goal, original_return) in enumerate(zip(z['xy'], z['lengths'], z['goals'], z['returns'])):
            n = int(length); line = line[:n+1].astype(np.float64)
            assert np.isfinite(line).all()
            distances = np.linalg.norm(line[:, None, :]-np.asarray(goals)[None, :, :], axis=-1)
            d = distances.min(1)
            terminal = np.zeros(n, bool); terminal[-1] = bool(goal)
            if goal:
                assert distances[-1, int(goal)-1] <= .5001
            old = -d[1:]
            old_error = abs(old.sum()-float(original_return))
            assert old_error < .05, (path, ix, old_error)
            max_old_error = max(max_old_error, old_error)
            raw = d[:-1]-d[1:]
            offset = (1-GAMMA)*d[1:]
            f = raw+offset
            if goal:
                f[-1] = d[-2]  # Phi(true terminal)=0; do not zero timeout Phi.
            w = GAMMA**np.arange(n)
            endpoint_term = 0. if goal else GAMMA**n*d[-1]
            identity_error = abs(np.dot(w, f)-(d[0]-endpoint_term))
            assert identity_error < 1.e-10
            max_identity_error = max(max_identity_error, identity_error)
            a = np.column_stack([raw, offset, f, old, np.linalg.norm(np.diff(line, axis=0), axis=1),
                                 np.full(n, bool(goal)), terminal])
            file_arrays.append(a)
            row = dict(task=task, step=int(z['env_steps']), episode=ix, length=n,
                       success=bool(goal), start_distance=float(d[0]), end_distance=float(d[-1]),
                       old_return=float(old.sum()), discounted_potential_sum=float(np.dot(w, f)),
                       timeout_endpoint_correction=endpoint_term,
                       note='Finite observed prefix only for timeouts; not a bootstrapped Q estimate.')
            for c in COEFFICIENTS:
                row[f'new_discounted_return_c{c}'] = float(np.dot(w, f-c))
            episodes.append(row)
        a = np.concatenate(file_arrays); arrays.append(a)
        summaries.append(dict(step=int(z['env_steps']), episodes=len(z['lengths']),
                              successes=int((z['goals'] > 0).sum()), transitions=len(a), statistics=summarize(a)))
        checks.append(dict(input=str(path), old_reward_max_abs_error=max_old_error,
                           potential_telescoping_max_abs_error=max_identity_error))
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    combined = np.concatenate(arrays); data[task] = combined
    groups[task] = dict(checkpoints=len(files), episodes=sum(x['episodes'] for x in summaries),
                        transitions=len(combined), statistics=summarize(combined), by_checkpoint=summaries)

time_cost = []
for c in COEFFICIENTS:
    cost = lambda n: c*(1-GAMMA**n)/(1-GAMMA)
    time_cost.append(dict(c=c, cost_300=cost(300), cost_400=cost(400),
                          faster_300_vs_400_gain=cost(400)-cost(300),
                          nonterminating_cost=c/(1-GAMMA)))
output = dict(training_source='484f92e7d6d34c964d85b4493ff17c5a9ebcf32e',
              report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              gamma=GAMMA, potential_coefficient=1., candidates=COEFFICIENTS,
              groups=groups, time_cost=time_cost, inputs=inputs, checks=checks,
              scope='Existing T1 DACER-OFF seed0 random-start direct-policy evaluation trajectories. '
                    'Multiple checkpoints of the same training seed; transitions and episodes are not independent training replicates. '
                    'v4 snapshot ends at 4M, not final 5M. No new training, rollout, or policy change.',
              limitations='Counterfactual rewards on fixed old trajectories do not predict a retrained policy. '
                          'Per-step potential reward can be positive without progress due to (1-gamma)*distance. '
                          'All timeout return totals are truncated prefixes, with the nonzero endpoint term retained.')
(OUT/'results.json').write_text(json.dumps(output, indent=2)+'\n')
with (OUT/'episodes.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(episodes[0])); writer.writeheader(); writer.writerows(episodes)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.7))
for panel, task in zip(ax, GOALS):
    a = data[task]; b = a[(a[:, 5] == 1) & (a[:, 6] == 0)]
    panel.hist(b[:, 0], bins=np.linspace(-.15, .6, 101), density=True, alpha=.5,
               label='Literal distance decrease', color='#3976ba')
    panel.hist(b[:, 2], bins=np.linspace(-.15, .6, 101), density=True, alpha=.4,
               label='Discount-corrected potential', color='#df8b32')
    for c, color in [(.1, '#39956a'), (.3, '#b94343')]:
        panel.axvline(c, color=color, linestyle='--', label=f'Penalty magnitude {c}')
    panel.set_title(f'{task}: successful-episode transitions\n{len(b):,} nonterminal samples')
    panel.set_xlabel('Reward contribution (lambda = 1)')
    panel.set_xlim(-.15, .6); panel.set_ylabel('Density')
ax[0].legend(fontsize=8)
fig.suptitle('Proposed reward scale on preserved trajectories; no retraining')
fig.text(.5, .015, 'Random-start direct policy, one training seed across multiple checkpoints. '
         'Successful episodes shown; failures and all samples retained in results.json.\n'
         'Gamma = 0.99. A positive reward fraction is not a measure of policy quality. Penalty 1.0 is beyond this axis.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .11, 1, .92]); fig.savefig(OUT/'reward_components.png', dpi=170); plt.close(fig)

lines = [
    '# AntMaze step penalty 보상 스케일 검토', '',
    '기존 T1, DACER OFF, NovelD OFF, seed0의 random-start direct-policy 평가 궤적에 새 보상을 사후 계산했다. '
    '새 학습이나 rollout은 수행하지 않았다. v4는 보관된 4M 평가까지만 포함하며 최종 5M 결과가 아니다.', '',
    f"총 {sum(g['episodes'] for g in groups.values()):,} episodes, "
    f"{sum(g['transitions'] for g in groups.values()):,} transitions. 여러 checkpoint는 독립 training seed가 아니다.", '',
    '보상 정의: r = -c + lambda * [gamma Phi(next) - Phi(current)], Phi=-nearest-goal distance, '
    'gamma=.99, lambda=1. True goal terminal만 Phi=0; timeout은 실제 endpoint potential을 보존한다.', '',
    '|환경|평가 episodes|성공 episode transition 수(종료 제외)|순수 거리 감소량 중앙값|할인 보정 보조항 중앙값|보조항 10~90백분위|',
    '|---|---:|---:|---:|---:|---:|']
for task, group in groups.items():
    g = group['statistics']['successful_episode_nonterminal']
    r, p = g['raw_progress'], g['potential_increment']
    lines.append(f"|{task}|{group['episodes']}|{r['n']}|{r['p50']:.4f}|{p['p50']:.4f}|{p['p10']:.4f}~{p['p90']:.4f}|")
lines += ['', '## 후보 비교: 같은 기존 궤적에서 보상만 재계산', '',
          '|환경|c|평균 새 보상(성공 궤적, 비종료)|새 보상>0 비율|', '|---|---:|---:|---:|']
for task, group in groups.items():
    for c in [.1, .3, 1.]:
        x = group['statistics']['successful_episode_nonterminal']['candidate_rewards'][str(c)]
        lines.append(f"|{task}|{c}|{x['mean_potential_reward']:.4f}|{100*x['positive_potential_reward_fraction']:.2f}%|")
lines += ['',
    '양수 비율은 성공률이 아니며, 높을수록 더 좋은 보상 설계라는 뜻도 아니다. 실패 궤적과 전체 표본 통계는 results.json에 포함했다.', '',
    '## 해석', '',
    '- c=1은 성공 궤적의 보조항 중앙값보다 약7~11배, c=.3은 약2~3배다. '
    '같은 수치 규모의 벌점을 원한다면 c=.1이 더 가까운 후보다. 단일 step 비교로 최적 c를 확정할 수는 없다.',
    '- gamma 보정 때문에 F = (d_current-d_next) + .01*d_next 이다. '
    '거리17m에서 정지해도 F=.17이고 c=.1이면 r=.07이다. '
    '따라서 이 식을 순수한 "전진은 양수, 후퇴는 음수" 보상이라고 설명하면 부정확하다.',
    '- 올바른 terminal 처리에서 잠재함수의 할인합은 d_start - gamma^H*d_endpoint이다. '
    '목표 도달 시 endpoint 항은0이다. 정지 보상이 양수라는 이유만으로 정지가 최적이라고 판단할 수는 없다. '
    '모든 trajectory에서 이 항등식을 수치 검증했다.',
    '- 같은 시작 상태에서300step과400step에 성공하는 두 경로의 보상 차이는 c=.1/.3/1에서 '
    '각각 .311/.933/3.109다. T=1을 유지하면 이 차이는 시간 비용 대비 탐색 비중도 바꾼다.',
    '- 현재 할인 보정식을 유지하는 한 c=.1 중심, c=.3 강한 시간 비용 대조를 우선 검토할 수 있다. '
    'c=1을 잘못된 값이라고 확정하거나 학습 실패를 예측한 결과는 아니다.',
    '- 문자 그대로의 r=-c+(d_current-d_next)를 선택한다면 비교해야 할 것은 raw_progress 열이다. '
    '성공 궤적 중앙값이 .046~.076이므로 .03~.05가 비슷한 수치 규모지만, 이는 별도 보상 정의이며 '
    'gamma<1에서 위 potential-based 정책 보존 성질을 그대로 주장할 수 없다. 이번에 변경하거나 승인된 설정이 아니다.', '',
    '## 검증과 한계', '',
    f"기존 dense return 재구성 최대 오차: {max(x['old_reward_max_abs_error'] for x in checks):.8g}. "
    f"Potential 할인합 항등식 최대 오차: {max(x['potential_telescoping_max_abs_error'] for x in checks):.8g}.",
    '원자료 SHA256, 학습 source SHA, 분석 script SHA는 results.json에 기록했다. '
    '새 보상으로 재학습한 정책의 행동 변화나 critic 오차는 이번 사후 계산으로 알 수 없다. '
    '기존 정책의 Q는 새 보상의 Q가 아니므로 직접 비교하지 않았다.', '',
    '![보상 성분 분포](' + str(OUT/'reward_components.png') + ')', '']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines))

print(json.dumps({task: {k:v for k,v in group.items() if k not in ('statistics','by_checkpoint')}
                  for task,group in groups.items()}, indent=2))
for task, group in groups.items():
    for subset in ['all_nonterminal', 'successful_episode_nonterminal', 'away_from_goal_nonterminal',
                   'xy_motion_under_1mm_nonterminal']:
        a = group['statistics'].get(subset)
        if a:
            print(task, subset, json.dumps(a))
