"""Post-hoc replay/cadence hypothesis audit; no learner or environment calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
SOURCE = WORK / 'tmp/reward-progress-worktree'
sys.path.insert(0, str(SOURCE))
from antmaze_experiments.critic_diagnostics import route_label


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


data = {}
inputs = {}
for task, lost, kept in [('v3', 'left', 'right'), ('v4', 'lower', 'upper')]:
    replay_path = ROOT/'data'/task/'replay-audit.json'
    replay = read(replay_path)
    run = next((WORK/'artifacts/antmaze_critic_dynamics/results').glob(
        '*/runs/'+task+'-optiq-control-500k-s0'))
    config, result, proof = [read(run/f) for f in
                             ['config.json', 'result.json', 'checkpoint-verification.json']]
    assert replay['checkpoint_sha256'] == proof['sha256']
    assert replay['step'] == proof['replay_count'] == proof['steps'] == 508416
    assert not replay['replay_metadata']['if_full'] and replay['all_data_retained']
    assert replay['completed_episodes'] == result['training_episodes']
    assert sum(replay['successful_route_counts'].values()) == result['training_successes']
    assert config['source_commit'] == replay['source_commit'] == '23603a7e7696aa64e8e49a38b986d6b430b6d6f3'
    actor = config['native']['alg']['actor']
    assert actor['density_correction'] and actor['density_beta'] == 1. and actor['temperature'] == 1.
    assert (config['num_envs'], config['batch_size'], config['updates_per_vector_step']) == (256, 4096, 8)
    evaluations = []
    for p in sorted(run.glob('evaluations/*/policy-fixed/rollouts.npz')):
        summary = read(p.parent/'summary.json')
        with np.load(p, allow_pickle=False) as d:
            labels = [route_label(task, xy[:int(n)+1])[0]
                      for xy, n in zip(d['xy'], d['lengths'])]
            starts = d['initial_full_state']
            np.testing.assert_array_equal(starts, np.repeat(starts[:1], len(starts), axis=0))
            assert len(labels) == summary['episodes']
            assert np.isclose(np.mean(d['goals'] > 0), summary['success_rate'])
            evaluations.append(dict(step=int(d['env_steps']), episodes=len(labels),
                routes=dict(Counter(labels)), successes=dict(Counter(
                    label for label, goal in zip(labels, d['goals']) if goal > 0))))
        inputs[str(p.relative_to(WORK))] = sha(p)
    assert evaluations[-1]['step'] == replay['step']
    for r in replay['occupancy']:
        assert r['overwritten'] == 0
    final = replay['final_regions']
    minibatch = {k: v['fraction'] * 4096 for k, v in final.items()}
    waves = Counter(((e['end']-1)//256+1)*256 for e in replay['episodes'])
    data[task] = dict(lost=lost, kept=kept, replay=replay, evaluations=evaluations,
                      expected_region_samples_per_batch=minibatch,
                      reset_waves=waves.most_common(8))
    inputs[str(replay_path.relative_to(WORK))] = sha(replay_path)
    inputs[str((run/'config.json').relative_to(WORK))] = sha(run/'config.json')

env32 = read(WORK/'artifacts/antmaze_env32_v34_250k/report/results.json')
cadence = env32['latest']
clock = dict(capacity_transitions=1000000, envs=256, updates_per_collection=8,
    vector_steps_per_window=1000000/256, learner_updates_per_window=1000000/256*8,
    first_eviction_total_transition=1000192, updates_at_first_eviction=31000,
    steady_state_expected_draws_per_transition=4096*8/256,
    per_700_step_episode_updates_256env=700*8,
    per_700_step_episode_updates_32env=700,
    miss_probability_64_candidates={str(p): (1-p)**64 for p in [.05, .01, .001, .0001]})
out = dict(analysis_only=True, training_modified=False, source_commit='d3ae8fefccd9d68fe9a6326d3c2f21e452940638',
    audit_sha256=sha(ROOT/'audit_replay.py'), reporter_sha256=sha(Path(__file__)),
    clock=clock, data=data, prior_env32_comparison=cadence, inputs_sha256=inputs,
    conclusion='FIFO eviction is not necessary for observed pre1M route loss. Broader state-distribution/policy-feedback effects remain unisolated.',
    limitations=['Single training seed per condition.', 'Region occupancy is not route success or matched-state action coverage.',
                 'Zero observed branch use is not proof of exactly zero probability.',
                 'Historical controls use256x3/mean-init1 and progress reward; no new basic-profile training is represented.',
                 'Full v4 checkpoint was checked in place on server199 using CPU; the checkpoint remains on that server.'])
(ROOT/'results.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), layout='constrained')
for row, task in enumerate(['v3', 'v4']):
    d = data[task]; lost, kept = d['lost'], d['kept']
    es, rs = d['evaluations'], d['replay']['occupancy']
    for name, color in [(lost, '#276AB0'), (kept, '#D77A26')]:
        axes[row, 0].plot([x['step']/1000 for x in es],
            [x['routes'].get(name, 0)/x['episodes']*100 for x in es], marker='o',
            color=color, lw=2, label=name.title())
        axes[row, 1].plot([x['step']/1000 for x in rs],
            [x['cumulative_counts'][name]/x['retained']*100 for x in rs], color=color, lw=2)
        axes[row, 2].plot([x['step']/1000 for x in rs],
            [x['recent_counts'][name]/(x['step']-x['recent_start'])*100 for x in rs], color=color, lw=2)
    axes[row, 0].plot([x['step']/1000 for x in es],
        [x['routes'].get('uncommitted', 0)/x['episodes']*100 for x in es],
        color='#9A9FA5', ls=':', label='No gate entry')
    axes[row, 0].legend(fontsize=9)
    for col, title in enumerate(['Direct-policy route entry', 'All replay: coordinate regions', 'New data: last ~50k transitions']):
        ax=axes[row, col]
        ax.set_title(f'{task.upper()} | {title}')
        ax.set_xlim(0, 525); ax.set_ylim(0, 105 if col==0 else 100)
        ax.set_xlabel('Total environment transitions (k)');ax.grid(alpha=.15)
        ax.set_ylabel('Episodes (%)' if col==0 else 'Transitions (%)')
    axes[row, 1].text(.04,.96,'FIFO deletions = 0\nFinal buffer: 508,416 / 1,000,000',
                      transform=axes[row,1].transAxes, va='top', fontsize=10)
fig.suptitle('Both policies lose a branch before the replay buffer is full', fontsize=16)
fig.supxlabel('Historical seed0 controls: T=1, reward=100 distance progress, DACER/NovelD OFF.\nPolicy draws include fresh latent and conditional sigma. Spatial occupancy does not measure successful routes.',fontsize=10)
fig.savefig(ROOT/'replay_vs_route.png', dpi=170)
fig.savefig(ROOT/'replay_vs_route.pdf')
plt.close(fig)

lines = ['# Replay 가설 검증 — 저장된 두 정책 전체 버퍼 감사', '',
    '학습과 환경 rollout 없이 기존 v3/v4 control checkpoint를 CPU에서 읽었다. 원본 SHA256, FIFO metadata, 전체29D transition의 환경별 연속성/700step timeout을 검증했다. 학습 source는23603a7이며, 방금 복원한 basic 프로필의 새 결과가 아니다.', '',
    '**결론: 1M 버퍼의 FIFO 교체가 최초 양방향 소실을 일으켰다는 설명은 두 control에서 성립하지 않는다. 정책·방문 분포·critic의 상호작용은 여전히 가능하지만, 버퍼 용량 원인과는 분리해야 한다.**', '',
    '|환경|최종 직접평가 경로|전체 버퍼 영역 transition|완료된 학습 episode의 통로|학습 성공|',
    '|---|---|---|---|---|']
for task, d in data.items():
    r=d['replay']; e=d['evaluations'][-1]
    lines.append(f"|{task}|{e['routes']}|{ {k:v['count'] for k,v in r['final_regions'].items()} }|{r['completed_route_counts']}|{r['successful_route_counts']}|")
lines += ['', '두 정책 모두508,416 transitions/15,632 updates에서 if_full=false,next_p=cur_capacity=total_samples=508416이다. 삭제·덮어쓰기0개. 학습 episode는 v3 533개,v4 512개로 기존 result와 정확히 일치한다. 미완료 episode는 따로 집계했으며 성공 경로와 통로 진입을 혼동하지 않았다.', '',
    'v3는300032 평가 left14/right24→500224 left0/right39가 됐지만, 같은500224 buffer에는 left91,746/right46,836 transition이 남아 있었다. v4는200192 upper14/lower19→500224 upper40/lower0이고 buffer에는 upper135,913/lower49,208이 있었다. 큰 버퍼가 필요한 시점보다 먼저 소실됐다.', '',
    '![정책 경로와 전체 replay 분포](replay_vs_route.png)', '',
    '## 수집 시간·학습 시간', '',
    '1M/256=3906.25는 vector collection 반복 수다. Warmup 이후1M개를수집하는동안 learner update는31250회이며, 32env/1update에서도 같은31250회다. 첫덮어쓰기는total1,000,192에서일어나며,8192warmup을제외한당시업데이트는31,000회다. 정상상태에서한transition의기대batch추출횟수는4096×8/256=128이다. 반면warmup후700step episode동안 online정책업데이트는256env에서5600회,32env에서700회로달라진다. 병렬화가설에서별도로검증할부분은에피소드내정책변화와수집상관성이다.', '',
    '실제로 두control 모두179200과358400 total transitions에서256개환경이같은수집반복에동시에reset했다. v4는500k까지성공이없어완료episode512개가정확히이두묶음이다. 따라서동일한episode길이로초기화된병렬환경의수집위상동기화는관측되었다. 이것이경로소실의원인인지는분리실험이필요하다.', '',
    '기존256→32 paired cadence screen에서도 양방향 성공 유지가 회복되지는 않았다. v3 최종직접평가의 left/right 진입은11/87→7/88, 성공은right44→48이다. v4는upper71/lower29→lower100으로 바뀌었다. 각 조건1seed/258304 transitions이며 보상이 현재control과 다르므로 보편적인 병렬효과 추정은 아니다.', '',
    '## Importance correction이 보장하는 범위', '',
    '실제코드와 해당run config에서 beta1,density_correction=true를 확인했다. w_i ∝ exp(Q(s,a_i)/T)/q(a_i|s), a_i~q이고 이상적인 고정-Q/충분한 proposal/정확한 최적화에서는 q편향이 상쇄되어 exp(Q/T) target을 따른다. Q가 변하면 target 자체가 변하며, 그 보정은 replay의 state 빈도나 critic 오차를 보정하지 않는다. 낮은-density 행동이64개 후보에 없으면 직접적인 가중치도 생기지 않는다. 예를 들어 후보영역확률1%면64개모두누락될 확률52.6%,0.1%면93.8%다. 이 확률은 한state의 행동영역에 대한 계산이며 장기경로 확률과 동일시하면 안 된다.', '',
    'GMM40 fixed-Q 분포모사 결과는 주어진 energy의 여러 mode를 표현/피팅하는 능력을 보여준다. 부트스트랩 Q가 바뀌고 replay의 state 분포와 미래 행동이 바뀌는 RL에서 양쪽 성공경로 유지가 자동보장됨을 증명하지는 않는다. 이번control의 두쪽 초기진입도 양쪽 성공정책 획득을 뜻하지 않았다. v3 학습성공22개는모두right이고v4는0개다.', '',
    '## 남은 인과검증', '',
    '1M보다 큰 buffer만 바꾸는 비교는첫1M까지 동일한 저장/표본분포이므로 이번500k 소실의 원인을 직접 시험하지 못한다. 용량1M/4M을 비교하려면1M를 넘겨추가학습하면서 늦은반대경로 회복/재학습을 보아야 한다. 병렬수32/256은별도요인으로두고8/256=1/32 update ratio,batch4096,T,보상,모델,seed를고정해야 한다. 별도로같은분기state와양쪽행동bank에서 Q·proposal확률·teacher질량·실제조건부return의선후변화를봐야state분포→Q→actor피드백을판별할수있다.', '',
    '이번 작업은 원자료 분석만 수행했으며 새 학습 큐를 등록하지 않았다. 전체checkpoint SHA와 입력rollout/분석코드 SHA는results.json에 기록했다. 0/40·0/100 관측은 정확히0 확률이라는 뜻이 아니다.']
(ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({task: dict(final_regions=d['replay']['final_regions'],
                           minibatch=d['expected_region_samples_per_batch']) for task,d in data.items()},indent=2))
