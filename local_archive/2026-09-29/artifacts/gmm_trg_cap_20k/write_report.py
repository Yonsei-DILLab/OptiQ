import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
jobs=json.loads((ROOT/'host180.json').read_text())['jobs']
rows=json.loads((ROOT/'summary.json').read_text())
assert len(jobs)==4 and all(j['status']=='completed' for j in jobs)
for task in ('ant','humanoid'):
    pair=[j for j in jobs if j['task']==task]
    for mode in ('zero_z','stochastic_z'):
        a,b=(j['evaluations'][mode] for j in pair)
        for key in ('timesteps','env_seeds','policy_seeds'):
            assert a[key]==b[key]
        np.testing.assert_array_equal(np.array(a['results'])[:6],np.array(b['results'])[:6])

lines=['# Ant·Humanoid: log σ 상한 0 대 −2, 20k 비교','',
'4개 run 모두 20,000 환경 step 완료. seed 0, 초기 σ=0.1로 통일했습니다. ',
'기본 warmup 5,000 step을 포함하므로 실제 actor/critic update는 각각 15,000회입니다.',
'상한 외 학습 설정은 동일합니다: 64×64, 랜덤 normal latent, T=0.25, batch 256, UTD 1, '
'policy delay 1, Adam 3e-4, 256×2 GELU, gradient clipping 없음.','',
'기존 sweep은 사용자 요청으로 중지하고, 이 비교는 네 조건 모두 처음부터 실행했습니다.',
'학습 시작 전 두 조건의 평균과 σ가 동일함을 검증했습니다. 평가 episode seed와 policy seed가 '
'조건 간 동일하며, warmup 동안의 평가 reward도 정확히 일치함을 확인했습니다.','',
'## Reward 비교','',
'평가마다 mode별 10개 episode 평균입니다. `stochastic_z`는 매 action마다 z를 새로 뽑고 '
'Gaussian noise ε는 0으로 두는 conditional-mean 평가입니다. `zero_z`는 z=0, ε=0입니다.',
'아래 AUC는 5k~20k reward 곡선을 사다리꼴 적분한 후 15,000으로 나눈 구간 평균입니다.','',
'| 환경 | log σ 상한 | 평가 | 10k reward | 20k reward | 20k episode 표준편차 | 5k~20k 평균 |',
'|---|---:|---|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['task'],r['mode'],-r['log_sigma_cap'])):
    lines.append(f"| {r['task']} | {r['log_sigma_cap']:g} | {r['mode']} | {r['return_10k']:.1f} | {r['final_return']:.1f} | {r['eval_episode_std']:.1f} | {r['mean_return_after_warmup']:.1f} |")
lines+=['','## σ가 상한에 붙는 정도','',
'σ는 box truncation 이전 Gaussian의 scale parameter입니다. 실제 제한된 action 분포의 표준편차와 다릅니다.',
'상한 도달률은 훈련 minibatch의 state × z × action coordinate 출력 중 '
'`log σ >= configured_max - 1e-6`인 비율입니다. 고정된 64개 component의 비율이 아닙니다.','',
'| 환경 | log σ 상한 | σ 상한 | 20k 평균 σ | 20k 평균 log σ | 20k 최대 log σ | 상한 도달률 |',
'|---|---:|---:|---:|---:|---:|---:|']
for r in sorted((r for r in rows if r['mode']=='stochastic_z'),key=lambda r:(r['task'],-r['log_sigma_cap'])):
    lines.append(f"| {r['task']} | {r['log_sigma_cap']:g} | {np.exp(r['log_sigma_cap']):.4f} | {r['mean_sigma']:.4f} | {r['mean_log_sigma']:.4f} | {r['max_log_sigma']:.4f} | {100*r['fraction_at_cap']:.2f}% |")
lines+=['','## 실제 관측한 reward 추이','']
for task in ('ant','humanoid'):
    rs={r['log_sigma_cap']:r for r in rows if r['task']==task and r['mode']=='stochastic_z'}
    a,b=rs[0],rs[-2]
    lines.append(f"- {task}: 10k→20k reward는 상한 0에서 {a['return_10k']:.1f}→{a['final_return']:.1f}, "
        f"상한 −2에서 {b['return_10k']:.1f}→{b['final_return']:.1f}입니다. "
        f"5k~20k 구간 평균은 각각 {a['mean_return_after_warmup']:.1f}, {b['mean_return_after_warmup']:.1f}입니다.")
    first=next(j for j in jobs if j['task']==task)['evaluations']['stochastic_z']
    baseline=float(np.mean(first['results'][first['timesteps'].index(5000)]))
    if max(a['final_return'],b['final_return'])<baseline:
        lines.append(f"  두 조건 모두 warmup 시점의 미학습 mean-policy reward {baseline:.1f}보다 낮습니다. "
            "따라서 상대적으로 높은 조건을 성공적인 수렴으로 해석하면 안 됩니다.")
    if abs(a['mean_return_after_warmup']-b['mean_return_after_warmup'])<.02*max(abs(a['mean_return_after_warmup']),abs(b['mean_return_after_warmup'])):
        lines.append('  구간 평균 차이가 작으므로 최종 reward의 차이만으로 어느 조건이 더 빨리 학습한다고 결론낼 수 없습니다.')
lines+=['',f'![reward와 scale 추이]({ROOT / "cap_comparison.png"})','',
    f'![평가 mode별 reward]({ROOT / "both_eval_modes.png"})','']
lines+=['','## 해석할 때 구분할 점','',
'- 상한 −2는 σ를 최대 0.1353으로 제한하며, 상한 0은 최대 1을 허용합니다. '
'따라서 이 변경은 rollout noise와 teacher proposal 분포 양쪽을 바꿉니다.',
'- 두 평가 mode 모두 ε=0입니다. reward 차이는 평가 시 Gaussian noise를 더 넣어서 생기는 직접 효과가 아니라, '
'서로 다른 rollout/teacher 분포로 학습된 conditional mean policy의 차이입니다.',
'- policy는 raw log σ를 hard clip합니다. 상한 바깥에서는 해당 출력의 clipping gradient가 0이 됩니다. '
'상한 도달률이 높은 상태가 지속된다면 σ가 자유롭게 학습된다는 해석보다 상한이 scale을 결정하고 있다는 해석이 적절합니다.',
'- teacher 가중치는 exp(Q/T)/proposal_density에 비례합니다. 이론적으로 Q가 평평하고 샘플이 충분하면 '
'bounded action 공간의 uniform target에 가까워집니다. 이것은 초기 scale 확대의 가능한 설명이지만, '
'이번 두 상한 비교만으로 density correction과 Q, hard clipping의 기여를 각각 확정할 수는 없습니다.',
'- 같은 seed의 20k 초기 학습 비교입니다. episode 표준편차는 training seed 간 불확실성이 아닙니다. '
'최종 한 시점의 reward와 전체 구간 학습 속도를 구분해야 합니다.','',
'## 기록과 재현','',
'- W&B 프로젝트: https://wandb.ai/OptiQ/abla',
'- source commit: `d01fc50c004f8bd0f7f9a10a8a5dac63d4e97654`',
'- 전체 곡선: `cap_comparison.png`, `cap_comparison.pdf`',
'- 평가 mode별 곡선: `both_eval_modes.png`',
'- 원시 평가 reward와 episode seed: `host180.json`',
'- 비교 수치: `comparison.csv`, `summary.json`',
'- source backup: `source.bundle`','']
for j in jobs:lines.append(f"- {j['task']}, cap {j['cap']:g}: {j['wandb_url']}")
(ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print('Verified paired evaluation seeds, identical warmup rewards, and wrote REPORT_KO.md')
