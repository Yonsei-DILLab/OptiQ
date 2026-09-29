"""Reproducible analysis of preserved logs and CPU-only checkpoint diagnostics."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CONTROL = 'antmaze-optiq-dense-off-T1-s0-20260924'
histories = {}
for host in (180, 199):
    histories.update(json.loads((ROOT/f'histories-{host}.json').read_text()))
assert all(not run['read_warnings'] for run in histories.values())


def diagnostic(name):
    return json.loads((ROOT/name).read_text().split('AUDIT_JSON=')[-1])


forward = diagnostic('forward-180.log')
other = diagnostic('forward-199.log')
forward['results'].update(other['results'])
forward['inputs'].update(other['inputs'])
own = diagnostic('forward-baseline-own.log')


def entropy_rows(run):
    unique = {}
    for row in run['history']:
        if 'exploration/entropy_proxy' not in row:
            continue
        key = int(row['exploration/updates'])
        if key in unique:
            assert abs(unique[key]['entropy_total']-row['exploration/entropy_proxy']) < 1e-7
        else:
            unique[key] = dict(count=key, first_logged_step=row['step'],
                               entropy_total=row['exploration/entropy_proxy'],
                               entropy_per_dim=row['exploration/entropy_proxy']/8,
                               noise_std=row['exploration/noise_std'])
    return [unique[k] for k in sorted(unique)]


def replay_adam(rows, target):
    p, m, v = math.log(.27), 0., 0.
    curve = []
    for i, row in enumerate(rows, 1):
        g = row['entropy_total']-8*target
        m = .9*m+.1*g
        v = .999*v+.001*g*g
        p -= .03*(m/(1-.9**i))/(math.sqrt(v/(1-.999**i))+1e-8)
        curve.append(.1*math.exp(p))
    return curve


def late_mean(run, key):
    budget = run['config']['steps']
    values = [r[key] for r in run['history'] if budget-100000 < r.get('step', -1) <= budget and key in r]
    return float(np.mean(values)) if values else None


targets = [-.9, -.5, -.1, .3, .4, .5, .6, .65, .7]
controls, sweep = {}, {}
for maze in ('v1','v2','v3','v4'):
    run = histories[f'{CONTROL}/{maze}-optiq-s0']
    rows = entropy_rows(run)
    assert len(rows) == run['regulator']['updates']
    simulated = {str(t):replay_adam(rows,t) for t in targets}
    assert np.allclose(simulated['-0.9'],[r['noise_std'] for r in rows],atol=2e-7,rtol=0)
    controls[maze] = dict(source=run['config']['source_commit'],budget=run['config']['steps'],
        regulator_observations=rows,final_entropy_per_dim=rows[-1]['entropy_per_dim'],
        final_noise_std=rows[-1]['noise_std'],last100k_sigma_parameter_mean=late_mean(run,'train/actor_std_mean'),
        fixed_observation_adam_replay={t:dict(final_std=c[-1],curve=c) for t,c in simulated.items()})
for name, run in histories.items():
    if 'dacer-entropy' in name:
        rows = entropy_rows(run)
        if not rows: continue
        sweep[name] = dict(target=run['config']['dacer_target_entropy_per_dim'],
                           last_logged_step=max(r.get('step',0) for r in run['history']),
                           regulator_observations=rows)

baseline_logs = {}
for name, run in histories.items():
    if run['config']['method'] not in ('sac','mfpo'): continue
    baseline_logs[name] = dict(method=run['config']['method'], task=run['config']['task'],
        native_entropy_last100k=late_mean(run,'entropy'),
        sac_entropy_weight_last100k=late_mean(run,'train/alpha'))

result = dict(time=datetime.now(timezone.utc).isoformat(), single_training_seed=0,
    controls=controls,cancelled_sweep=sweep,baseline_logs=baseline_logs,
    checkpoint_diagnostics=forward,baseline_own_state_diagnostics=own,
    counterfactual_caveat='Scalar Adam replay holds observed entropy fixed. This is not a retrained-policy prediction and does not include policy/state feedback.',
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('histories-*.json')},
    recommendation=dict(target_entropy_per_dim_candidates=[.6,.65],preferred_initial_candidate=.65,
        total_for_8_actions=5.2,validated_optimum=False,
        reason='These targets reverse the observed regulator tendency, but very sparse alpha updates keep added noise small. Current OptiQ action spread is already large relative to baselines on their own replay states.',
        no_positive_target_experiments_launched=True))
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

colors=dict(v1='#2563eb',v2='#0f9d87',v3='#d97706',v4='#934bb4')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes=plt.subplots(2,2,figsize=(12.5,8.8),layout='constrained')
for maze, item in controls.items():
    rows=item['regulator_observations'];x=np.array([r['first_logged_step'] for r in rows])/1e6
    axes[0,0].plot(x,[r['entropy_per_dim'] for r in rows],'-o',ms=3,label=maze,color=colors[maze])
    axes[0,1].step(np.r_[0,x],np.r_[.027,[r['noise_std'] for r in rows]],where='post',label=maze,color=colors[maze])
    ts=[.3,.4,.5,.6,.65,.7]
    axes[1,0].plot(ts,[item['fixed_observation_adam_replay'][str(t)]['final_std'] for t in ts],'-o',ms=4,label=maze,color=colors[maze])
axes[0,0].axhspan(.6,.65,color='#9ca3af',alpha=.18,label='Candidate target +0.60 to +0.65')
axes[0,0].set(title='Recorded GMM entropy proxy / action dimension',xlabel='Environment interactions (million)',ylabel='Proxy (nats / dimension)',ylim=(.30,.76))
axes[0,0].legend(fontsize=9,ncol=2,loc='lower right')
axes[0,1].axhline(.027,color='black',ls=':',lw=1)
axes[0,1].set(title='Actual DACER extra-noise std: target -0.9',xlabel='Environment interactions (million)',ylabel='Extra-noise standard deviation',ylim=(.015,.029))
axes[0,1].text(.04,.1,'Only 10 / 10 / 13 / 16 regulator updates\nPolicy sigma parameters: 0.357-0.365',transform=axes[0,1].transAxes,fontsize=10)
axes[1,0].axhline(.027,color='black',ls=':',lw=1,label='Initial extra-noise std = .027')
axes[1,0].set(title='Fixed-log Adam replay: final extra-noise std',xlabel='Hypothetical target / dimension',ylabel='Extra-noise standard deviation')
axes[1,0].legend(fontsize=9,loc='upper left')
axes[1,0].text(.97,.04,'Counterfactual only; no retraining',ha='right',transform=axes[1,0].transAxes,fontsize=9)
labels=['OptiQ','SAC','DIPO','DIPO\n+ train noise','MFPO']
rr=own['results']
values=[rr[f'v3-{m}']['policy']['action_rms_coordinate_std'] for m in ('optiq','sac','dipo')]
values += [rr['v3-dipo']['population_behavior_with_mixed_noise']['action_rms_coordinate_std'],rr['v3-mfpo']['policy']['action_rms_coordinate_std']]
bars=axes[1,1].bar(labels,values,color=['#2563eb','#64748b','#d97706','#fbbf24','#0f9d87'],width=.64)
axes[1,1].bar_label(bars,fmt='%.3f',padding=3,fontsize=10)
axes[1,1].set(title='v3: action spread on each policy\'s own replay states',ylabel='RMS within-state coordinate std',ylim=(0,.53))
axes[1,1].text(.5,.94,'24 states x 400 actions; state pools differ',ha='center',transform=axes[1,1].transAxes,fontsize=9)
for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('AntMaze entropy / exploration-noise audit\nPreserved T=1 policies, one seed; no new learning in diagnostics',fontsize=16)
fig.savefig(ROOT/'entropy_noise_audit.png',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
for maze in ('v1','v2','v3','v4'):
    row=forward['results'][maze+'-optiq'];noise=[0,.02,.05,.1,.2,.35]
    samples=[row['policy']]+[row['noise_grid'][str(n)] for n in noise[1:]]
    axes[0].plot(noise,[s['gmm_joint_proxy_per_dim'] for s in samples],'-o',ms=4,color=colors[maze],label=maze)
    axes[1].plot(noise,[100*(s['action_rms_coordinate_std']/samples[0]['action_rms_coordinate_std']-1) for s in samples],'-o',ms=4,color=colors[maze],label=maze)
axes[0].set(xlabel='Added noise std before clipping',ylabel='GMM proxy / action dimension',title='Frozen-policy noise calibration')
axes[1].set(xlabel='Added noise std before clipping',ylabel='Change in action spread (%)',title='Current extra std ~.02 barely changes action spread')
axes[0].legend();axes[1].legend()
for ax in axes:ax.grid(alpha=.18)
fig.suptitle('CPU forward sampling only: 24 replay states / maze, 400 draws / state',fontsize=13)
fig.savefig(ROOT/'frozen_policy_noise_grid.png',dpi=180)
plt.close(fig)

def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])

control_table=table(['환경','마지막 Ĥ/d','추가 noise σ: 초기 → 최종','정책 σ 파라미터 평균','DACER 갱신 수'],[
    [m,f"{v['final_entropy_per_dim']:.3f}",f".027 → {v['final_noise_std']:.4f}",f"{v['last100k_sigma_parameter_mean']:.3f}",len(v['regulator_observations'])] for m,v in controls.items()])
replay_table=table(['환경','h*=+.50','h*=+.60','h*=+.65'],[
    [m]+[f"{v['fixed_observation_adam_replay'][str(t)]['final_std']:.4f}" for t in (.5,.6,.65)] for m,v in controls.items()])
baseline_table=table(['방법 / v3','상태 내 action 표준편차','추가 정보'],[
    ['OptiQ',f'{values[0]:.3f}','random z + conditional sigma; DACER 추가 noise 제외'],
    ['SAC',f'{values[1]:.3f}',f"tanh Gaussian 실제 log-density MC H/d={rr['v3-sac']['exact_squashed_gaussian_entropy_per_dim_mc']:.3f}; 목표 −1"],
    ['DIPO',f'{values[2]:.3f}','native diffusion 출력; reverse noise 포함'],
    ['DIPO + 학습 탐색',f'{values[3]:.3f}','환경별 mixed Gaussian σ=.05~.6 (원시 RMS .362), clip 적용'],
    ['MFPO',f'{values[4]:.3f}',f"학습된 density의 H/d={rr['v3-mfpo']['learned_logp_entropy_per_dim']:.3f}; 목표 −.5"]])
report=f'''# DACER OFF 실행 및 entropy/noise 분석

기존 entropy sweep의 실행·대기 작업을 중지했다. 완료된 v1 −1/−.5 두 결과와 중단 로그는 보존했다. 새 v1~v4 OptiQ seed0 네 개는 `dacer.enabled=false`, T=1, dense reward, NovelD OFF로 실행했다. 학습 소스는 `484f92e7d6d34c964d85b4493ff17c5a9ebcf32e`, 브랜치는 `direct-gmm-trg-antmaze`, W&B는 OptiQ/antmaze다. 양 서버의 canonical HEAD와 frozen source를 일치시켰다.

등록 검증은 `registration-verification.json`에 있다. 4개 모두 8 learner updates 사전검사 및 dense replay/checkpoint 검증을 통과했다. 동일 RNG에서 train action과 direct-policy action이 실제로 일치하며 DACER noise/update가 0이다. random latent와 conditional sigma는 유지한다. 기존 T1 대조군과 초기 actor/critic 해시, 모델·optimizer·reward·평가 설정이 일치하며 DACER enabled만 변경됐다. 예산은 v1/v2 3,008,256, v3 4,008,448, v4 5,008,384 interactions. 256환경, batch4096, 256수집당 8업데이트, 256×3, actor3e-4/critic5e-4를 유지했다.

## 결론

**현재 DACER 추정치 기준으로 잡음을 증가시키려면 차원당 +0.60~+0.65가 합리적인 시험 후보이며, 하나를 고르면 +0.65(8차원 총 +5.2)를 먼저 검토한다. 최적값으로 검증된 것은 아니다.** 현재 late Ĥ/d≈.42~.53보다 높고, 초기에 .6 이상으로 올라가는 구간도 고려한 값이다. 음수 −1~−.1은 모두 현재 추정치보다 낮아 같은 감소 방향이다. 기존 설명에서 +.5도 후보라고 했지만, 전체 로그를 같은 Adam으로 재계산하면 +.5만으로는 초기 noise .027보다 크게 성장하지 않는다.

더 중요한 제약은 **조절이 약 320k environment interactions마다 한 번**이라는 점이다. interval10000은 learner update 단위이고, 256개 수집당 8회 학습하므로 이 환산이 된다. 초기 한 번을 포함해 전 학습에서 10~16회에 불과하다. Adam lr=.03로 log-alpha를 갱신하므로 목표를 크게 올린다고 잡음이 비례해서 커지지 않는다. 목표를 충분히 초과시키는 일정 방향의 기울기라면 대략 갱신 한 번당 exp(.03)≈1.03배다. .027→.1에는 단순 계산으로 약44회, 현재 간격으로 약14M interactions가 필요하다. 이는 대략적인 속도 설명이며 Adam의 실제 변동 기울기에 대한 상한은 아니다.

## 실제 완료 대조군 로그

{control_table}

Ĥ는 GMM joint-entropy proxy를 action_dim=8로 나눈 값이다. 전체 학습에서는 첫 값 약.37에서 .62~.69까지 올랐다가 후반 .42~.53으로 내려왔다. 표의 Ĥ/noise는 마지막 DACER 갱신값이며, 정책 σ는 마지막100k 로그의 평균이다. σ는 truncated Gaussian의 파라미터이므로 실제 bounded action 표준편차와 다르다.

## 목표만 바꾼 계산

기존 Ĥ 관측열을 고정하고 동일한 scalar Adam(초기alpha.27, lr.03, b1=.9, b2=.999, eps1e-8)을 재생했다. 원래 −.9를 넣으면 실제 noise 로그를 2e-7 이내로 재현한다. 아래는 최종 추가 noise 표준편차다.

{replay_table}

초기 noise는 .027이다. **정책/방문상태가 바뀌지 않는 가상 계산이며, 새 목표로 실제 학습한 결과나 예측구간이 아니다.** +.65도 .031~.039 정도여서 증가 폭은 작다. 현재 요청대로 실행한 실험은 DACER OFF 네 개뿐이며 양수 목표 실험은 실행하지 않았다.

## baseline과 action-level 지표

저장된 최종 체크포인트 SHA256을 확인하고 CPU에서만 forward sampling했다. 각 방법 자신의 v3 replay에서 24개 상태를 뽑아 상태당 400개 행동을 샘플링했다. 아래 표준편차는 상태 내 분산을 좌표·상태에 평균한 뒤 제곱근을 취한 값이다. 서로 다른 replay 상태이므로 원인·성능의 공정한 비교로 단정하지 않는다.

{baseline_table}

이 진단에서는 **OptiQ의 직접 정책 행동 분산이 이미 작지 않다.** 낮은 추가 DACER noise만 보고 전체 행동 탐색이 좁다고 말할 수 없다. SAC alpha는 noise σ가 아니라 entropy reward 가중치이며, SAC의 −1이나 MFPO의 −.5를 현재 GMM proxy의 적정 목표로 그대로 옮길 수 없다. MFPO의 density 추정치와 GMM proxy도 서로 다른 양이다.

같은 v3 OptiQ replay 상태로도 네 정책을 샘플링했으며 그 결과는 analysis.json에 보존했다. SAC의 pre-tanh std가 커져 극단 포화되는 OOD 상태가 있어, 이 공통 상태 비교만으로 baseline의 평소 entropy를 판정하지 않았다. 위 표에는 별도로 측정한 각자 replay 결과를 사용했다.

네 OptiQ 모델에서 추가 noise std=.02는 action spread를 약0.1%만 변화시킨다. .1에서는 약2%, .2에서는 약6~7%, .35에서는 약15~18% 늘어난다. 이 역시 frozen-policy forward 진단이며 새 궤적이나 성공률 개선 결과가 아니다. .35에서는 clip으로 ±1 경계에 놓이는 좌표가 약13~15%이므로, 크게 넣은 noise가 그대로 유효한 탐색이 된다고 볼 수 없다.

## entropy 추정의 범위와 권고

현재 DACER는 3개 full-covariance Gaussian과 상태당200samples로 `H(component)+Σw H(Gaussian)`을 계산한다. 이는 **맞춘 GMM의 marginal entropy에 대한 상계**이지 실제 bounded policy entropy의 정확한 값이 아니다. 별도 샘플링에서 joint−marginal 차이는 OptiQ 약.018~.021/dim이었다. 이론상 최대 ln3/8≈.137/dim와 별도로 fitting 오차가 존재한다. fitted GMM에서 나온 행동 중 약37~39%는 8좌표 중 적어도 하나가 [-1,1] 밖에 있었다. 실제 policy 샘플은 모두 범위 안이다. 따라서 +.65는 **이 추정기 스케일에서의 제어 후보**이며, 실제 entropy를 uniform의 ln2에 맞춘다는 뜻이 아니다.

다음 판단은 학습 중 실제 extra-noise std, conditional sigma, clip fraction, critic 상태와 함께 공간 coverage·분기점 양쪽 방문·같은 시작상태에서의 성공 경로 비율을 같이 보아야 한다. 큰 action entropy가 시간적으로 일관된 두 경로를 보장하지 않는다. 현재 OptiQ는 plain TD를 유지하고 DACER는 behavior-only 조절이므로 SAC/MFPO의 MaxEnt 목적함수와도 다르다. 완료된 v3 모델의 높은 action spread와 경로 편중이 함께 나타난 점은 단순한 '행동 잡음 부족'으로 설명하기 어렵다.

권고는 (1) 현재 OFF 대조군과 기존 ON을 비교해 실제 영향 확인, (2) 이후 DACER를 다시 쓸 경우 +.60/.65 후보와 **갱신 간격**을 별도 요인으로 평가, (3) 목표값만 크게 올려 해결하려 하지 않는 것이다. 갱신 간격·noise scale·sigma 상한은 이번 실행에서 바꾸지 않았다.

근거: 보존 로그27개, SHA256 검증한 완료 체크포인트7개, 각 모델 24상태의 CPU forward sampling. 모든 결과는 training seed0 하나이며 상태 표본이 작다. 코드 기준은 [DACER 원문 Eq.15/16](https://proceedings.neurips.cc/paper_files/paper/2024/file/6174c67b136621f3f2e4a6b1d3286f6b-Paper-Conference.pdf), [DDiffPG 공식 SAC](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/sac.py), 현재 보존 source의 regulator.py 및 native baseline config다. 논문 공식 환경별 최적 entropy를 찾았다는 의미는 아니다.

![로그와 baseline 분석](entropy_noise_audit.png)

![고정 정책 noise 진단](frozen_policy_noise_grid.png)
'''
(ROOT/'REPORT_KO.md').write_text(report)
print(json.dumps(dict(report=str(ROOT/'REPORT_KO.md'),plot=str(ROOT/'entropy_noise_audit.png'),
    controls={m:{k:v[k] for k in ('final_entropy_per_dim','final_noise_std')} for m,v in controls.items()}),indent=2))
