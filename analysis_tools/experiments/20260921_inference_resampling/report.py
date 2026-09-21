from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
r=p.parse_args().root
manifest=json.loads((r/'manifest.json').read_text())
modes=['mu_one','mu_q64','mu_kde_is64']
labels=['One μ','Q-resample\n64 μ','KDE-corrected\n64 μ (approx.)']
all_results=[]
for job in manifest['jobs']:
    d=json.loads((r/'results'/(job['name']+'.json')).read_text())
    assert d['complete'] and d['checkpoint_state_unchanged'] and d['preflight_passed']
    assert set(d['modes'])==set(modes)
    assert d['job']['input_sha256']==job['input_sha256']
    for m in modes:
        x=d['modes'][m]
        assert len(x['returns'])==20 and np.isfinite(x['returns']).all()
        assert x['reset_seeds']==d['modes']['mu_one']['reset_seeds']
    all_results.append(d)
fig,axs=plt.subplots(1,2,figsize=(11,5))
summary={};tables=[]
for ax,task in zip(axs,['halfcheetah','ant']):
    runs=sorted([d for d in all_results if d['job']['task']==task],key=lambda d:d['job']['seed'])
    values=np.array([[d['modes'][m]['mean_return'] for m in modes] for d in runs])
    x=np.arange(3)
    for d,vals in zip(runs,values):
        ax.plot(x,vals,'o-',alpha=.45,lw=1,label=f'seed {d["job"]["seed"]}')
    ax.errorbar(x,values.mean(0),yerr=values.std(0,ddof=1),fmt='o-',c='#122e47',lw=2.5,capsize=5,label='Mean ± seed SD')
    ax.set_xticks(x,labels);ax.set_ylabel('Episode return');ax.grid(axis='y',alpha=.2)
    ax.set_title(f'{task.title()} · {len(runs)} training seeds × 20 episodes')
    ax.legend(fontsize=8,loc='best')
    details={}
    for j,m in enumerate(modes):
        delta=values[:,j]-values[:,0]
        details[m]=dict(mean=float(values[:,j].mean()),seed_sd=float(values[:,j].std(ddof=1)),
            per_seed=values[:,j].tolist(),mean_delta=float(delta.mean()),delta_seed_sd=float(delta.std(ddof=1)),
            per_seed_delta=delta.tolist(),improved_seed_count=int((delta>0).sum()),
            mean_ess=float(np.mean([d['modes'][m]['mean_ess'] for d in runs])),
            mean_max_weight=float(np.mean([d['modes'][m]['mean_max_weight'] for d in runs])),
            mean_selected_q_gain=float(np.mean([d['modes'][m]['mean_selected_q_gain'] for d in runs])))
    summary[task]=dict(training_seeds=[d['job']['seed'] for d in runs],modes=details)
fig.suptitle('Frozen 1M checkpoints · T=0.25 · DACER-trained · μ-only evaluation',fontsize=14)
fig.text(.5,.015,'Paired fresh reset seeds · no σ / DACER action noise · KDE uses 256 independent μ pilot samples',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.94))
for ext in ['png','pdf']:fig.savefig(r/f'comparison.{ext}',dpi=180)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
deltafig,daxs=plt.subplots(1,2,figsize=(10,4.7))
for ax,task in zip(daxs,['halfcheetah','ant']):
    d=summary[task]['modes']
    for x,m,color in [(0,'mu_q64','#247ba0'),(1,'mu_kde_is64','#e4844a')]:
        vals=np.array(d[m]['per_seed_delta'])
        ax.scatter(x+np.linspace(-.07,.07,len(vals)),vals,s=40,color=color,zorder=3)
        ax.errorbar([x],[vals.mean()],yerr=[vals.std(ddof=1)],fmt='D',c='#1a2634',capsize=6,ms=6,zorder=4)
    ax.axhline(0,c='gray',lw=1,ls='--')
    ax.set(xticks=[0,1],xticklabels=['Q-resample 64 μ','KDE-corrected 64 μ'],ylabel='Return change from one μ',xlim=(-.5,1.5))
    ax.set_title(f'{task.title()} · {len(summary[task]["training_seeds"])} training seeds')
    ax.grid(axis='y',alpha=.2)
deltafig.suptitle('Paired reward improvement · final 1M checkpoint · T=0.25')
deltafig.text(.5,.015,'Colored dots: each training seed · black diamond: mean ± seed SD · 20 paired episodes per seed',ha='center',fontsize=9)
deltafig.tight_layout(rect=(0,.055,1,.93))
for ext in ['png','pdf']:deltafig.savefig(r/f'paired_improvement.{ext}',dpi=180)
lines=['완료된 T=.25, beta=1, DACER=true의 최종 1M actor와 critic을 재학습 없이 비교했다. HalfCheetah 5개 학습 seed, Ant 4개 학습 seed마다 동일한 새 reset/policy seed 20개로 평가했다. 기존 마지막100k 평균과는 별도의 최종 체크포인트 재평가다.\n',
'| 환경 | 평가 | return 평균 ± 학습 seed SD | 기존 대비 Δ | ESS / 후보 수 |',
'|---|---|---:|---:|---:|']
for task,d in summary.items():
    for m,v in d['modes'].items():
        lines.append(f'| {task} | {m} | {v["mean"]:.1f} ± {v["seed_sd"]:.1f} | {v["mean_delta"]:+.1f} | {v["mean_ess"]:.2f} / {1 if m=="mu_one" else 64} |')
lines += ['\n![결과](comparison.png)\n',
'`mu_one`: 기존 stochastic_z와 같은 μ-only sampler. `mu_q64`: μ 64개를 softmax(Q/.25)로 categorical 재샘플링. `mu_kde_is64`: 동일 후보를 softmax(Q/.25-log q_hat_mu)로 재샘플링. Q는 각 학습 설정과 동일하게 current twin critic 평균이다. 세 방식 모두 조건부 σ와 DACER 행동잡음을 넣지 않는다.\n',
'μ 분포의 정확한 밀도는 직접 계산할 수 없어, 독립 μ 256개로 diagonal Scott bandwidth의 box-normalized KDE를 추정했다. bandwidth floor는 정규화 action 단위 .001이다. 학습 σ를 μ 분포 밀도로 대신 사용하지 않았다. 이는 근사 importance resampling이며, Q-only 방식은 일반적으로 q_mu(a)exp(Q/T)에 비례하는 쪽으로 재가중한다. reward가 높아져도 실제 exp(Q/T) 분포에 가까워졌다는 증거는 아니다. KDE 추정 오차와 critic 오차에 민감하다.\n',
'기존 sample_action과 baseline 수치 일치, σ head 변경에 대한 세 방식 불변성, 모든 입력 checkpoint/config 해시 보존, 평가 전후 actor/critic 상태 불변성, finite action 및 가중치를 검증했다. 1M 체크포인트 source를 각 run별로 로드했다. 20개 episode는 반복 학습 seed 20개가 아니다.\n',
'평가 commit: `'+manifest['evaluation_commit']+'`. 실행 환경·checkpoint SHA256와 seed별 모든 return/length는 [manifest](manifest.json) 및 results/*.json에 보존했다.\n',
'| 환경 | seed | 기존 μ | Q64 | 근사 IS64 | Δ Q64 | Δ IS64 |',
'|---|---:|---:|---:|---:|---:|---:|']
for d in sorted(all_results,key=lambda d:(d['job']['task'],d['job']['seed'])):
    vals=[d['modes'][m]['mean_return'] for m in modes]
    lines.append(f'| {d["job"]["task"]} | {d["job"]["seed"]} | {vals[0]:.1f} | {vals[1]:.1f} | {vals[2]:.1f} | {vals[1]-vals[0]:+.1f} | {vals[2]-vals[0]:+.1f} |')
(r/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
