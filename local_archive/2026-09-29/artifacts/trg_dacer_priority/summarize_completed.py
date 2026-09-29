"""Recompute final-window returns from verified local evaluation archives."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'summary';OUT.mkdir(exist_ok=True)
summary={};curves={};rows=[]
for task in ('ant','humanoid'):
    root=ROOT/task;assert (root/'LOCAL_COPY_VERIFIED.json').exists()
    result=json.loads((root/f'result-{task}.json').read_text())
    jobs=sorted([j for j in result['jobs'] if j['stage']=='dacer'],key=lambda j:j['seed'])
    assert len(jobs)==4
    curves[task]={};summary[task]={}
    for mode in ('stochastic_z','zero_z'):
        perseed=[];ys=[];steps=None
        for j in jobs:
            folder=root/'runs'/j['name'];cfg=json.loads((folder/'config.json').read_text())
            assert cfg['alg']['actor']['mean_output_init_scale']==1e-4 and cfg['runtime']['git_commit']=='efe67fde083513b89f57a34a54743cb1e6d4b49d'
            with np.load(folder/f'evaluations_{mode}.npz') as z:
                x=z['timesteps'];y=z['results'];assert x[-1]==1000000 and y.shape==(len(x),10) and np.isfinite(y).all()
                window=(x>900000)&(x<=1000000);assert window.sum()==20
                score=float(y[window].mean());np.testing.assert_allclose(score,j['metrics'][mode+'_last_100k_mean'])
                if steps is None:steps=x
                else:np.testing.assert_array_equal(steps,x)
                ys.append(y.mean(1));perseed.append(score)
                rows.append(dict(task=task,mode=mode,seed=j['seed'],last100k_mean=score,wandb=j['wandb_url']))
        curves[task][mode]=(steps,np.array(ys))
        summary[task][mode]=dict(mean=float(np.mean(perseed)),sample_sd=float(np.std(perseed,ddof=1)),per_seed=perseed)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for ax,task in zip(axes,('ant','humanoid')):
    for mode,color in [('stochastic_z','#187bb6'),('zero_z','#bb7046')]:
        x,y=curves[task][mode];m=y.mean(0);sd=y.std(0,ddof=1)
        ax.plot(x/1000,m,color=color,label=mode);ax.fill_between(x/1000,m-sd,m+sd,color=color,alpha=.14)
    ax.axvspan(900,1000,color='gray',alpha=.1);ax.set(title=task.title(),xlabel='Environment steps (thousands)',ylabel='Episode return');ax.grid(alpha=.18);ax.legend()
fig.suptitle('DACER · T=0.25 · beta=1 · mean init=1e-4 · four seeds (mean ± sample SD)',fontsize=14)
for ext in ('png','pdf'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,task in zip(axes,('ant','humanoid')):
    for i,(mode,color) in enumerate([('stochastic_z','#187bb6'),('zero_z','#bb7046')]):
        s=summary[task][mode];ax.bar(i,s['mean'],yerr=s['sample_sd'],color=color,alpha=.65,capsize=5)
        ax.scatter(i+np.linspace(-.09,.09,4),s['per_seed'],s=24,color='#28323b',zorder=3)
    ax.set(xticks=[0,1],xticklabels=['stochastic_z','zero_z'],ylabel='Last 100k mean return',title=task.title());ax.grid(axis='y',alpha=.15)
fig.suptitle('900k < step ≤ 1M · 20 evaluations × 10 episodes per seed · mean ± seed SD')
for ext in ('png','pdf'):fig.savefig(OUT/f'last100k.{ext}',dpi=160)
plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 기존 DACER 8개 최종 결과','', 'Ant/Humanoid 각각 seed0~3, 1M step. T=.25, beta=1, DACER=true, mean init=1e-4. 새 mean-init=1 캠페인과 구분합니다.',
'마지막100k(900000<step<=1000000)의 20회 평가×10episode를 시드 내 평균 후 4시드 평균. ±는 sample SD(ddof=1).', '',
'| 환경 | stochastic_z | zero_z |','|---|---:|---:|']
for task in ('ant','humanoid'):
    cells=[task.title()]+[f"{summary[task][mode]['mean']:.1f} ± {summary[task][mode]['sample_sd']:.1f}" for mode in ('stochastic_z','zero_z')]
    lines.append('| '+' | '.join(cells)+' |')
lines += ['', '학습곡선은 각 평가의 10episode 평균을 시드 간 평균한 값이며 음영은 시드 간 SD입니다. 두 평가 모두 mu-only이며 stochastic_z는 행동마다 normal latent를 뽑고 DACER 잡음은 평가에 더하지 않습니다.',
'기존 temperature 결과와는 T도 달라 순수 DACER 효과를 분리할 수 없습니다. Beta 취소 작업은 완료 수에 포함하지 않았습니다.', '', '![학습곡선](learning_curves.png)','![마지막100k](last100k.png)']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
