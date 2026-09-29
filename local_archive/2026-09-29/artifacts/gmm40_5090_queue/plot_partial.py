"""Plot only observed evaluations, with explicit incomplete-seed labels."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);args=p.parse_args()
root=args.snapshot;data=json.loads((root/'snapshot.json').read_text())
out=root/'figures';out.mkdir(exist_ok=True)
target=json.loads((root/'target.json').read_text())
means=np.array(target['means']);std=np.array(target['std'])
methods=data['manifest']['plan']['methods']
labels=dict(optiq_trg='OptiQ Direct GMM',sac='SAC',dipo='DIPO',meow='MEOW',mfpo='MFPO',sql='SQL (JAX SVGD)')
runs={(r['job']['method'],r['job']['seed']):r for r in data['runs']}
by_step=lambda run:{row['step']:row for row in run['history']}
common=max(set.intersection(*(set(by_step(runs[(m,0)])) for m in methods)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.facecolor':'white'})
theta=np.linspace(0,2*np.pi,90)
blue='#2682b5';orange='#ea8b53'

def panel(ax,method,seed,step):
    run=runs[(method,seed)];metric=by_step(run)[step]
    samples=np.load(root/'samples'/run['job']['name']/str(step)/'samples.npy')
    assert samples.shape==(10000,2) and np.isfinite(samples).all()
    d=((samples[:,None,:]-means[None,:,:])/std[None,:,None])**2
    near=d.sum(-1).min(-1)<=9
    assert np.isclose(near.mean(),metric['high_density_fraction'])
    # All 10,000 observations are shown. No subsampling or density filtering.
    ax.scatter(*samples[~near].T,s=1.1,c=orange,alpha=.38,rasterized=True)
    ax.scatter(*samples[near].T,s=1.1,c=blue,alpha=.40,rasterized=True)
    for center,sigma in zip(means,std):
        ax.plot(center[0]+3*sigma*np.cos(theta),center[1]+3*sigma*np.sin(theta),
                color='#7e858a',ls='--',lw=.65)
    ax.scatter(*means.T,s=26,c='#182532',marker='+',lw=1.1)
    ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40],
        xlabel='Action x₁',ylabel='Action x₂')
    ax.grid(alpha=.12)
    ax.set_title(f"{labels[method]} · seed {seed} · {step//1000}k updates\n"
        f"coverage {metric['mode_coverage']}/40 · near {near.mean():.1%} · MMD² {metric['mmd2']:.4f}",fontsize=11,pad=10)

def distributions(items,title,subtitle,name):
    fig,axes=plt.subplots(2,3,figsize=(15,10.9))
    for ax,(method,seed,step) in zip(axes.flat,items):panel(ax,method,seed,step)
    fig.suptitle(title,fontsize=19,y=.98,fontweight='bold')
    fig.text(.5,.942,subtitle,ha='center',fontsize=11,color='#495866')
    handles=[Line2D([],[],marker='o',linestyle='',color=blue,label='Within any GT 3σ boundary'),
             Line2D([],[],marker='o',linestyle='',color=orange,label='Outside all GT 3σ boundaries'),
             Line2D([],[],marker='+',linestyle='',color='#182532',label='40 target component centers')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025),fontsize=10)
    fig.text(.5,.012,'All 10,000 IID full-policy samples shown per panel · dashed circles: GT 3σ · incomplete campaign',ha='center',fontsize=10)
    fig.subplots_adjust(top=.895,bottom=.10,left=.055,right=.985,wspace=.23,hspace=.34)
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=150)
    plt.close(fig)

distributions([(m,0,common) for m in methods],f'GMM40 · all six algorithms at {common//1000}k updates',
    'Equal update count · seed 0 only · fixed Q, T=1, batch=256 · native baseline architectures',
    'equal_updates_seed0')
completed=[('optiq_trg',0,100000),('optiq_trg',1,100000),('mfpo',0,100000),
           ('sac',0,100000),('sac',1,100000),('sql',0,100000)]
assert all(runs[(m,s)]['queue']['status']=='completed' for m,s,_ in completed)
distributions(completed,'GMM40 · completed 100k results',
    'Six completed runs; four-seed comparison is pending · OptiQ: N=M=64, random latent, log σ=[−5, −1]',
    'completed_100k')

fig,axes=plt.subplots(2,2,figsize=(13.5,8.5),constrained_layout=True)
specs=[('mode_coverage','Component coverage /40 ↑'),('high_density_fraction','Within GT 3σ (%) ↑'),
       ('mmd2','MMD² ↓ (log scale)'),('mode_mass_tv','Component mass TV ↓')]
colors=plt.get_cmap('tab10').colors
for ax,(key,label) in zip(axes.flat,specs):
    for i,m in enumerate(methods):
        for seed in range(4):
            if (m,seed) not in runs:continue
            history=list(by_step(runs[(m,seed)]).values())
            history=sorted([h for h in history if h['step']>0],key=lambda h:h['step'])
            x=np.array([h['step']/1000 for h in history]);y=np.array([h[key] for h in history])
            if key=='high_density_fraction':y*=100
            ax.plot(x,y,color=colors[i],lw=1.9,ls='-' if seed==0 else '--',alpha=1 if seed==0 else .65)
            ax.scatter(x[-1:],y[-1:],s=20,color=colors[i])
    ax.set(xlabel='Actor updates (thousands)',ylabel=label,xlim=(0,101));ax.grid(alpha=.2)
    if key=='mmd2':ax.set_yscale('log')
    if key=='mode_coverage':ax.set_ylim(0,42)
    if key=='high_density_fraction':ax.set_ylim(0,102)
fig.legend([Line2D([],[],color=colors[i],lw=2) for i in range(6)],
    [labels[m] for m in methods],loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Observed learning curves · solid: seed 0, dashed: seed 1\n'
    'Each line stops at its latest saved evaluation; no extrapolation or four-seed average',fontsize=14)
for ext in ['png','pdf']:fig.savefig(out/f'learning_curves_interim.{ext}',dpi=150)
plt.close(fig)

metrics=['mode_coverage','high_density_fraction','mmd2','sliced_wasserstein2','mode_mass_tv']
rows=[]
for (m,s),run in runs.items():
    last=run['latest']
    rows.append(dict(method=m,seed=s,step=last['step'],status=run['queue']['status'],**{k:last[k] for k in metrics}))
with (out/'latest_per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
aggregate={}
for m in methods:
    selected=[r for r in rows if r['method']==m and r['status']=='completed' and r['step']==100000]
    if not selected:continue
    aggregate[m]={'n':len(selected),'seeds':[r['seed'] for r in selected],
        **{k:{'mean':float(np.mean([r[k] for r in selected])),
              'sample_sd':float(np.std([r[k] for r in selected],ddof=1)) if len(selected)>1 else None} for k in metrics}}
summary={'scope':'INTERIM, not a four-seed result','snapshot_time':data['captured_at'],
    'equal_update_comparison':common,'completed_100k':aggregate,'latest_per_seed':rows}
(out/'interim_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# GMM40 중간 결과','',f'공통 업데이트 비교: {common:,}, seed 0. 전체 24개 중 6개 완료, 4개 실행, 14개 대기인 스냅샷.',
    '100k 결과는 완료된 seed만 표시하며 4시드 평균이 아닙니다. DIPO/MEOW 학습은 계속됩니다.',
    'OptiQ: 현재 Direct GMM/TRG, N=M=64, random latent, log sigma [-5,-1].',
    '평가: 각 패널에서 저장된 실제 full-policy 샘플 10,000개 전부 표시. 파랑은 임의의 GT 3σ 내부, 주황은 모두 외부.',
    'Coverage는 3σ 내 샘플 수가 평가 코드의 임계치를 넘는 component 수입니다. 소수 샘플만 있는 모드도 포함할 수 있으므로 near와 MMD²를 함께 해석합니다.',
    '비교는 동일 actor update 수 기준이며 baseline 구조와 Q 질의량/계산량은 서로 다릅니다.','',
    '![완료된 100k](completed_100k.png)','![동일 업데이트](equal_updates_seed0.png)','![학습곡선](learning_curves_interim.png)']
(out/'REPORT_KO.md').write_text('\n\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
print('Figures:',out)
