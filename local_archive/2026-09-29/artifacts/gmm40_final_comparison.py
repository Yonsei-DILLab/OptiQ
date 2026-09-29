"""Compare verified final GMM40 campaigns without modifying their source archives."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE/'gmm40_final_comparison';OUT.mkdir(exist_ok=True)
specs=[('gmm40_5090_queue','Original: mean 1e-4, cap/init -1'),
       ('gmm40_capm1_mean1_queue','Mean 1, cap/init -1'),
       ('gmm40_capm3_queue','Mean 1, cap/init -3')]
metrics=[('mode_coverage','Coverage / 40'),('high_density_fraction','Within GT 3σ'),('mmd2','MMD² (lower is better)'),('mode_mass_tv','Mass TV (lower is better)')]
roots=[BASE/name/'results' for name,_ in specs]
target=json.loads((roots[0]/'results/target/definition.json').read_text())
for root in roots:
    assert (root/'LOCAL_COPY_VERIFIED.json').exists()
    t=json.loads((root/'results/target/definition.json').read_text())
    for key in ('means','std','weights'):np.testing.assert_array_equal(t[key],target[key])
    for p in (root/'results').glob('*/update_count_audit.json'):
        a=json.loads(p.read_text());assert a['status']=='passed' and a['actor_updates']==100000 and not a['errors']
means=np.array(target['means']);std=np.array(target['std']);theta=np.linspace(0,2*np.pi,90)
colors=['#485d73','#2381b4','#dc843d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
runs=[];summary={}
for root,(_,label) in zip(roots,specs):
    selected=[]
    for folder in (root/'results').glob('*optiq_trg_s*_100k'):
        cfg=json.loads((folder/'config.json').read_text());latest=json.loads((folder/'latest.json').read_text())
        assert cfg['n']==cfg['m']==64 and cfg['batch']==256 and cfg['temperature']==1 and latest['step']==100000
        hist={r['step']:r for r in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines())}
        selected.append(dict(folder=folder,config=cfg,latest=latest,history=[hist[s] for s in sorted(hist)]))
    selected.sort(key=lambda r:r['config']['seed']);assert [r['config']['seed'] for r in selected]==list(range(4))
    runs.append(selected)
    summary[label]={k:dict(mean=float(np.mean([r['latest'][k] for r in selected])),sample_sd=float(np.std([r['latest'][k] for r in selected],ddof=1)),per_seed=[r['latest'][k] for r in selected]) for k,_ in metrics}
for seed in range(4):
    configs=[r[seed]['config'] for r in runs]
    for key in ('n','m','batch','width','depth','temperature','actor_learning_rate','latent_mode','density_beta','teacher_std_floor','coordinate_convention'):
        assert all(c[key]==configs[0][key] for c in configs),key
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(key,label) in zip(axes.flat,metrics):
    for group,(_,name),color in zip(runs,specs,colors):
        steps=np.array([r['step'] for r in group[0]['history']]);ys=[]
        for run in group:
            np.testing.assert_array_equal(steps,[r['step'] for r in run['history']]);ys.append([r[key] for r in run['history']])
        ys=np.array(ys);mean=ys.mean(0);sd=ys.std(0,ddof=1)
        ax.plot(steps/1000,mean,label=name,c=color);ax.fill_between(steps/1000,mean-sd,mean+sd,color=color,alpha=.13)
    ax.set(xlabel='Actor updates (thousands)',ylabel=label);ax.grid(alpha=.15)
    if key=='mmd2':ax.set_yscale('log')
axes[0,0].legend(fontsize=9);fig.suptitle('GMM40 · OptiQ TRG · four seeds · mean ± sample SD',fontsize=15)
for ext in ('png','pdf'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=160)
plt.close(fig)

def draw(ax,run,title):
    sample=np.load(run['folder']/'evaluations/step_0100000/samples.npy');assert sample.shape==(10000,2) and np.isfinite(sample).all()
    near=(((sample[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9
    assert np.isclose(near.mean(),run['latest']['high_density_fraction'])
    for mask,col in ((~near,'#e98e54'),(near,'#2381b4')):ax.scatter(*sample[mask].T,s=1.25,c=col,alpha=.4,rasterized=True)
    ax.scatter(*means.T,marker='+',s=24,c='#17242e',lw=1)
    for mu,sigma in zip(means,std):ax.plot(mu[0]+3*sigma*np.cos(theta),mu[1]+3*sigma*np.sin(theta),c='gray',ls='--',lw=.5)
    m=run['latest'];ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂',
        title=f"{title}\nCoverage {m['mode_coverage']}/40 · near {near.mean():.1%}\nMMD² {m['mmd2']:.4f} · TV {m['mode_mass_tv']:.3f}")
    ax.grid(alpha=.12)

for allseeds in (False,True):
    nrows=4 if allseeds else 1
    fig,axes=plt.subplots(nrows,3,figsize=(17,5.4*nrows+1.1),squeeze=False)
    for row in range(nrows):
        for col,(group,(_,name)) in enumerate(zip(runs,specs)):draw(axes[row,col],group[row],f'{name} · seed {row}')
    fig.suptitle('GMM40 · 100k updates · 10,000 full-policy samples per panel',fontsize=17,y=.995)
    fig.tight_layout(rect=(0,.055 if allseeds else .10,1,.95 if allseeds else .91))
    fig.text(.5,.018,'Blue: within any GT 3σ   |   Orange: outside all GT 3σ   |   +: GT centers; dashed: GT 3σ',ha='center')
    for ext in ('png','pdf'):fig.savefig(OUT/f"{'all_seeds' if allseeds else 'seed0'}_distributions.{ext}",dpi=155)
    plt.close(fig)

lines=['# GMM40 최종 비교','', '세 캠페인 모두 100k actor updates, seed 0~3. ±는 시드 간 sample SD(ddof=1). 원본 archive SHA256와 32개 optimizer audit를 검증했습니다.', '',
'| 설정 | Coverage /40 | Near 3σ (%) | MMD² ↓ | Mass TV ↓ |','|---|---:|---:|---:|---:|']
for _,label in specs:
    cells=[label]
    for key,_ in metrics:
        m=summary[label][key];factor=100 if key=='high_density_fraction' else 1
        cells.append(f"{m['mean']*factor:.5g} ± {m['sample_sd']*factor:.3g}")
    lines.append('| '+' | '.join(cells)+' |')
lines += ['', '상한 −1을 유지하고 mean 초기화만 1로 바꾼 설정은 평균 MMD²와 mass TV가 낮아졌으나, 4시드만으로 보편적인 개선을 단정하지 않습니다. Coverage는 거의 같습니다.',
'mean=1에서 상한·초기 log sigma를 −3으로 낮추면 near 비율은 높지만 coverage가 감소하고 MMD²가 악화됩니다. 상한과 초기값을 함께 바꿨으므로 두 효과를 분리할 수 없습니다.',
'Target 파일의 원시 checksum은 메타데이터 차이로 다르지만 실제 means/std/weights는 정확히 같습니다. 공통 학습 설정을 config로 대조했습니다.',
'', '![학습곡선](learning_curves.png)','![seed0 분포](seed0_distributions.png)','![4seed 분포](all_seeds_distributions.png)']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
