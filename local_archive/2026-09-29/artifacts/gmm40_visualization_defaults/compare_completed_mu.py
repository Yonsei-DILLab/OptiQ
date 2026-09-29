"""Primary mu-only comparisons from separately verified reports and saved samples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'gmm40_mu_primary_comparison';OUT.mkdir(exist_ok=True)
SPECS=[('gmm40_5090_queue','Mean 1e-4 / cap -1'),('gmm40_capm1_mean1_queue','Mean 1 / cap -1'),
       ('gmm40_capm3_queue','Mean 1 / cap -3'),('gmm40_oldinit_fresh_queue','Old-init profile / cap +1')]
groups=[];summary={}
for name,label in SPECS:
 root=BASE/name;report=root/'visualization-mu-primary-17cfcc1'
 assert (report/'LOCAL_COPY_VERIFIED.json').exists()
 data=json.loads((report/'results.json').read_text());rows=[r for r in data['per_seed'] if r['method']=='optiq_trg']
 assert len(rows)==4 and all(r['visualization_mode']=='mu_only' and r['updates']==100000 for r in rows)
 summary[label]=data['aggregate']['optiq_trg']
 runs=[]
 for p in sorted((root/'results/results').glob('*optiq_trg_s*_100k')):
  cfg=json.loads((p/'config.json').read_text());assert cfg['latent_mode']=='random'
  history=[]
  for h in sorted((report/'inputs'/p.name).glob('step_*/metrics_mu_only.json')):
   history.append(dict(json.loads(h.read_text()),step=int(h.parent.name.split('_')[1])))
  assert len(history)==16 and history[-1]['step']==100000
  runs.append(dict(root=p,config=cfg,history=history,latest=history[-1]))
 groups.append(runs)
(OUT/'summary.json').write_text(json.dumps(dict(view='mu_only',latent='random normal as trained',report_source='17cfcc1ea82618e838fb692d254cbfceeefdfa9b',groups=summary),indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#62707f','#2381b4','#d68a38','#a34478']
metrics=[('mode_coverage','Coverage /40'),('high_density_fraction','Within GT 3σ'),('mmd2','MMD² ↓'),('mode_mass_tv','Mass TV ↓')]
fig,axes=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
for ax,(key,label) in zip(axes.flat,metrics):
 for runs,(_,name),color in zip(groups,SPECS,colors):
  steps=np.array([r['step'] for r in runs[0]['history']]);ys=[]
  for run in runs:
   np.testing.assert_array_equal(steps,[r['step'] for r in run['history']]);ys.append([r[key] for r in run['history']])
  ys=np.array(ys);mu=ys.mean(0);sd=ys.std(0,ddof=1)
  ax.plot(steps/1000,mu,color=color,label=name);ax.fill_between(steps/1000,mu-sd,mu+sd,color=color,alpha=.11)
 ax.set(xlabel='Actor updates (thousands)',ylabel=label);ax.grid(alpha=.15)
 if key=='mmd2':ax.set_yscale('log')
axes[0,0].legend(fontsize=9)
fig.suptitle('OptiQ GMM40 · μ only · random latent · four seeds, mean ± sample SD',fontsize=15)
for ext in ('png','pdf'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=160)
plt.close(fig)

target=json.loads((BASE/'gmm40_oldinit_fresh_queue/results/results/target/definition.json').read_text())
means=np.array(target['means']);std=np.array(target['std']);theta=np.linspace(0,2*np.pi,90)
fig,axes=plt.subplots(2,2,figsize=(12.5,12.8))
for ax,run in zip(axes.flat,groups[-1]):
 samples=np.load(run['root']/'evaluations/step_0100000/samples_mu_only.npy');assert samples.shape==(10000,2)
 near=(((samples[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9
 m=run['latest'];assert np.isclose(near.mean(),m['high_density_fraction'])
 for mask,color in ((~near,'#e98b4d'),(near,'#2381b4')):ax.scatter(*samples[mask].T,s=1.2,c=color,alpha=.42,rasterized=True)
 for center,sigma in zip(means,std):ax.plot(center[0]+3*sigma*np.cos(theta),center[1]+3*sigma*np.sin(theta),ls='--',lw=.65,c='gray')
 ax.scatter(*means.T,marker='+',c='#192630',s=23,lw=1)
 ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂',title=f"Seed {run['config']['seed']} · coverage {m['mode_coverage']}/40 · near {near.mean():.2%}\nMMD² {m['mmd2']:.4f} · mass TV {m['mode_mass_tv']:.3f}");ax.grid(alpha=.12)
fig.suptitle('Old-init profile on current TRG · 100k updates · μ-only output',fontsize=17,y=.99)
fig.text(.5,.956,'Fresh normal latent · no conditional σ noise · 10,000 saved samples per panel',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.07,1,.93))
fig.text(.5,.036,'Blue: within any GT 3σ   |   Orange: outside all GT 3σ   |   +: GT centers',ha='center',fontsize=10)
fig.text(.5,.015,'log σ [−5,+1] · initial σ=0.5 · teacher floor=0.05 · mean init=1 · N=M=64',ha='center',fontsize=10)
for ext in ('png','pdf'):fig.savefig(OUT/f'oldinit_four_seeds.{ext}',dpi=155)
plt.close(fig)
print(OUT)
