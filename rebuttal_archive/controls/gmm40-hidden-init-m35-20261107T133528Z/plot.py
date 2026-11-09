import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'summary.json').read_text())
b=json.loads((ROOT.parent/'gmm40-no-floor-N256-20261107T123651Z'/'summary.json').read_text())
series=[('Xavier uniform',b['full_policy']['per_seed'])]
for name,title in [('orthogonal','Orthogonal (gain 1)'),('orthogonal_sqrt2','Orthogonal (gain sqrt(2))')]:
 if r['summary'].get(name,{}).get('complete_five_seeds'):
  series.append((title,[v for v in r['rows'] if v['hidden_init']==name]))
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key in zip(axs,['mmd','coverage']):
 for i,(title,rows) in enumerate(series):
  rows=sorted(rows,key=lambda r:r['seed'])
  vals=np.array([v[key if key=='mmd' or 'coverage' in v else 'mode_coverage'] for v in rows])
  ax.scatter(i+np.linspace(-.12,.12,5),vals,s=32,color=f'C{i}',zorder=3)
  ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='s',color='black',capsize=5,zorder=4)
 ax.set_xticks(range(len(series)),[s[0] for s in series],fontsize=8)
 ax.grid(axis='y',alpha=.2)
axs[0].axhspan(.021,.031,alpha=.12,color='green',label='Requested mean range')
axs[0].set_ylabel('MMD (original evaluation)'); axs[0].legend(fontsize=8)
axs[1].set(ylim=(38.5,40.5),yticks=[39,40],ylabel='Covered target components / 40')
fig.suptitle('GMM40 | seeds 0-4 | 100K updates | floor OFF | log-sigma [-5, -3.5]\nDots: individual seeds; black square and bar: mean and sample SD')
fig.savefig(ROOT/'initialization-comparison.png',dpi=180);fig.savefig(ROOT/'initialization-comparison.pdf')
plt.close(fig)
for init in r['summary']:
 if not r['summary'][init]['complete_five_seeds']: continue
 fig,axs=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
 reference=np.load(ROOT/init/'seed-0/reference.npy')
 cfg=json.loads((ROOT/init/'seed-0/config.json').read_text()); means=np.array(cfg['target']['means'])
 for ax,seed in zip(axs.flat,[None,0,1,2,3,4]):
  if seed is None: points=reference; title='Bounded target reference'
  else:
   points=np.load(ROOT/init/f'seed-{seed}'/'evaluations/step_0100000/samples_full_policy.npy')
   row=next(v for v in r['rows'] if v['hidden_init']==init and v['seed']==seed)
   title=f"Seed {seed}: {row['coverage']}/40 | MMD {row['mmd']:.5f}"
  ax.scatter(points[:,0],points[:,1],s=1,alpha=.35,rasterized=True)
  ax.scatter(means[:,0],means[:,1],marker='+',c='black',s=15)
  ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',title=title)
 fig.suptitle(f'{init} | hidden kernels only | floor OFF | log-sigma [-5, -3.5]')
 fig.savefig(ROOT/f'{init}-samples.png',dpi=180);fig.savefig(ROOT/f'{init}-samples.pdf');plt.close(fig)
