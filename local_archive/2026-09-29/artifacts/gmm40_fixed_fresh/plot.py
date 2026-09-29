from pathlib import Path
import json,numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
p=Path(__file__).parent;d=p/'results';t=json.loads((p/'target.json').read_text());means=np.array(t['means']);std=np.array(t['std'])
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axs=plt.subplots(2,2,figsize=(11,11),sharex=True,sharey=True)
summary={}
for i,mode in enumerate(['fixed','fresh']):
 summary[mode]=[]
 for seed in range(2):
  folder=d/f'{mode}_seed{seed}';r=json.loads((folder/'metrics.json').read_text());last=r['rows'][-1];assert last['step']==100000
  x=np.load(folder/'step100000.npz')['samples'][:10000];dist=cdist(x,means,'sqeuclidean')/std[None,:]**2;near=dist.min(1)<=9;ax=axs[i,seed]
  ax.scatter(*x[~near].T,s=2,color='#ee873e',alpha=.28,rasterized=True);ax.scatter(*x[near].T,s=2,color='#1683b4',alpha=.38,rasterized=True)
  for mu,s in zip(means,std):ax.add_patch(Circle(mu,3*s,fill=False,ls='--',lw=.7,color='#6e7984',alpha=.8))
  ax.scatter(*means.T,marker='+',s=50,color='#1a2531',lw=1.2)
  label='Fixed z64' if mode=='fixed' else 'Fresh Gaussian z64 / update'
  ax.set_title(f"{label} | seed {seed}\nCoverage {last['coverage']}/40 | near {last['near_fraction']:.1%}\nMMD² {last['mmd2']:.4f} | mode-mass TV {last['mode_mass_tv']:.3f}")
  ax.set(xlim=(-44,44),ylim=(-44,44),aspect='equal',xlabel='Action x1',ylabel='Action x2');ax.grid(alpha=.12)
  summary[mode].append({k:last[k] for k in ['coverage','near_fraction','mmd2','mode_mass_tv','train_seconds']})
fig.suptitle('GMM40 | Direct GMM | 64 x 64 | 100,000 updates\nT=1 | batch=256 | mean-head init scale=1 | latent skip=0',fontsize=17,y=.98)
legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#1683b4',label='Inside any target 3σ'),Line2D([0],[0],marker='o',color='w',markerfacecolor='#ee873e',label='Outside all target 3σ'),Line2D([0],[0],marker='+',color='#1a2531',ls='',label='Target centers')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.024),ncol=3,frameon=False)
fig.text(.5,.01,'10,000 unfiltered policy samples per panel; metrics use 32,768. Coverage counts target components. Mode-mass TV conditions on near samples.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.065,1,.925]);fig.savefig(p/'comparison_100k.png',dpi=170);fig.savefig(p/'comparison_100k.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4.2));colors={'fixed':'#d98b32','fresh':'#258ba1'}
for mode in ['fixed','fresh']:
 runs=[json.loads((d/f'{mode}_seed{s}'/'metrics.json').read_text())['rows'] for s in range(2)]
 steps=np.array([r['step'] for r in runs[0]])
 for ax,key,title in zip(axs,['coverage','near_fraction','mmd2'],['Covered components /40','Fraction inside target 3σ','MMD² (lower is better)']):
  vals=np.array([[r[key] for r in run] for run in runs])
  for v in vals:ax.plot(steps/1000,v,color=colors[mode],alpha=.3,lw=1)
  ax.plot(steps/1000,vals.mean(0),marker='o',ms=4,lw=2,color=colors[mode],label='Fixed z64' if mode=='fixed' else 'Fresh z64')
  ax.set(title=title,xlabel='Updates (thousands)');ax.grid(alpha=.15)
axs[0].set_ylim(0,41);axs[1].set_ylim(0,1);axs[0].legend();fig.suptitle('GMM40 training progress | thick lines: mean of 2 seeds',fontsize=15)
fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(p/'training_progress.png',dpi=170);plt.close(fig)
(p/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
