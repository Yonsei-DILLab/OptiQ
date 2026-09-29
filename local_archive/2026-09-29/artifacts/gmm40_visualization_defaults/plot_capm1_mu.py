from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
base=Path('artifacts/gmm40_capm1_mean1_queue/results/results')
t=json.loads((base/'target/definition.json').read_text());means=np.asarray(t['means']);std=np.asarray(t['std']);theta=np.linspace(0,2*np.pi,100)
fig,axs=plt.subplots(2,2,figsize=(12,13));rows=[]
for seed,ax in zip(range(4),axs.flat):
 p=base/f'capm1_mean1_optiq_trg_s{seed}_100k'/'evaluations/step_0100000'
 x=np.load(p/'samples_mu_only.npy');m=json.loads((p/'metrics_mu_only.json').read_text());rows.append(dict(seed=seed,**m))
 near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
 assert np.isclose(near.mean(),m['high_density_fraction'])
 for mask,color in ((near,'#2381b4'),(~near,'#e9884f')):ax.scatter(*x[mask].T,s=1.2,alpha=.4,c=color,rasterized=True)
 for c,s in zip(means,std):ax.plot(c[0]+3*s*np.cos(theta),c[1]+3*s*np.sin(theta),ls='--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e')
 ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂')
 ax.grid(alpha=.1)
 ax.set_title(f"Seed {seed} · 100k · μ only\nnear {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.5f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=12)
fig.suptitle('GMM40 · random latent · μ-only evaluation',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.943,'log σ=[−5,−1] · initial log σ=−1 · mean-head scale=1 · N=M=64 · batch=256',ha='center',fontsize=11)
fig.subplots_adjust(top=.88,bottom=.10,wspace=.18,hspace=.38,left=.065,right=.985)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within any GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside all GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.042),ncol=2,frameon=False)
fig.text(.5,.019,'All 10,000 saved μ outputs shown per panel. Higher near fraction does not imply better coverage.',ha='center',fontsize=10)
fig.savefig('artifacts/gmm40_visualization_defaults/capm1_mean1_mu_four_seeds.png',dpi=170)

print(json.dumps([dict(seed=r["seed"],near=r["high_density_fraction"],coverage=r["mode_coverage"],mmd2=r["mmd2"]) for r in rows],indent=2))
