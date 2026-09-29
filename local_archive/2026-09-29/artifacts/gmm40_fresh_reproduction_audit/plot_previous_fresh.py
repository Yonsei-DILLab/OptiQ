import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1]/'gmm40_fixed_fresh'
out=Path(__file__).resolve().parent
t=json.loads((root/'target.json').read_text());means=np.asarray(t['means']);std=np.asarray(t['std']);theta=np.linspace(0,2*np.pi,100)
fig,axs=plt.subplots(1,2,figsize=(12.5,7.2))
for seed,ax in enumerate(axs):
 d=root/'results'/f'fresh_seed{seed}'
 m=json.loads((d/'metrics.json').read_text());assert m['manifest']['mode']=='fresh'
 row=m['rows'][-1];x=np.load(d/'step100000.npz')['samples']
 near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
 assert np.isclose(near.mean(),row['near_fraction'])
 for sel,c in [(~near[:10000],'#e9884f'),(near[:10000],'#2381b4')]:ax.scatter(*x[:10000][sel].T,s=1.6,alpha=.5,c=c,rasterized=True)
 for center,s in zip(means,std):ax.plot(center[0]+3*s*np.cos(theta),center[1]+3*s*np.sin(theta),ls='--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e')
 ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂')
 ax.grid(alpha=.12)
 ax.set_title(f"Fresh Gaussian z · seed {seed} · 100k\nnear {near.mean():.2%} · coverage {row['coverage']}/40\nMMD² {row['mmd2']:.5f} · mass TV {row['mode_mass_tv']:.4f}",fontsize=12)
fig.suptitle('Previous GMM40 Direct GMM: NON-FIXED latent',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.918,'Train and evaluate with fresh z ~ N(0,I) · N=M=64 · batch=256 · T=1 · 256×2 · mean init=1',ha='center',fontsize=11)
fig.subplots_adjust(top=.79,bottom=.18,left=.065,right=.985,wspace=.18)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within any GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside all GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
fig.text(.5,.035,'Historical tanh-Gaussian policy. All-sample metrics: 32,768 draws; first 10,000 unfiltered draws shown.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'previous_fresh_only_100k.{ext}',dpi=170)
print(out/'previous_fresh_only_100k.png')
