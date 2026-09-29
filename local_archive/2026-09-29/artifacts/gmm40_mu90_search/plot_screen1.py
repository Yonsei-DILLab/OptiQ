from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=Path(__file__).resolve().parent;b=r/'screen1/results'
t=json.loads((b/'target/definition.json').read_text());means=np.asarray(t['means']);std=np.asarray(t['std']);theta=np.linspace(0,2*np.pi,100)
fig,axs=plt.subplots(2,2,figsize=(11,12))
for ax,(init,n) in zip(axs.flat,[(16,64),(16,256),(256,64),(256,256)]):
 d=b/f'gmm40-mu90-screen1-20260921-init{init}_nm{n}-s0';p=d/'evaluations/step_0100000';x=np.load(p/'samples_mu_only.npy');m=json.loads((p/'metrics_mu_only.json').read_text());near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
 assert np.isclose(near.mean(),m['high_density_fraction'])
 for mask,c in [(near,'#2381b4'),(~near,'#e9884f')]:ax.scatter(*x[mask].T,s=1.2,alpha=.4,c=c,rasterized=True)
 for c,s in zip(means,std):ax.plot(c[0]+3*s*np.cos(theta),c[1]+3*s*np.sin(theta),'--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e');ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂');ax.grid(alpha=.1)
 ax.set_title(f"Mean-head scale {init} · N=M={n}\nnear {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.5f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=11)
fig.suptitle('GMM40 · 256×2 · random latent · μ only',fontsize=18,y=.98)
fig.text(.5,.945,'100k updates · seed 0 · log σ=[−5,−2] · initial −2.5 · teacher floor .05',ha='center',fontsize=10)
fig.subplots_adjust(top=.88,bottom=.10,wspace=.19,hspace=.36,left=.07,right=.985)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
fig.savefig(r/'screen1_mu_only.png',dpi=150);fig.savefig(r/'screen1_mu_only.pdf')
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
for init,n in [(16,64),(16,256),(256,64),(256,256)]:
 d=b/f'gmm40-mu90-screen1-20260921-init{init}_nm{n}-s0';rows=[]
 for p in sorted(d.glob('evaluations/*/metrics_mu_only.json')):
  m=json.loads(p.read_text());rows.append((int(p.parent.name[5:]),m['high_density_fraction']*100,m['mode_coverage']))
 arr=np.asarray(rows);label=f'init {init}, N=M {n}'
 for i,ax in enumerate(axs):ax.plot(arr[:,0]/1000,arr[:,i+1],label=label)
axs[0].axhline(90,color='k',ls='--',lw=1);axs[1].axhline(40,color='k',ls='--',lw=1)
for ax,label in zip(axs,['Near GT 3σ (%)','Covered modes']):ax.set(xlabel='Actor updates (k)',ylabel=label);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('256×2 screening · μ-only evaluation · seed 0');fig.tight_layout();fig.savefig(r/'screen1_curves.png',dpi=160)
