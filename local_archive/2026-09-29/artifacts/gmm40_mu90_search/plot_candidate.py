from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=Path(__file__).resolve().parent
b=r/'gmm40-mu90-narrow64-256x3-capm3p5-20260921/results';target=json.loads((b/'target/definition.json').read_text());means=np.asarray(target['means']);std=np.asarray(target['std']);theta=np.linspace(0,2*np.pi,100)
old=r/'gmm40-mu90-256x3-20260921/results/gmm40-mu90-256x3-20260921-init16_nm64-s0/evaluations/step_0100000'
new=b/'gmm40-mu90-narrow64-256x3-capm3p5-20260921-init16_nm64-s0/evaluations/step_0100000'
fig,axs=plt.subplots(1,2,figsize=(13,7.6))
for ax,p,title in zip(axs,[old,new],['log σ cap −2 · initial −2.5','log σ cap −3.5 · initial −4']):
 m=json.loads((p/'metrics_mu_only.json').read_text());x=np.load(p/'samples_mu_only.npy');near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
 assert np.isclose(near.mean(),m['high_density_fraction'])
 for mask,c in [(near,'#2381b4'),(~near,'#e9884f')]:ax.scatter(*x[mask].T,s=1.5,alpha=.45,c=c,rasterized=True)
 for c,s in zip(means,std):ax.plot(c[0]+3*s*np.cos(theta),c[1]+3*s*np.sin(theta),'--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e');ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂');ax.grid(alpha=.1)
 ax.set_title(f"{title}\nnear {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.5f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=12)
fig.suptitle('GMM40 · N=M=64 · random latent · μ-only evaluation',fontsize=18,y=.97)
fig.text(.5,.922,'256×3 GELU · mean-head scale 16 · batch 256 · T=1 · 100k updates · seed 0',ha='center',fontsize=11)
fig.subplots_adjust(top=.80,bottom=.12,wspace=.21,left=.06,right=.985)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.016),ncol=2,frameon=False)
fig.savefig(r/'candidate_nm64_mu_comparison.png',dpi=160);fig.savefig(r/'candidate_nm64_mu_comparison.pdf')
print('saved comparison')
