"""Four final 256x2 reproduction profiles; initial sigma labeled per panel."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=Path(__file__).resolve().parent;theta=np.linspace(0,2*np.pi,100);rows=[]
fig,axs=plt.subplots(2,2,figsize=(11,12.3))
for ax,cap in zip(axs.flat,(-3.,-3.5,-4.,-4.5)):
 label=str(-cap).replace('.','p');b=r/f'gmm40-mu90-repro-256x2-capm{label}-20260921'/'results';d=next(b.glob('*nm64-s0'));p=d/'evaluations/step_0100000';c=json.loads((d/'config.json').read_text());a=json.loads((d/'update_count_audit.json').read_text());assert a['status']=='passed' and a['actor_updates']==100000 and a['full_budget_completed']
 assert c['width']==256 and c['depth']==2 and c['n']==c['m']==64 and c['latent_mode']=='random'
 t=json.loads((b/'target/definition.json').read_text());means=np.array(t['means']);std=np.array(t['std']);m=json.loads((p/'metrics_mu_only.json').read_text());f=json.loads((p/'metrics.json').read_text());x=np.load(p/'samples_mu_only.npy');near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9);assert np.isclose(near.mean(),m['high_density_fraction'])
 for mask,color in [(near,'#2381b4'),(~near,'#e9884f')]:ax.scatter(*x[mask].T,s=1.3,alpha=.4,c=color,rasterized=True)
 for center,s in zip(means,std):ax.plot(center[0]+3*s*np.cos(theta),center[1]+3*s*np.sin(theta),'--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e');ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂');ax.grid(alpha=.1)
 ax.set_title(f"log σ cap {cap:g} · initial {c['trg_initial_log_std']:g}\nnear {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.5f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=11)
 rows.append(dict(cap=cap,initial=c['trg_initial_log_std'],width=256,depth=2,near=m['high_density_fraction'],coverage=m['mode_coverage'],mmd2=m['mmd2'],mass_tv=m['mode_mass_tv'],full_policy_near=f['high_density_fraction'],full_policy_coverage=f['mode_coverage'],source_commit=c['source_git_commit'],run=str(d)))
fig.suptitle('256×2 reproduction · N=M64 · random latent · μ only',fontsize=17,y=.98)
fig.text(.5,.945,'Mean-head scale 16 · teacher floor .05 · batch 256 · T=1 · seed 0 · 100k updates',ha='center',fontsize=10)
fig.subplots_adjust(top=.875,bottom=.09,wspace=.22,hspace=.36,left=.065,right=.985)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
fig.savefig(r/'reproduction_256x2.png',dpi=160);fig.savefig(r/'reproduction_256x2.pdf')
(r/'reproduction_256x2_summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
