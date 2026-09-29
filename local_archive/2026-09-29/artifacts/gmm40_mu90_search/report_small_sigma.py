"""Matched cap comparison; refuses partial budgets and mismatched sample labels."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=Path(__file__).resolve().parent;rows=[];series={};theta=np.linspace(0,2*np.pi,100)
fig,axs=plt.subplots(2,3,figsize=(16,11.9))
for i,width in enumerate((256,512)):
 for j,cap in enumerate((-3.5,-4.,-4.5)):
  label=str(-cap).replace('.','p');campaign=f'gmm40-mu90-smallsigma-{width}x3-capm{label}-20260921';b=r/campaign/'results';d=next(b.glob('*nm64-s0'));p=d/'evaluations/step_0100000'
  cfg=json.loads((d/'config.json').read_text());a=json.loads((d/'update_count_audit.json').read_text());assert a['status']=='passed' and a['actor_updates']==100000 and a['full_budget_completed']
  assert cfg['n']==cfg['m']==64 and cfg['latent_mode']=='random' and cfg['trg_initial_log_std']==-4.75
  t=json.loads((b/'target/definition.json').read_text());means=np.asarray(t['means']);std=np.asarray(t['std']);m=json.loads((p/'metrics_mu_only.json').read_text());full=json.loads((p/'metrics.json').read_text());x=np.load(p/'samples_mu_only.npy');near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
  assert len(x)==10000 and np.isclose(near.mean(),m['high_density_fraction'])
  row=dict(width=width,depth=3,cap=cap,initial=-4.75,seed=0,near=m['high_density_fraction'],coverage=m['mode_coverage'],mmd2=m['mmd2'],mass_tv=m['mode_mass_tv'],full_policy_near=full['high_density_fraction'],full_policy_coverage=full['mode_coverage'],source_commit=cfg['source_git_commit'],run=str(d));rows.append(row)
  ax=axs[i,j]
  for mask,c in [(near,'#2381b4'),(~near,'#e9884f')]:ax.scatter(*x[mask].T,s=1.1,alpha=.4,c=c,rasterized=True)
  for c,s in zip(means,std):ax.plot(c[0]+3*s*np.cos(theta),c[1]+3*s*np.sin(theta),'--',lw=.5,c='#838b94')
  ax.scatter(*means.T,marker='+',s=20,c='#17242e');ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂');ax.grid(alpha=.1)
  ax.set_title(f"{width}×3 · log σ cap {cap:g}\nnear {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.4f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=10.5)
  vals=[]
  for f in sorted(d.glob('evaluations/*/metrics_mu_only.json')):
   z=json.loads(f.read_text());vals.append((int(f.parent.name[5:]),z['high_density_fraction']*100,z['mode_coverage']))
  series[(width,cap)]=np.asarray(vals)
fig.suptitle('Smaller conditional σ · matched initialization · μ-only GMM40',fontsize=19,y=.98)
fig.text(.5,.945,'N=M=64 · initial log σ=−4.75 · mean-head scale 16 · teacher floor .05 · seed 0 · 100k updates',ha='center',fontsize=11)
fig.subplots_adjust(top=.87,bottom=.09,wspace=.20,hspace=.36,left=.05,right=.98)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.018),ncol=2,frameon=False)
fig.savefig(r/'small_sigma_comparison.png',dpi=160);fig.savefig(r/'small_sigma_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,8))
for i,width in enumerate((256,512)):
 for cap,color in zip((-3.5,-4.,-4.5),('#2575ad','#d38826','#7c51a0')):
  a=series[(width,cap)]
  for j in range(2):axs[i,j].plot(a[:,0]/1000,a[:,j+1],color=color,label=f'cap {cap:g}')
 for j in range(2):
  ax=axs[i,j];ax.axhline(90 if j==0 else 40,c='k',ls='--',lw=.8);ax.grid(alpha=.2);ax.set(xlabel='Actor updates (k)',ylabel='Near (%)' if j==0 else 'Covered modes',title=f'{width}×3');ax.legend(fontsize=8)
fig.suptitle('Random latent · N=M64 · μ-only metrics · seed 0');fig.tight_layout();fig.savefig(r/'small_sigma_curves.png',dpi=160)
(r/'small_sigma_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
