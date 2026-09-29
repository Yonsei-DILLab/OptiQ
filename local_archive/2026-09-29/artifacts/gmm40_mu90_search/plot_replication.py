from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=Path(__file__).resolve().parent
seed0=r/'gmm40-mu90-narrow64-256x3-capm3p5-20260921/results/gmm40-mu90-narrow64-256x3-capm3p5-20260921-init16_nm64-s0'
b=r/'gmm40-mu90-replication-nm64-20260921/results'
dirs=[seed0]+[b/f'mu90_rep_optiq_trg_s{s}_100k' for s in (1,2,3)]
t=json.loads((b/'target/definition.json').read_text());means=np.array(t['means']);std=np.array(t['std']);theta=np.linspace(0,2*np.pi,100);rows=[]
fig,axs=plt.subplots(2,2,figsize=(11,12.4))
for seed,(d,ax) in enumerate(zip(dirs,axs.flat)):
 p=d/'evaluations/step_0100000';a=json.loads((d/'update_count_audit.json').read_text());assert a['actor_updates']==100000 and a['status']=='passed'
 c=json.loads((d/'config.json').read_text());m=json.loads((p/'metrics_mu_only.json').read_text());x=np.load(p/'samples_mu_only.npy');near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9);assert np.isclose(near.mean(),m['high_density_fraction'])
 rows.append(dict(seed=seed,near=m['high_density_fraction'],coverage=m['mode_coverage'],mmd2=m['mmd2'],mass_tv=m['mode_mass_tv'],source_commit=c['source_git_commit']))
 for mask,color in [(near,'#2381b4'),(~near,'#e9884f')]:ax.scatter(*x[mask].T,s=1.3,alpha=.4,c=color,rasterized=True)
 for center,s in zip(means,std):ax.plot(center[0]+3*s*np.cos(theta),center[1]+3*s*np.sin(theta),'--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=25,c='#17242e');ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂');ax.grid(alpha=.1)
 ax.set_title(f"Seed {seed} · near {near.mean():.2%} · coverage {m['mode_coverage']}/40\nMMD² {m['mmd2']:.5f} · mass TV {m['mode_mass_tv']:.3f}",fontsize=11)
fig.suptitle('Training-seed replication · N=M64 · random latent · μ only',fontsize=17,y=.98)
fig.text(.5,.945,'256×3 · log σ=[−5,−3.5] · initial −4 · mean-head scale 16 · 100k updates',ha='center',fontsize=10)
fig.subplots_adjust(top=.885,bottom=.09,wspace=.22,hspace=.30,left=.065,right=.985)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False)
fig.savefig(r/'replication_four_seeds_mu.png',dpi=150);fig.savefig(r/'replication_four_seeds_mu.pdf')
summary=dict(per_seed=rows,mean={k:float(np.mean([x[k] for x in rows])) for k in ('near','coverage','mmd2','mass_tv')},seed_sd={k:float(np.std([x[k] for x in rows],ddof=1)) for k in ('near','coverage','mmd2','mass_tv')},all_seeds_pass=all(x['near']>=.9 and x['coverage']==40 for x in rows))
(r/'replication_four_seeds.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
