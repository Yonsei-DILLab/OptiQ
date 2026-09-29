"""Replot saved SQL GMM40 outputs only; never retrain or resample a policy."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/gmm40_5090_queue/results/results'
OUT=Path(__file__).resolve().parent
metadata=json.loads((DATA/'target/definition.json').read_text())
means=np.asarray(metadata['means']);std=np.asarray(metadata['std'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,10.5))
records=[];histories=[];hashes={}
for seed,ax in enumerate(axes.flat):
 folder=DATA/f'sql_s{seed}_100k';ev=folder/'evaluations/step_0100000'
 config=json.loads((folder/'config.json').read_text())
 audit=json.loads((folder/'update_count_audit.json').read_text())
 m=json.loads((ev/'metrics.json').read_text());x=np.load(ev/'samples.npy',allow_pickle=False)
 assert config['method']=='sql' and config['seed']==seed and m['step']==100000
 assert audit['status']=='passed' and audit['actor_updates']==100000 and audit['full_budget_completed']
 assert x.shape==(10000,2) and np.isfinite(x).all()
 d=(((x[:,None,:].astype(float)-means)/std[None,:,None])**2).sum(-1)
 labels=d.argmin(1);near=d[np.arange(len(x)),labels]<=9
 counts=np.bincount(labels[near],minlength=40)
 np.testing.assert_allclose(near.mean(),m['high_density_fraction'])
 np.testing.assert_array_equal(counts,m['mode_counts_3sigma'])
 assert int((counts>=np.asarray(m['coverage_threshold'])).sum())==m['mode_coverage']
 for center,sigma in zip(means,std):ax.add_patch(Circle(center,3*sigma,fill=False,ls='--',lw=.7,color='#8a8e94',zorder=1))
 for mask,color in [(~near,'#ec8a45'),(near,'#1479b8')]:
  ax.scatter(x[mask,0],x[mask,1],s=2.1,c=color,alpha=.4,linewidths=0,zorder=2,rasterized=True)
 ax.scatter(means[:,0],means[:,1],marker='+',c='#151a21',s=34,linewidths=1,zorder=3)
 ax.set(xlim=(-43,43),ylim=(-43,43),aspect='equal',xlabel='Action x1',ylabel='Action x2',xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
 ax.grid(alpha=.10)
 ax.set_title(f"Seed {seed}  |  covered {m['mode_coverage']}/40\nWithin 3σ: {100*m['high_density_fraction']:.2f}%  |  MMD²: {m['mmd2']:.3f}",fontsize=12,pad=9)
 records.append(dict(seed=seed,**{k:m[k] for k in ['mode_coverage','high_density_fraction','mmd2','mode_mass_tv']}))
 h={r['step']:r for r in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines())};histories.append(h)
 for name in ['config.json','update_count_audit.json','evaluations/step_0100000/samples.npy','evaluations/step_0100000/metrics.json','metrics.jsonl']:
  f=folder/name;hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
fig.suptitle('SQL / SVGD on GMM40 — 100,000 actor updates\nJAX amortized SVGD · native generator output · 10,000 saved samples per seed',fontsize=15,y=.982)
handles=[Line2D([],[],ls='',marker='o',color='#1479b8',label='Within any target 3σ boundary',markersize=5),Line2D([],[],ls='',marker='o',color='#ec8a45',label='Outside all target 3σ boundaries',markersize=5),Line2D([],[],ls='',marker='+',color='black',label='40 target component centers',markersize=8),Line2D([],[],ls='--',color='#8a8e94',label='Target 3σ boundaries')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.024),ncol=2,frameon=False,fontsize=10)
fig.text(.5,.008,'T=1 · actor 256×2 · batch 256 · kernel particles 16 · LR 3e−4 · training seeds 0–3',ha='center',fontsize=10,color='#4e5560')
fig.tight_layout(rect=(0,.09,1,.93))
for ext in ['png','pdf']:fig.savefig(OUT/f'sql_gmm40_final_4seeds.{ext}',dpi=180)
plt.close(fig)
steps=sorted(set.intersection(*(set(h) for h in histories)))
assert steps[-1]==100000
colors=['#156d9b','#d18030','#3a927a','#9566ad']
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for ax,key,label,scale in zip(axes,['mode_coverage','high_density_fraction','mmd2'],['Covered target components / 40','Within target 3σ (%)','MMD² (lower is better)'],[1,100,1]):
 values=np.array([[h[s][key]*scale for s in steps] for h in histories]);avg=values.mean(0);sd=values.std(0,ddof=1);xx=np.array(steps)/1000
 for i,color in enumerate(colors):ax.plot(xx,values[i],lw=1,color=color,alpha=.5,label=f'Seed {i}')
 ax.plot(xx,avg,c='#18252f',lw=2,label='4-seed mean');ax.fill_between(xx,avg-sd,avg+sd,color='#697e90',alpha=.16,label='± seed SD')
 ax.set(xlabel='Actor updates (thousands)',ylabel=label,xlim=(0,100));ax.grid(alpha=.18)
 if key=='mode_coverage':ax.set_ylim(0,41);ax.axhline(40,c='gray',ls='--',lw=.8)
 if key=='high_density_fraction':ax.set_ylim(0,101)
axes[0].legend(fontsize=8,ncol=2,loc='upper left')
fig.suptitle('SQL / SVGD · GMM40 learning curves · seed 0–3 · saved evaluations',fontsize=14)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'sql_gmm40_learning_curves.{ext}',dpi=180)
plt.close(fig)
summary={k:{'mean':float(np.mean([r[k] for r in records])),'sample_sd':float(np.std([r[k] for r in records],ddof=1))} for k in ['mode_coverage','high_density_fraction','mmd2','mode_mass_tv']}
(OUT/'summary.json').write_text(json.dumps(dict(source_commit=config['source_git_commit'],evaluation_mode='native_generator_output',actor_updates=100000,samples_per_seed=10000,per_seed=records,aggregate=summary,input_sha256=hashes,retrained=False),indent=2)+'\n')
print(json.dumps(summary,indent=2))
