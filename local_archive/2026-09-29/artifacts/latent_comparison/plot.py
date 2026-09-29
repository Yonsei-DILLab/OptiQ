from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
runs={mode:[json.loads((p/f'{mode}_seed{s}'/'metrics.json').read_text()) for s in range(2)] for mode in ['fixed','fresh']}
steps=np.array([r['step'] for r in runs['fixed'][0]['rows']])
fig,axs=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
colors={'fixed':'#db8733','fresh':'#258c9f'}
for mode in runs:
 for ax,key,label in [(axs[0],'tv' if mode=='fixed' else 'fresh_z_tv','Actual policy prior'),(axs[1],'fresh_z_tv','Common evaluation: fresh Gaussian z')]:
  vals=np.array([[r[key] for r in run['rows']] for run in runs[mode]])
  for i in range(2):ax.plot(steps[1:],vals[i,1:],color=colors[mode],alpha=.3,lw=1,ls=['-','--'][i])
  ax.plot(steps[1:],vals.mean(0)[1:],color=colors[mode],lw=2.5,marker='o',ms=4,label='Fixed z64' if mode=='fixed' else 'Resampled z64')
  ax.set_title(label);ax.set_xscale('log');ax.set_xlabel('Actor updates (log scale)');ax.grid(alpha=.15);ax.axhline(.1,color='#64748b',ls=':',lw=1)
axs[0].set_ylabel('Histogram TV to target (lower is better)');axs[0].set_ylim(0,.43);axs[0].legend()
fig.suptitle('Fixed vs resampled latent | N=M=64 | T=1 | batch=32 | same initialization',fontsize=14)
fig.text(.5,.01,'Thick lines: mean of 2 seeds. Thin lines: individual seeds. Fixed training defines a finite 64-component prior; fresh training uses Gaussian z.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(p/'convergence.png',dpi=170)
summary={}
for mode in runs:
 key='tv' if mode=='fixed' else 'fresh_z_tv'
 summary[mode]=dict(first_observed_TV_below_01=[next((r['step'] for r in run['rows'] if r[key]<.1),None) for run in runs[mode]],final_primary_TV=[run['rows'][-1][key] for run in runs[mode]],final_fresh_z_TV=[run['rows'][-1]['fresh_z_tv'] for run in runs[mode]],mean_TV_by_step={int(step):float(np.mean([run['rows'][i][key] for run in runs[mode]])) for i,step in enumerate(steps)},source_commit=runs[mode][0]['source_commit'])
 assert all(run['rows'][-1]['step']==100000 for run in runs[mode])
 for run in runs[mode]:assert all(np.isfinite(r[key]) for r in run['rows'])
(p/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
