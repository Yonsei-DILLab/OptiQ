from pathlib import Path
import json, statistics, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
snapshot = json.loads((P/'snapshot.json').read_text())
rows = [r for r in snapshot['rows'] if r['job']['family']=='fetch' and r['job']['task']=='pick-and-place']
temperatures = [.05,.1,.25,.5,1.]
colors = ['#3366aa','#e36a20','#219878','#9663b4','#b59a27']
sac = sorted([r for r in rows if r['job']['method']=='sac'],key=lambda r:r['job']['seed'])
assert [r['job']['seed'] for r in sac] == list(range(5))
maps = [{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in sac]
shared = sorted(set.intersection(*(set(m) for m in maps)))
ys = np.array([[m[k]['success']*100 for k in shared] for m in maps])
baseline = dict(steps=shared,mean=ys.mean(0).tolist(),sd=ys.std(0,ddof=1).tolist())
fixed=[]
for step in [1000192,2000128,3000064,4000000]:
    if step in shared:
        ix=shared.index(step);fixed.append(dict(step=step,sac_mean=baseline['mean'][ix],sac_sd=baseline['sd'][ix],sac_by_seed=ys[:,ix].tolist()))
summaries=[]
fig,axs=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
for ax,seed in zip(axs,[5,6]):
    x=np.array(shared)/1e6;mean=np.array(baseline['mean']);sd=np.array(baseline['sd'])
    ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color='#697582',alpha=.12,lw=0)
    ax.plot(x,mean,color='#43505e',ls='--',lw=2,zorder=3)
    for temp,color in zip(temperatures,colors):
        r=next(r for r in rows if r['job']['method']=='ibolt' and r['job']['seed']==seed and r['job']['temperature']==temp)
        es=sorted([e for e in r['curve'] if e['episodes']==10],key=lambda e:e['steps'])
        final=[e for e in r['curve'] if e['episodes']==100 and e['steps']==5000192]
        ax.plot([e['steps']/1e6 for e in es],[e['success']*100 for e in es],color=color,lw=1.4,alpha=.88)
        if final:ax.scatter([final[-1]['steps']/1e6],[final[-1]['success']*100],marker='*',s=170,color=color,edgecolor='white',linewidth=.6,zorder=5)
        windows=[]
        for lo,hi in [(3000000,4000000),(4000000,4500000),(4500000,5000000)]:
            vals=[e['success']*100 for e in es if lo<=e['steps']<hi]
            if vals:windows.append(dict(start=lo,end=hi,n_evaluations=len(vals),mean_success=statistics.mean(vals),full_window=es[-1]['steps']>=hi-5000))
        summaries.append(dict(id=r['job']['id'],seed=seed,temperature=temp,status=r['job']['status'],progress_steps=(r.get('progress')or{}).get('steps'),latest_regular=es[-1] if es else None,final100=final[-1] if final else None,late_windows=windows))
    ax.set_title(f'iBOLT: pilot seed {seed}',fontsize=13)
    ax.set_xlim(0,5.12);ax.set_ylim(0,104);ax.set_xticks(range(6));ax.set_yticks(range(0,101,20));ax.set_xlabel('Environment steps (M)');ax.grid(alpha=.17)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Final-step success rate (%)')
handles=[Line2D([0],[0],color=c,lw=2,label=f'iBOLT T={t:g}') for t,c in zip(temperatures,colors)]
handles += [Line2D([0],[0],color='#43505e',ls='--',lw=2,label='JAX SAC: seeds 0-4 mean +/- SD'),Line2D([0],[0],color='#666',lw=0,marker='*',markersize=11,label='iBOLT final 100-episode evaluation')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),ncol=4,frameon=False,fontsize=9)
stamp=datetime.datetime.fromtimestamp(snapshot['at'],datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
fig.suptitle('Fetch Pick-and-Place | F1, HER k=4, UTD=1/8, 256 x 2\n'+stamp,fontsize=14,y=.99)
fig.text(.5,.012,'No smoothing. Curves: 10 episodes per evaluation. SAC uses different seeds and is a reference, not a paired confirmatory comparison.',ha='center',fontsize=8,color='#555')
fig.subplots_adjust(bottom=.24,top=.83,wspace=.12,left=.07,right=.98)
fig.savefig(P/'pickplace-learning-curves.png',dpi=180)
plt.close(fig)
report=dict(at=snapshot['at'],timestamp_kst=stamp,scope='Pilot diagnostic only; iBOLT seeds5/6 versus SAC seeds0-4',runs=summaries,sac_fixed_steps=fixed,sac_final100=[dict(seed=r['job']['seed'],evaluation=next((e for e in r['curve'] if e['episodes']==100 and e['steps']==5000192),None)) for r in sac],selection_released=bool(snapshot['selections'].get('fetch-pick-and-place')))
(P/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(image=str(P/'pickplace-learning-curves.png'),timestamp=stamp,completed_ibolt=sum(r['final100'] is not None for r in summaries),completed_sac=sum(x['evaluation'] is not None for x in report['sac_final100']))))
