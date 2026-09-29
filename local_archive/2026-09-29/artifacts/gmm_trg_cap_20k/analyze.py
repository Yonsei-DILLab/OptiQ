"""Compare two log-scale caps without pooling environments or evaluation modes."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
jobs=json.loads((ROOT/'host180.json').read_text())['jobs']
COMPLETE=len(jobs)==4 and all(j['status']=='completed' for j in jobs)
if '--partial' not in sys.argv:assert COMPLETE
COLORS={0.:'#2374bd',-2.:'#e17a27'}
TASKS=('ant','humanoid')
NAMES={'ant':'Ant','humanoid':'Humanoid'}
rows=[]
for j in jobs:
    a=j.get('config',{}).get('alg',{}).get('actor',{})
    if not a:continue
    assert a['log_std_min']==-5 and a['log_std_max']==j['cap']
    assert np.isclose(np.exp(a['initial_log_std']),.1)
    if COMPLETE:
        assert j['timesteps']==20000 and j['updates']==15000
        assert {'actor_state_20000.msgpack','critic_state_20000.msgpack'}<=set(j['final_checkpoints'])
    for mode in ('zero_z','stochastic_z'):
        ev=j.get('evaluations',{}).get(mode)
        if not ev:continue
        x=np.asarray(ev['timesteps']);y=np.asarray(ev['results']);means=y.mean(axis=1)
        if COMPLETE:
            assert x.tolist()==[1]+list(range(1000,20001,1000)) and y.shape==(21,10)
            assert np.isfinite(y).all()
        mask=x>=5000
        auc=float(np.trapz(means[mask],x[mask])/(x[-1]-5000)) if x[-1]>5000 else None
        m=j.get('last_metrics',{})
        rows.append(dict(task=j['task'],log_sigma_cap=j['cap'],mode=mode,
            last_step=int(x[-1]),return_10k=float(means[list(x).index(10000)]) if 10000 in x else None,
            final_return=float(means[-1]),eval_episode_std=float(y[-1].std()),
            final_mean_ep_length=float(np.asarray(ev['ep_lengths'])[-1].mean()),
            mean_return_after_warmup=auc,mean_sigma=m.get('train/actor_std_mean'),
            mean_log_sigma=m.get('train/actor_log_std_mean'),max_log_sigma=m.get('train/actor_log_std_max'),
            fraction_at_cap=m.get('train/actor_std_at_max_fraction'),wandb_url=j['wandb_url']))
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if rows:
    with (ROOT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(16,8.5))
for ri,task in enumerate(TASKS):
    for j in sorted((j for j in jobs if j['task']==task),key=lambda j:-j['cap']):
        cap=j['cap'];color=COLORS[cap];label=f'log cap {cap:g} (sigma <= {np.exp(cap):.3g})'
        ev=j.get('evaluations',{}).get('stochastic_z')
        if ev:
            axs[ri,0].plot(np.asarray(ev['timesteps'])/1000,np.asarray(ev['results']).mean(axis=1),
                'o-',ms=3,color=color,lw=2,label=label)
        hist=j.get('sigma_history',[])
        if hist:
            axs[ri,1].plot([5]+[p['time/total_timesteps']/1000 for p in hist],
                [.1]+[p['train/actor_std_mean'] for p in hist],color=color,lw=2,label=label)
            diags=[p for p in hist if 'train/actor_std_at_max_fraction' in p]
            axs[ri,2].plot([5]+[p['time/total_timesteps']/1000 for p in diags],
                [0]+[100*p['train/actor_std_at_max_fraction'] for p in diags],
                'o-',ms=3,color=color,lw=2,label=label)
    axs[ri,0].axvspan(0,5,color='#eeeeee',zorder=-1)
    axs[ri,0].set(title=f'{NAMES[task]} | stochastic-z return',ylabel='Mean episode return',xlim=(0,20))
    axs[ri,1].set(title=f'{NAMES[task]} | learned scale',ylabel='Mean Gaussian sigma',ylim=(0,1.05),xlim=(5,20))
    for cap in (0.,-2.):axs[ri,1].axhline(np.exp(cap),color=COLORS[cap],alpha=.45,lw=1,ls='--')
    axs[ri,2].set(title=f'{NAMES[task]} | saturation at each cap',ylabel='Outputs at configured cap (%)',ylim=(-3,103),xlim=(5,20))
    axs[ri,2].axhline(95,color='#999999',lw=1,ls='--')
    for ax in axs[ri]:ax.set_xlabel('Environment steps (thousands)');ax.grid(alpha=.15)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False,fontsize=12)
status='completed' if COMPLETE else 'in progress'
fig.suptitle(f'Log-sigma upper bound 0 vs -2 | matched initial sigma 0.1 | seed 0 | {status}',fontsize=17)
fig.text(.5,.012,'20k environment steps = 5k warmup + 15k updates. Evaluation: 10 episodes per mode. Sigma is before truncation; cap fractions aggregate training minibatch outputs.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.9))
fig.savefig(ROOT/'cap_comparison.png',dpi=180)
fig.savefig(ROOT/'cap_comparison.pdf')
plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,8))
for ri,task in enumerate(TASKS):
    for ci,mode in enumerate(('zero_z','stochastic_z')):
        ax=axs[ri,ci]
        ax.axvspan(0,5,color='#eeeeee',zorder=-1)
        for j in jobs:
            ev=j.get('evaluations',{}).get(mode)
            if j['task']!=task or not ev:continue
            ax.plot(np.asarray(ev['timesteps'])/1000,np.asarray(ev['results']).mean(axis=1),
                'o-',ms=3,color=COLORS[j['cap']],label=f"log cap {j['cap']:g}")
        ax.set(title=f'{NAMES[task]} | {mode}',xlabel='Environment steps (thousands)',ylabel='Mean episode return',xlim=(0,20));ax.grid(alpha=.15)
axs[0,0].legend(frameon=False)
fig.suptitle('Separate zero-z and stochastic-z evaluation | 10 episodes per point',fontsize=15)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(ROOT/'both_eval_modes.png',dpi=170);plt.close(fig)
print(json.dumps(rows,indent=2))
