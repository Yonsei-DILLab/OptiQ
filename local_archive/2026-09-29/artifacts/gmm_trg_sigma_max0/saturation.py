"""Plot measured occupancy of the configured log-sigma cap (zero)."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
TASKS=('ant','walker2d','humanoid','hopper','halfcheetah')
NAMES=dict(ant='Ant',walker2d='Walker2d',humanoid='Humanoid',hopper='Hopper',halfcheetah='HalfCheetah')
SIGMAS=(.01,.05,.1,.2)
COLORS={.01:'#8c52b8',.05:'#259e8f',.1:'#2374bd',.2:'#e68a2e'}
jobs=[j for path in sorted(ROOT.glob('host*.json')) for j in json.loads(path.read_text())['jobs']]
rows=[]
for j in jobs:
    m=j.get('last_metrics',{})
    history=[p for p in j.get('sigma_history',[]) if 'train/actor_std_at_max_fraction' in p]
    if not history:continue
    assert j['config']['alg']['actor']['log_std_max']==0
    assert j['config']['alg']['actor']['log_std_min']==-5
    high=[p['time/total_timesteps'] for p in history if p['train/actor_std_at_max_fraction']>=.95]
    rows.append(dict(task=j['task'],initial_sigma=j['sigma'],status=j['status'],
        final_diagnostic_step=history[-1]['time/total_timesteps'],
        fraction_at_log_sigma_0=m['train/actor_std_at_max_fraction'],
        mean_sigma=m['train/actor_std_mean'],mean_log_sigma=m['train/actor_log_std_mean'],
        max_log_sigma=m['train/actor_log_std_max'],
        first_95_percent_step=high[0] if high else None,
        last_3_diagnostics_all_above_95_percent=len(history)>=3 and all(p['train/actor_std_at_max_fraction']>=.95 for p in history[-3:]),
        wandb_url=j['wandb_url']))
(ROOT/'saturation_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if rows:
    with (ROOT/'saturation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(15,8.5))
for ax,task in zip(axs.flat,TASKS):
    for sigma in SIGMAS:
        matches=[j for j in jobs if j['task']==task and j['sigma']==sigma]
        if not matches:continue
        history=[p for p in matches[0].get('sigma_history',[]) if 'train/actor_std_at_max_fraction' in p]
        if not history:continue
        ax.plot([5]+[p['time/total_timesteps']/1000 for p in history],
            [0]+[100*p['train/actor_std_at_max_fraction'] for p in history],
            'o-',ms=4,lw=2,color=COLORS[sigma],label=f'Initial sigma = {sigma:g}')
    ax.axhline(95,color='#9e9e9e',ls='--',lw=1)
    ax.set(title=NAMES[task],xlabel='Environment steps (thousands)',
        ylabel='Outputs at log-sigma cap 0 (%)',xlim=(5,10),ylim=(-3,103))
    ax.grid(alpha=.15)
axs.flat[5].axis('off')
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],color=COLORS[s],marker='o',label=f'Initial sigma = {s:g}') for s in SIGMAS]
axs.flat[5].legend(handles=handles,loc='upper left',frameon=False,fontsize=13)
axs.flat[5].text(.03,.5,'log-sigma range: [-5, 0]\nCap 0 corresponds to sigma = 1\nDashed line: 95% of sampled outputs\n\nFractions over states x latents x action dims.\nTraining minibatch diagnostics every 1k.\nSeed 0; warmup ends at step 5k.\nSigma is the scale before truncation.',va='top',transform=axs.flat[5].transAxes,linespacing=1.6)
complete=sum(j['status']=='completed' for j in jobs)
fig.suptitle(f'Does log-sigma stick to the new cap? | {complete}/20 runs completed',fontsize=17)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(ROOT/'cap_saturation.png',dpi=170)
fig.savefig(ROOT/'cap_saturation.pdf')
plt.close(fig)
for task in TASKS:
    print(NAMES[task],[(r['initial_sigma'],round(r['fraction_at_log_sigma_0']*100,2),round(r['mean_sigma'],4),r['final_diagnostic_step']) for r in rows if r['task']==task])
