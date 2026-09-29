"""Render rewards and summarize the completed single-seed sigma comparison."""
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
assert len(jobs)==20 and all(j['status']=='completed' for j in jobs)
rows=[]
for job in jobs:
    assert job['timesteps']==10000 and job['updates']==5000
    assert {'actor_state_10000.msgpack','critic_state_10000.msgpack'}<=set(job['final_checkpoints'])
    for mode in ('zero_z','stochastic_z'):
        ev=job['evaluations'][mode]
        steps=np.asarray(ev['timesteps']);rewards=np.asarray(ev['results'])
        assert steps.tolist()==[1]+list(range(1000,10001,1000))
        assert rewards.shape==(11,10) and np.isfinite(rewards).all()
        means=rewards.mean(axis=1);mask=steps>=5000
        rows.append(dict(task=job['task'],sigma=job['sigma'],mode=mode,
            final_return=float(means[-1]),eval_episode_std=float(rewards[-1].std()),
            mean_return_5k_10k=float(np.trapz(means[mask],steps[mask])/5000),
            improvement_from_5k=float(means[-1]-means[5]),wandb_url=job['wandb_url']))
with (ROOT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(15,8.5))
for ax,task in zip(axs.flat,TASKS):
    ax.axvspan(0,5,color='#eeeeee',zorder=-1)
    ax.axvline(5,color='#9b9b9b',ls='--',lw=1)
    for sigma in SIGMAS:
        job=next(j for j in jobs if j['task']==task and j['sigma']==sigma)
        ev=job['evaluations']['stochastic_z']
        values=np.asarray(ev['results']).mean(axis=1)
        ax.plot(np.asarray(ev['timesteps'])/1000,values,'o-',ms=3,lw=2,color=COLORS[sigma],label=f'Initial sigma = {sigma:g}')
    ax.set(title=NAMES[task],xlabel='Environment steps (thousands)',ylabel='Mean evaluation return',xlim=(0,10))
    ax.grid(alpha=.15)
axs.flat[5].axis('off')
handles,labels=axs.flat[0].get_legend_handles_labels()
axs.flat[5].legend(handles,labels,loc='upper left',frameon=False,fontsize=13)
axs.flat[5].text(.03,.5,'direct-gmm-trg | truncated Gaussian\nSeed 0 | 10 episodes per evaluation\nStochastic z, conditional mean action\n\nGray: 5k-step random-action warmup\nOnly initialization differs; sigma is learned.\n64 x 64 | T=0.25 | batch=256 | UTD=1\n256 x 2 | Adam 3e-4 | no grad clipping',va='top',transform=axs.flat[5].transAxes,linespacing=1.6)
fig.suptitle('Initial sigma comparison: 10k environment steps / 5k learning updates',fontsize=17)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(ROOT/'stochastic_reward.png',dpi=170)
fig.savefig(ROOT/'stochastic_reward.pdf')
plt.close(fig)

fig,axs=plt.subplots(5,2,figsize=(13,17))
for row,task in enumerate(TASKS):
    for col,mode in enumerate(('zero_z','stochastic_z')):
        ax=axs[row,col]
        ax.axvspan(0,5,color='#eeeeee',zorder=-1)
        ax.axvline(5,color='#999999',ls='--',lw=1)
        for sigma in SIGMAS:
            job=next(j for j in jobs if j['task']==task and j['sigma']==sigma)
            ev=job['evaluations'][mode]
            ax.plot(np.asarray(ev['timesteps'])/1000,np.asarray(ev['results']).mean(axis=1),'o-',ms=3,color=COLORS[sigma],label=f'sigma={sigma:g}')
        ax.set(title=f'{NAMES[task]} | {mode}',xlabel='Environment steps (thousands)',ylabel='Mean evaluation return')
        ax.grid(alpha=.15)
axs[0,0].legend(frameon=False,ncol=2)
fig.suptitle('Separate zero-z and stochastic-z evaluation | seed 0 | 10 episodes / point',fontsize=16)
fig.tight_layout(rect=(0,0,1,.97))
fig.savefig(ROOT/'both_eval_modes.png',dpi=150)
fig.savefig(ROOT/'both_eval_modes.pdf')
plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(15,8.5))
for ax,task in zip(axs.flat,TASKS):
    for sigma in SIGMAS:
        job=next(j for j in jobs if j['task']==task and j['sigma']==sigma)
        history=job.get('sigma_history',[])
        ax.plot([5]+[p['time/total_timesteps']/1000 for p in history],
                [sigma]+[p['train/actor_std_mean'] for p in history],
                color=COLORS[sigma],label=f'Initial sigma={sigma:g}')
    ax.axhline(1.0,color='#999999',ls='--',lw=1)
    ax.set(title=NAMES[task],xlabel='Environment steps (thousands)',ylabel='Mean learned Gaussian scale',xlim=(5,10),ylim=(0,1.05))
    ax.grid(alpha=.15)
axs.flat[5].axis('off')
handles,labels=axs.flat[0].get_legend_handles_labels()
axs.flat[5].legend(handles,labels,loc='upper left',frameon=False)
axs.flat[5].text(.03,.5,'Dashed: branch scale ceiling exp(0)\n\nBatch means on training observations;\nthese are Gaussian scale parameters\nbefore truncation, not action std.',va='top',transform=axs.flat[5].transAxes,linespacing=1.5)
fig.suptitle('How quickly the initial scale changes during learning',fontsize=17)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(ROOT/'learned_sigma.png',dpi=170)
plt.close(fig)

lines=['# 초기 sigma 비교 결과','',
'20/20 runs completed: seed 0, 10k environment steps, 5k learning updates.',
'W&B: https://wandb.ai/OptiQ/abla','',
'Source commit: `31e84e85c0847e3ffd8c9cb3dbaa20031c300e6d`. Initial sigma only; other training defaults unchanged.',
'The sweep changes the Gaussian scale parameter before truncation, which remains trainable.',
'Each evaluation uses 10 paired-seed episodes per mode. Episode standard deviation is not training-seed uncertainty.','',
'## Stochastic-z return at 10k','',
'| Environment | sigma=.01 | sigma=.05 | sigma=.1 | sigma=.2 | Highest 5k–10k mean return |',
'|---|---:|---:|---:|---:|---|']
for task in TASKS:
    rs=[next(r for r in rows if r['task']==task and r['sigma']==s and r['mode']=='stochastic_z') for s in SIGMAS]
    best=max(rs,key=lambda r:r['mean_return_5k_10k'])
    lines.append('| '+NAMES[task]+' | '+' | '.join(f"{r['final_return']:.1f}" for r in rs)+f" | sigma={best['sigma']:g} |")
lines+=['','5k–10k mean return is trapezoidal reward AUC divided by 5,000 environment steps.',
'One seed and only 5k updates describe this short comparison; they do not establish long-run convergence or a general best sigma.','',
'## Runs','']
for j in sorted(jobs,key=lambda j:(TASKS.index(j['task']),j['sigma'])):
    lines.append(f"- {NAMES[j['task']]} sigma={j['sigma']:g}: {j['wandb_url']}")
(ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:22]))
