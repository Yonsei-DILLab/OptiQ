"""Render measured GPU outputs from this directory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parent
mechanism = json.loads((root / 'mechanism.json').read_text())
learned = json.loads((root / 'learned.json').read_text())
controls = json.loads((root / 'controls.json').read_text())
plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#292e35', 'xtick.color': '#454951', 'ytick.color': '#454951'})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout='constrained')
blue, red, gray = '#19678a', '#bd5532', '#555b65'

ax = axes[0,0]
a = np.linspace(-.86, .86, 400)
mu = mechanism['calibration']['aligned']['mean']
for label, case, color in [('Aligned', 'aligned', blue), ('Lagged', 'lagged', red)]:
    d = mechanism['calibration'][case]['mean'] - mu
    x = a-d
    y = 1.25*x-2*np.maximum(x+.25, 0)
    ax.plot(a,y,color=color,label=label,lw=2)
    pair = mechanism['analytic_pair'][mechanism['conditions'].index(case)]
    ax.scatter([-.65,.65],pair['q'],s=30,color=color,zorder=3)
ax.set(xlabel='Action',ylabel='Critic value',title='A   Same local slopes, reversed action ranking')
ax.legend(frameon=False,loc='lower center',ncol=2)

ax = axes[0,1]
for i, method in enumerate(['ibolt','pathwise']):
    vals = [np.array(mechanism['results'][k][method]['true_return_change']['per_seed'])
            for k in ('aligned','lagged')]
    x = np.array([0,.45])+i*1.2
    ax.plot(x,np.stack(vals),color=gray,alpha=.16,lw=.8)
    for xi, v, color in zip(x, vals, (blue,red)):
        ax.scatter(np.repeat(xi,len(v)),v,s=8,color=color,alpha=.4)
        ax.scatter(xi,v.mean(),color=color,s=44,zorder=4,edgecolor='white',linewidth=.7)
ax.axhline(0,ls='--',color=gray,lw=1)
ax.set(xlim=(-.2,1.87),xticks=[0,.45,1.2,1.65],xticklabels=['Aligned','Lagged','Aligned','Lagged'],
       ylabel='True return change after 80 updates',title='B   Policy performance on the terminal bandit')
top = ax.secondary_xaxis('top'); top.set_xticks([.225,1.425],labels=['iBOLT','Matched pathwise'])
top.spines['top'].set_visible(False);top.tick_params(length=0,pad=1,labelsize=9)
ax.title.set_y(1.12)

ax = axes[1,0]
for flag, text, color in [(False,'Before warmup',red),(True,'After warmup',blue)]:
    rows=[row for row in controls['rows'] if row['warmup_passed']==flag]
    ax.plot([r['old_mean_shift_sd'] for r in rows],[r['inference_right_mass'] for r in rows],
            marker='o',color=color,lw=1.8,label=text)
ax.axhline(mechanism['analytic_pair'][0]['right_mass']['0.5'],color=gray,ls='--',lw=1,label='True pair target')
ax.axvline(5,ls=':',color=gray,lw=1)
ax.set(xlabel='Old feature mean shift (standard deviations)',ylabel='Right action weight (two fixed actions)',
       ylim=(.0,1.03),title='C   Default BRN correction and its clipping limit')
ax.legend(frameon=False,loc='lower right',fontsize=9)

ax=axes[1,1]
for arm,label,color in [('running','Stored statistics',red),('new_batch','Current statistics',blue)]:
    vec=[np.array(learned['summary'][f'{s}/{arm}']['pair_right_mass']['per_seed']) for s in learned['post_shift_steps']]
    m=np.array([v.mean() for v in vec]);err=np.array([t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) for v in vec])
    ax.errorbar(np.arange(len(m)),m,yerr=err,color=color,marker='o',lw=1.8,capsize=3,label=label)
ax.axhline(learned['summary']['oracle']['pair_right_mass']['mean'],color=gray,ls='--',lw=1,label='True pair target')
ax.set(xticks=range(4),xticklabels=learned['post_shift_steps'],xlabel='Critic updates since replay switch (categorical)',
       ylabel='Right action weight (two fixed actions)',ylim=(0,1),title='D   Trained twin CrossQ critic (12 seeds, 95% CI)')
ax.legend(frameon=False,loc='lower right',fontsize=9)
for axis in axes.flat:
    axis.title.set_fontsize(11)
    axis.title.set_weight('semibold')
    axis.grid(axis='y',linewidth=.5,alpha=.22)
fig.savefig(root/'toy-results.png',dpi=160,bbox_inches='tight')
