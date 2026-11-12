from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
d=json.loads((P/'q-comparison.json').read_text());rows=d['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
colors={'ibolt':'#d35b26','sac_auto':'#2166ac','sac_zero':'#238b45'}
names={'ibolt':'iBOLT (reward Q; N=64, M=64)','sac_auto':'JAX SAC (automatic alpha; soft Q)','sac_zero':'JAX SAC (fixed alpha=0; reward Q)'}
titles={'reach':'Reach','push':'Push','pick-and-place':'Pick-and-Place','slide':'Slide (20 cm)','door':'Door','hammer':'Hammer','pen':'Pen','relocate':'Relocate'}
def get(f,t,m,s=None):
    a=[r for r in rows if (r['family'],r['task'],r['method'])==(f,t,m) and (s is None or r['step']==s)]
    return sorted(a,key=lambda r:(r['step'],r['seed']))
def values(f,t,m,s):
    a=get(f,t,m,s);assert len(a)==2 and [x['seed'] for x in a]==[0,1]
    return np.array([x['q_min_mean'] for x in a])

fig,axes=plt.subplots(2,2,figsize=(12.8,8.8))
for ax,task in zip(axes.flat,['reach','push','pick-and-place','slide']):
    for method in ['ibolt','sac_auto','sac_zero']:
        ss=sorted({r['step'] for r in get('fetch',task,method)})
        v=np.array([values('fetch',task,method,s) for s in ss]);mu=v.mean(1);sd=v.std(1,ddof=1);xx=np.array(ss)/1e6
        ax.plot(xx,mu,color=colors[method],lw=2.4,marker='o',markersize=4,linestyle='--' if method=='sac_zero' else '-')
        ax.fill_between(xx,mu-sd,mu+sd,color=colors[method],alpha=.13)
        if method!='sac_zero':
            for seed in [0,1]:ax.plot(xx,v[:,seed],color=colors[method],lw=.7,alpha=.35)
    if task=='slide':
        ax.axhline(-100,color='#666',ls=':',lw=1.2)
        ax.text(.025,-100,' Reward-only lower bound = -100',transform=ax.get_yaxis_transform(),fontsize=9,va='bottom',color='#555')
        ax.set_ylim(-500,12)
    ax.axhline(0,color='#aaa',lw=.7)
    ax.set_title(titles[task]);ax.set_xlabel('Environment steps (millions)')
    ax.set_ylabel('Mean min(Q1, Q2) on fixed inputs');ax.grid(alpha=.16)
    ax.set_xlim(0,5.1 if task=='slide' else 3.1)
fig.suptitle('Fetch: same-seed, same-step critic comparison',fontsize=18,y=.985)
fig.legend([Line2D([0],[0],color=colors[m],lw=2.5,ls='--' if m=='sac_zero' else '-') for m in names],list(names.values()),loc='upper center',bbox_to_anchor=(.5,.951),ncol=3,frameon=False,fontsize=10)
fig.text(.055,.045,'500 identical raw state/action pairs per task from independent random rollouts; each checkpoint uses its own frozen normalizer.',fontsize=10,color='#444')
fig.text(.055,.023,'Mean ± sample SD across seeds 0/1. Checkpoint probes, not training minibatch logs. Alpha=0 currently shown only through 0.20M.',fontsize=10,color='#444')
fig.tight_layout(rect=[0,.075,1,.91])
for ext in ['png','pdf']:fig.savefig(P/('fetch-q-comparison.'+ext),dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,4,figsize=(14,5.8),sharey=True)
for ax,task in zip(axes,['door','hammer','pen','relocate']):
    for j,m in enumerate(['ibolt','sac_auto']):
        v=values('adroit',task,m,5000192);mu=v.mean();sd=v.std(ddof=1)
        ax.bar(j,mu,width=.57,color=colors[m],alpha=.80)
        ax.errorbar(j,mu,yerr=sd,fmt='none',ecolor='#333',capsize=5,lw=1.3)
        ax.scatter(j+np.array([-.055,.055]),v,c='white',s=32,edgecolors='#333',zorder=5)
        ax.text(j,mu+sd+70,f'{mu:,.0f}',ha='center',fontsize=11,fontweight='bold')
    ax.axhline(1000,color='#666',ls=':',lw=1.2)
    ax.set_xticks([0,1],['iBOLT','JAX SAC']);ax.set_title(titles[task]);ax.set_ylim(0,2700);ax.grid(axis='y',alpha=.16)
axes[0].set_ylabel('Mean min(Q1, Q2) on identical expert inputs')
fig.suptitle('Adroit at 5M: iBOLT vs JAX SAC',fontsize=18,y=.98)
fig.text(.5,.895,'All tasks: critic LayerNorm, identical verified expert dataset; same training seeds 0/1 and 5,000,192 steps.',ha='center',fontsize=11)
fig.text(.055,.10,'Bars: mean ± sample SD across seeds 0/1; dots: individual seeds. 500 identical expert state/action pairs per task.',fontsize=10,color='#444')
fig.text(.055,.064,'Dotted line: reward-only upper bound 1,000 (applies to iBOLT). Automatic-alpha SAC learns soft Q with future entropy terms.',fontsize=10,color='#444')
fig.text(.055,.028,'Only final SAC checkpoints were saved for Adroit, so these are matched 5M snapshots, not learning curves.',fontsize=10,color='#444')
fig.tight_layout(rect=[0,.15,1,.855])
for ext in ['png','pdf']:fig.savefig(P/('adroit-q-comparison.'+ext),dpi=180)
plt.close(fig)

table=[]
for f,t in dict.fromkeys((r['family'],r['task']) for r in rows):
    step=max(r['step'] for r in get(f,t,'ibolt'))
    methods={}
    for m in ['ibolt','sac_auto']:
        v=values(f,t,m,step);methods[m]={'seeds':v.tolist(),'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1))}
    table.append(dict(family=f,task=t,step=step,methods=methods))
(P/'summary.json').write_text(json.dumps(dict(checkpoints=len(rows),seeds=[0,1],latest_paired=table,source='q-comparison.json'),indent=2)+'\n')
print(json.dumps(table,indent=2))
