"""Plot actual matched support; never fill missing seeds or extrapolate runs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
data=json.loads((P/'snapshot.json').read_text())
stamp=datetime.fromtimestamp(data['at'],ZoneInfo('Asia/Seoul'))
out=P/('visualization-'+stamp.strftime('%Y%m%dT%H%M'));out.mkdir(exist_ok=True)
(out/'snapshot.json').write_bytes((P/'snapshot.json').read_bytes())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#394150','xtick.color':'#526071','ytick.color':'#526071',
                     'savefig.facecolor':'white','pdf.fonttype':42})
tasks=['fetch-reach','fetch-push','fetch-pick-and-place','fetch-slide',
       'adroit-door','adroit-hammer','adroit-pen','adroit-relocate']
names=['Fetch Reach','Fetch Push','Fetch Pick-and-Place','Fetch Slide (20 cm)',
       'Adroit Door','Adroit Hammer','Adroit Pen','Adroit Relocate']
colors=['#1768AC','#D36C2D']
report=[]
fig,axs=plt.subplots(2,4,figsize=(17.8,8.7),layout='constrained')
for ax,key,name in zip(axs.flat,tasks,names):
    family,task=key.split('-',1)
    chosen=data['entropy']['selections'][key];orig=data['selections'][key]
    phase='confirm' if chosen['target_entropy_per_dim']==-.9 and orig['temperature']==chosen['temperature'] else 'entropy_confirm'
    grouped=[]
    for stage in [phase,'baseline']:
        rows=sorted([r for r in data['rows'] if r['job']['family']==family and r['job']['task']==task and r['job']['phase']==stage],key=lambda r:r['job']['seed'])
        assert [r['job']['seed'] for r in rows]==list(range(5)),(key,stage)
        grouped.append(rows)
    for a,b in zip(*grouped):
        for field in ['seed','batch_size','num_envs','utd','horizon','gamma','tau']:
            assert a['config'].get(field)==b['config'].get(field),(key,field)
        if family=='adroit':assert a['config']['demonstration_dataset_sha256']==b['config']['demonstration_dataset_sha256']
    maps=[[{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in rows] for rows in grouped]
    shared=sorted(set.intersection(*(set(m) for group in maps for m in group)))
    assert shared,(key,'missing matched evaluations')
    for step in shared:
        assert len({m[step]['updates'] for group in maps for m in group})==1,(key,step)
    latest=[]
    for color,group in zip(colors,maps):
        y=np.array([[m[s]['success']*100 for s in shared] for m in group])
        mean=y.mean(0);sd=y.std(0,ddof=1);x=np.array(shared)/1e6
        ax.plot(x,mean,color=color,lw=1.8)
        ax.fill_between(x,np.clip(mean-sd,0,100),np.clip(mean+sd,0,100),color=color,alpha=.15,linewidth=0)
        latest.append(dict(mean=float(mean[-1]),sd=float(sd[-1])))
    finals=[[next((e for e in r['curve'] if e['steps']==5000192 and e['episodes']==100),None) for r in rows] for rows in grouped]
    complete=all(e is not None for group in finals for e in group)
    final_stats=[]
    if complete:
        assert len({e['updates'] for group in finals for e in group})==1
        for color,group in zip(colors,finals):
            y=np.array([e['success']*100 for e in group]);mean=y.mean();sd=y.std(ddof=1)
            ax.errorbar([5.000192],[mean],yerr=[[min(sd,mean)],[min(sd,100-mean)]],fmt='D',color=color,ms=5,capsize=3,elinewidth=1.1,zorder=5)
            final_stats.append(dict(mean=float(mean),sd=float(sd)))
    xmax=5.12 if complete else max(.05,math.ceil(shared[-1]/1e6/.05)*.05)
    ax.set_xlim(0,xmax);ax.set_ylim(0,103);ax.set_yticks([0,20,40,60,80,100])
    ax.grid(axis='both',alpha=.14,linewidth=.6)
    ax.set_xlabel('Environment steps (M)')
    ax.set_ylabel('Success rate (%)')
    ax.set_title(name+'\n'+f"T={chosen['temperature']:g}, H/d={chosen['target_entropy_per_dim']:g}",loc='left',pad=10)
    ax.text(.98,.035,f"Matched through {shared[-1]/1e6:.3f}M",ha='right',va='bottom',transform=ax.transAxes,
            fontsize=9,color='#526071',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    report.append(dict(task=key,temperature=chosen['temperature'],entropy_target=chosen['target_entropy_per_dim'],
                       phase=phase,run_ids=[[r['job']['id'] for r in rows] for rows in grouped],
                       common_steps=shared,latest_regular=latest,final_100_episode=final_stats))
fig.suptitle('iBOLT vs JAX SAC | independent seeds 0–4\n'+
             'Exact shared steps · mean ± sample SD · unsmoothed · panel x-axis ranges differ\n'+
             stamp.strftime('Updated %Y-%m-%d %H:%M KST'),fontsize=15)
fig.legend(handles=[Line2D([0],[0],color=colors[0],lw=2,label='iBOLT: selected T and H/d'),
                    Line2D([0],[0],color=colors[1],lw=2,label='JAX SAC'),
                    Line2D([0],[0],color='#556170',marker='D',ls='none',label='5M final: 100 episodes/run')],
           loc='outside lower center',ncol=3,frameon=False,fontsize=11)
for ext in ['png','pdf']:fig.savefig(out/f'matched-five-seed-success.{ext}',dpi=180)
plt.close(fig)
(out/'pairing.json').write_text(json.dumps(dict(snapshot_sha256=hashlib.sha256((out/'snapshot.json').read_bytes()).hexdigest(),
                                             at=data['at'],no_smoothing=True,ddof=1,comparisons=report),indent=2))
print(json.dumps(dict(directory=str(out),panels=[dict(task=r['task'],common_M=r['common_steps'][-1]/1e6) for r in report])))
