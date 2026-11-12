"""Exploratory pilot AUC ranking. Does not modify protocol or release jobs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections,json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
d=json.loads((P/'snapshot.json').read_text())
stamp=datetime.fromtimestamp(d['at'],ZoneInfo('Asia/Seoul'))
out=P/('best-curve-auc-'+stamp.strftime('%Y%m%dT%H%M'));out.mkdir(exist_ok=True)
(out/'snapshot.json').write_bytes((P/'snapshot.json').read_bytes())
TASKS=['fetch-reach','fetch-push','fetch-pick-and-place','fetch-slide','adroit-door','adroit-hammer','adroit-pen','adroit-relocate']
NAMES=['Fetch Reach','Fetch Push','Fetch Pick-and-Place','Fetch Slide (20 cm)','Adroit Door','Adroit Hammer','Adroit Pen','Adroit Relocate']
COLORS=['#1768AC','#D36C2D']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'pdf.fonttype':42})

def regular(r):
    return {e['steps']:e for e in r['curve'] if e['episodes']==10}

def stats(rows,steps):
    maps=[regular(r) for r in rows]
    y=np.array([[m[s]['success']*100 for s in steps] for m in maps])
    auc=np.trapezoid(y,np.array(steps),axis=1)/(steps[-1]-steps[0])
    return dict(y=y,mean=y.mean(0),sd=y.std(0,ddof=1),auc=auc,auc_mean=float(auc.mean()),auc_sd=float(auc.std(ddof=1)))

def plot(ax,rows,steps,color,label):
    v=stats(rows,steps);x=np.array(steps)/1e6
    ax.plot(x,v['mean'],lw=1.5,color=color,label=label)
    ax.fill_between(x,np.clip(v['mean']-v['sd'],0,100),np.clip(v['mean']+v['sd'],0,100),alpha=.12,color=color,linewidth=0)
    return v

def style(ax,name,sub,xmax=5):
    ax.set_title(name+'\n'+sub,loc='left',pad=9)
    ax.set_xlim(0,xmax);ax.set_ylim(0,102);ax.set_yticks([0,20,40,60,80,100]);ax.grid(alpha=.15,lw=.6)
    ax.set_xlabel('Environment steps (M)');ax.set_ylabel('Success rate (%)')

report=[];checks=[]
fig,axs=plt.subplots(2,4,figsize=(18,8.6),layout='constrained')
for ax,key,name in zip(axs.flat,TASKS,NAMES):
    family,task=key.split('-',1);rows=[r for r in d['rows'] if (r['job']['family'],r['job']['task'])==(family,task)]
    groups=collections.defaultdict(list)
    for r in rows:
        j=r['job']
        if j['phase'] in ['pilot','entropy_pilot']:
            assert r['completion']['completed'] and r['completion']['steps']==5000192
            groups[(j['temperature'],j.get('dacer_target_entropy_per_dim',-.9))].append(r)
    assert len(groups)==9
    for g in groups.values():assert sorted(r['job']['seed'] for r in g)==[5,6]
    sac=sorted([r for r in rows if r['job']['phase']=='baseline'],key=lambda r:r['job']['seed'])
    assert [r['job']['seed'] for r in sac]==list(range(5))
    allrows=sum(groups.values(),[])+sac
    steps=sorted(set.intersection(*(set(regular(r)) for r in allrows)))
    assert steps[0]==0 and steps[-1]>4900000
    for step in steps:assert len({regular(r)[step]['updates'] for r in allrows})==1
    if family=='adroit':assert len({r['config']['demonstration_dataset_sha256'] for r in allrows})==1
    for field in ['batch_size','num_envs','utd','horizon','gamma','tau']:
        assert len({json.dumps(r['config'].get(field),sort_keys=True) for r in allrows})==1,(key,field)
    scores=[]
    for (t,h),g in groups.items():
        v=stats(g,steps);scores.append(dict(T=t,H=h,auc=v['auc_mean'],auc_sd=v['auc_sd'],ids=[r['job']['id'] for r in g]))
    scores.sort(key=lambda s:(-round(s['auc'],10),s['T'],s['H']))
    best=scores[0];selected=groups[(best['T'],best['H'])]
    a=plot(ax,selected,steps,COLORS[0],'iBOLT');b=plot(ax,sac,steps,COLORS[1],'JAX SAC')
    style(ax,name,f"iBOLT T={best['T']:g}, H/d={best['H']:g}")
    delta=a['auc_mean']-b['auc_mean']
    ax.text(.98,.035,f"AUC: {a['auc_mean']:.1f} vs {b['auc_mean']:.1f}  ({delta:+.1f} pp)",ha='right',va='bottom',transform=ax.transAxes,
            fontsize=9,color='#3e4858',bbox=dict(facecolor='white',edgecolor='none',alpha=.86,pad=2))
    confirm=[r for r in rows if r['job']['phase'] in ['confirm','entropy_confirm'] and r['config'] and r['curve'] and
             (r['job']['temperature'],r['job'].get('dacer_target_entropy_per_dim',-.9))==(best['T'],best['H'])]
    confirm.sort(key=lambda r:r['job']['seed'])
    available=[r['job']['seed'] for r in confirm]==list(range(5))
    rec=dict(task=key,best_T=best['T'],best_H=best['H'],iBOLT_auc=a['auc_mean'],iBOLT_auc_sd=a['auc_sd'],SAC_auc=b['auc_mean'],SAC_auc_sd=b['auc_sd'],delta_pp=delta,
             common_steps=steps,pilot_ranking=scores,SAC_ids=[r['job']['id'] for r in sac],five_seed_confirmation_available=available)
    if available:
        support=sorted(set.intersection(*(set(regular(r)) for r in confirm+sac)))
        checks.append((key,name,best,confirm,sac,support));rec['confirmation_ids']=[r['job']['id'] for r in confirm];rec['confirmation_common_end']=support[-1]
    report.append(rec)
fig.suptitle('Exploratory: best iBOLT pilot learning curve vs JAX SAC\n'+
             'iBOLT: highest AUC among 9 settings, seeds 5/6 (n=2) | SAC: fixed setting, seeds 0–4 (n=5)\n'+
             'Same step grid · mean ± sample SD · no smoothing · training seeds are NOT paired',fontsize=14)
fig.legend(handles=[Line2D([0],[0],color=COLORS[0],lw=2,label='iBOLT pilot: post hoc AUC winner'),Line2D([0],[0],color=COLORS[1],lw=2,label='JAX SAC reference')],loc='outside lower center',ncol=2,frameon=False)
for ext in ['png','pdf']:fig.savefig(out/f'best-pilot-vs-sac.{ext}',dpi=180)
plt.close(fig)

fig,axs=plt.subplots(1,len(checks),figsize=(18,4.8),layout='constrained')
for ax,(key,name,best,conf,sac,steps) in zip(np.atleast_1d(axs),checks):
    for s in steps:assert len({regular(r)[s]['updates'] for r in conf+sac})==1
    a=plot(ax,conf,steps,COLORS[0],'iBOLT');b=plot(ax,sac,steps,COLORS[1],'JAX SAC')
    xmax=max(.1,math.ceil(steps[-1]/1e6/.1)*.1)
    style(ax,name,f"T={best['T']:g}, H/d={best['H']:g} | through {steps[-1]/1e6:.2f}M",xmax)
    rec=next(r for r in report if r['task']==key);rec['independent_common_auc']=dict(ibolt=a['auc_mean'],sac=b['auc_mean'],delta_pp=a['auc_mean']-b['auc_mean'])
fig.suptitle('Available independent checks of those pilot-selected settings\nBoth methods: seeds 0–4, exact shared steps, mean ± sample SD; x-axis ranges differ',fontsize=13)
fig.legend(handles=[Line2D([0],[0],color=COLORS[0],lw=2,label='iBOLT (5 seeds)'),Line2D([0],[0],color=COLORS[1],lw=2,label='JAX SAC (5 seeds)')],loc='outside lower center',ncol=2,frameon=False)
for ext in ['png','pdf']:fig.savefig(out/f'available-five-seed-checks.{ext}',dpi=180)
plt.close(fig)
result=dict(at=d['at'],snapshot_sha256=hashlib.sha256((out/'snapshot.json').read_bytes()).hexdigest(),
            scope='Current F1/A1 protocol only; Slide20cm; exclude all legacy variants',
            criterion='Trapezoidal success-rate AUC divided by shared elapsed environment steps, using regular 10-episode evaluations',
            caveat='Exploratory post hoc pilot ranking; unequal, unpaired seed groups. Original last-five-evaluation selections and running jobs unchanged.',
            comparisons=report)
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(directory=str(out),comparisons=[{k:v for k,v in r.items() if k in ['task','best_T','best_H','iBOLT_auc','SAC_auc','delta_pp','independent_common_auc','five_seed_confirmation_available']} for r in report])))
