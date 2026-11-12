from pathlib import Path
import json,statistics,math,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;data=json.loads((P/'evidence.json').read_text())
tasks=[('fetch',t) for t in ['reach','push','pick-and-place','slide']]+[('adroit',t) for t in ['door','hammer','pen','relocate']]
colors={(64,64):'#277DA8',(64,256):'#D9583F',(16,64):'#319269'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(16,8.2))
report=[];nonfinite=[]
for ax,(fam,task) in zip(axes.flat,tasks):
 rr=[r for r in data['runs'] if r['job']['family']==fam and r['job']['task']==task]
 available=[r for r in rr if r['series']]
 maps=[{z['step']:z for z in r['series']} for r in available]
 shared=sorted(set.intersection(*(set(m) for m in maps)))
 step=min(shared,key=lambda s:abs(s-2000000))
 entries=[]
 for r,mp in zip(available,maps):
  j=r['job'];nm=(j.get('num_policy_samples',64),j.get('num_reference_actions',64));ss=r['series']
  x=np.array([s['step'] for s in ss])/1e6;y=np.array([s['metrics']['train/current_q_values'] for s in ss])
  ax.plot(x,y,color=colors[nm],ls='-' if j['seed']==5 else '--',lw=1.25,alpha=.9,label=f'N{nm[0]}/M{nm[1]}, s{j["seed"]}')
  for z in ss:
   for k,v in z['metrics'].items():
    if isinstance(v,(int,float)) and not math.isfinite(v):nonfinite.append(dict(id=j['id'],step=z['step'],key=k))
  last=ss[-1];recent=[z for z in ss if z['step']>=last['step']-200000]
  def onset(bound,k=3):return next((ss[i]['step'] for i in range(len(ss)-k+1) if all(bound(z['metrics']['train/current_q_values']) for z in ss[i:i+k])),None)
  entries.append(dict(id=j['id'],seed=j['seed'],N=nm[0],M=nm[1],common_step=step,common_q=mp[step]['metrics']['train/current_q_values'],common_loss=mp[step]['metrics']['train/critic_loss'],latest_step=last['step'],latest_q=float(y[-1]),q_min=float(y.min()),q_max=float(y.max()),recent_q_median=statistics.median(z['metrics']['train/current_q_values'] for z in recent),recent_loss_median=statistics.median(z['metrics']['train/critic_loss'] for z in recent),latest_ess=last['metrics'].get('train/importance_ess_mean'),first_sustained_q_below_minus100=onset(lambda q:q< -100) if fam=='fetch' else None,first_sustained_q_above1100=onset(lambda q:q>1100) if fam=='adroit' else None))
 if fam=='fetch':
  ax.axhline(-100,color='#555',lw=.8,ls=':');ax.axhline(0,color='#555',lw=.8,ls=':')
  if task=='slide':ax.set_yscale('symlog',linthresh=20)
 else:
  ax.axhline(1000,color='#555',lw=1,ls=':');ax.set_ylim(-30,2300)
 ax.set_title(f'{fam.title()} {task.title()}'+(' (20 cm)' if task=='slide' else ''))
 ax.set_xlim(0,5.03);ax.set_xlabel('Environment steps (million)');ax.grid(alpha=.15)
 ax.legend(fontsize=6.2,ncol=2,loc='best')
 report.append(dict(family=fam,task=task,common_training_step=step,missing_history=[r['job']['id'] for r in rr if not r['series']],runs=entries))
axes[0,0].set_ylabel('Mean replay min-Q');axes[1,0].set_ylabel('Mean replay min-Q')
stamp=datetime.datetime.fromtimestamp(data['at'],datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
fig.suptitle(f'iBOLT critic audit | {stamp}\nActual unsmoothed logs; seeds5/6. Dotted lines: return bounds with gamma=0.99; Slide uses symlog.',fontsize=13)
fig.text(.5,.012,'Adroit N64/M64 historical Q logs were not saved; those controls are checked separately using frozen critics on shared expert transitions.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.92));fig.savefig(P/'training-q-curves.png',dpi=180);fig.savefig(P/'training-q-curves.pdf');plt.close(fig)
result=dict(at=data['at'],timestamp_kst=stamp,runs_checked=len(data['runs']),runs_with_history=sum(bool(r['series']) for r in data['runs']),nonfinite=nonfinite,return_bounds={'fetch':[-100,0],'adroit':[-10,1000]},bound_assumptions='Current sparse rewards Fetch[-1,0], Adroit[-0.1,10], gamma0.99; iBOLT Bellman target has no entropy reward bonus.',tasks=report)
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(stamp,'nonfinite',len(nonfinite))
for t in report:
 print(t['family'],t['task'],'common',t['common_training_step'],[(r['N'],r['M'],r['seed'],round(r['common_q'],1),round(r['latest_q'],1),r['latest_step']) for r in t['runs']])
