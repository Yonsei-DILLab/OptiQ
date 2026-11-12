from pathlib import Path
import json, subprocess, statistics, datetime, zoneinfo, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
remote=r"""
from pathlib import Path
import json,time
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');rows=[]
def read(p):return json.loads(p.read_text()) if p.exists() else None
for h in [1,3,4,5]:
 base=root/f'hosts/host{h}/workspace/heechan-unified-f1-a1-20261112';plan=read(base/'plan.json')
 for j in plan['jobs']:
  if j['task']!='slide' or j['phase'] not in ('pilot','baseline'):continue
  assert j['id'].startswith('fetch-slide-r20cm-') and j['distance_threshold_m']==.2
  d=base/'runs'/j['id'];ev=[]
  if (d/'evaluations.jsonl').exists():
   for line in (d/'evaluations.jsonl').read_text().splitlines():
    try:ev.append(json.loads(line))
    except json.JSONDecodeError:pass
  rows.append(dict(job=j,config=read(d/'config.json'),progress=read(d/'progress.json'),evaluations=ev))
print(json.dumps(dict(at=time.time(),rows=rows)))
"""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=remote,text=True,capture_output=True,timeout=45)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout);assert len(d['rows'])==15
(P/'evidence.json').write_text(json.dumps(d,indent=2))
rows=d['rows'];temps=[.05,.1,.25,.5,1.]
for r in rows:
 r['map']={}
 c=r['config']
 if c:assert c['distance_threshold_m']==.2 and c['protocol_variant']=='F1-slide-20cm' and c['horizon']==50
 for e in r['evaluations']:
  v=e[e['primary_mode']]
  assert v['distance_threshold_m']==.2 and v['episodes']==10
  assert all(x==50 for x in v.get('lengths',[50]))
  r['map'][e['steps']]=dict(updates=e['updates'],success=100*v['success_rate'],raw_return=v['mean_return'])

def stats(rr,step):
 if not rr or any(step not in r['map'] for r in rr):return None
 vals=[r['map'][step]['success'] for r in rr]
 assert len({r['map'][step]['updates'] for r in rr})==1
 return dict(mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else None,seeds=[r['job']['seed'] for r in rr],values=vals,updates=rr[0]['map'][step]['updates'])

sac=sorted([r for r in rows if r['job']['method']=='sac' and r['map']],key=lambda r:r['job']['seed'])
groups={t:sorted([r for r in rows if r['job']['method']=='ibolt' and r['job']['temperature']==t and r['map']],key=lambda r:r['job']['seed']) for t in temps}
summary=dict(at=d['at'],time_kst=datetime.datetime.fromtimestamp(d['at'],zoneinfo.ZoneInfo('Asia/Seoul')).isoformat(),scope='Slide 20cm; fresh scientific runs only',comparison='Same environment steps, optimizer updates, 10 episodes; different training/evaluation seeds between pilot iBOLT and SAC; exploratory, not paired confirmation.',status=[dict(id=r['job']['id'],method=r['job']['method'],temperature=r['job']['temperature'],seed=r['job']['seed'],status=r['job']['status'],host=r['job']['host'],steps=(r['progress']or{}).get('steps',0)) for r in rows],fixed_points={},latest_per_temperature={})
for step in [50176,100096,200192,300032,501760]:
 ss=stats(sac,step)
 if ss is None:continue
 out={'sac':ss,'ibolt':{}}
 for t,rr in groups.items():
  v=stats(rr,step)
  if v:assert v['updates']==ss['updates']
  out['ibolt'][str(t)]=v
 summary['fixed_points'][str(step)]=out
available=[t for t in temps if groups[t]]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,len(available),figsize=(14,4.9),sharey=True,squeeze=False)
max_common=0
for ax,t in zip(axs[0],available):
 rr=groups[t];common=sorted(set.intersection(*(set(r['map']) for r in rr+sac)))
 for step in common:assert len({r['map'][step]['updates'] for r in rr+sac})==1
 if not common:continue
 x=np.array(common)/1e6;max_common=max(max_common,common[-1]);summary['latest_per_temperature'][str(t)]=dict(step=common[-1],ibolt=stats(rr,common[-1]),sac=stats(sac,common[-1]))
 for data,color,label in [(rr,'#2166AC',f'iBOLT T={t:g} (n={len(rr)})'),(sac,'#D46A16',f'JAX SAC (n={len(sac)})')]:
  y=np.array([[r['map'][s]['success'] for s in common] for r in data]);mean=y.mean(0)
  ax.plot(x,mean,color=color,lw=1.8,label=label,ls='-' if len(data)>1 else '--')
  if len(data)>1:
   sd=y.std(0,ddof=1);ax.fill_between(x,np.clip(mean-sd,0,100),np.clip(mean+sd,0,100),color=color,alpha=.15)
  else:ax.plot(x,mean,color=color,marker='.',ms=3,lw=0)
 ax.set_title(f'Temperature {t:g}',fontweight='bold');ax.set_ylim(-2,102);ax.grid(alpha=.18);ax.set_xlabel('Environment steps (M)');ax.legend(loc='upper left',fontsize=9,frameon=False)
 ax.text(.98,.03,f'Common endpoint: {common[-1]:,}',ha='right',transform=ax.transAxes,fontsize=8,color='#555555')
for ax in axs[0]:ax.set_xlim(0,max_common/1e6*1.03)
axs[0,0].set_ylabel('Final-step success rate (%)')
fig.suptitle('FetchSlide 20 cm | iBOLT temperature pilots vs JAX SAC',fontsize=16,fontweight='bold',y=.99)
waiting=[f'{t:g}' for t in temps if not groups[t]]
subtitle=f"{summary['time_kst'][:16].replace('T',' ')} KST | DACER on | 256 x 2 | horizon 50 | 10 evaluation episodes"
fig.text(.5,.91,subtitle,ha='center',fontsize=10,color='#444444')
fig.text(.5,.035,'Unsmoothed mean +/- sample SD; single-seed lines have no SD. Exact common steps within each panel.\nPilot seeds 5/6; SAC available seeds '+','.join(str(r['job']['seed']) for r in sac)+'. Different train/evaluation seeds: exploratory comparison. Waiting temperatures: '+', '.join(waiting)+'.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.12,1,.88));fig.savefig(P/'slide-20cm-learning-curves.png',dpi=180);fig.savefig(P/'slide-20cm-learning-curves.pdf');plt.close(fig)
(P/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
