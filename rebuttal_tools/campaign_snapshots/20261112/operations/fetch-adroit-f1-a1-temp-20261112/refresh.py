"""Read central evidence only and render matched-seed, matched-step curves."""
from pathlib import Path
import json,subprocess,statistics
P=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import json,time
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');rows=[];hosts=[]
def read(p):return json.loads(p.read_text()) if p.exists() else None
for h in [1,2,3,4,5]:
 base=root/f'hosts/host{h}/workspace/heechan-unified-f1-a1-20261112';plan=read(base/'plan.json')
 if not plan:continue
 hosts.append(dict(host=h,counts=plan.get('counts'),updated_unix=plan.get('updated_unix'),free_gib=plan.get('free_gib'),sync=read(root/f'live-host{h}.json'),controller_failure=read(base/'controller-failure.json')))
 for j in plan['jobs']:
  if j['phase']=='smoke' or j['status'] in ['waiting_selection','not_selected']:continue
  p=base/'runs'/j['id'];curve=[];f=p/'evaluations.jsonl'
  if f.exists():
   for line in f.read_text().splitlines():
    try:e=json.loads(line)
    except json.JSONDecodeError:continue
    v=e[e['primary_mode']];eps=v['episodes'];n=len(eps) if isinstance(eps,list) else eps
    curve.append(dict(steps=e['steps'],updates=e['updates'],episodes=n,success=v['success_rate'],return_=v['mean_return'],primary_mode=e['primary_mode']))
  final=read(p/'final-evaluation.json')
  if final:
   v=final[final['primary_mode']];eps=v['episodes'];curve.append(dict(steps=final['steps'],updates=final['updates'],episodes=len(eps) if isinstance(eps,list) else eps,success=v['success_rate'],return_=v['mean_return'],primary_mode=final['primary_mode']))
  rows.append(dict(job=j,progress=read(p/'progress.json'),completion=read(p/('completed.json' if j['family']=='adroit' else 'verified-completion.json')),config=read(p/'config.json'),curve=curve))
print(json.dumps(dict(at=time.time(),hosts=hosts,rows=rows,selections={f.stem:read(f) for f in (root/'selections').glob('*.json')},live_summary=read(root/'live-summary.json'))))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=remote,capture_output=True,text=True,timeout=45);assert not r.returncode,r.stderr
snapshot=json.loads(r.stdout);(P/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
TASKS=[('fetch',x) for x in ['reach','push','pick-and-place','slide']]+[('adroit',x) for x in ['door','hammer','pen','relocate']]

def series(rows,seeds):
 if sorted(r['job']['seed'] for r in rows)!=seeds:return None
 maps=[{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in rows]
 common=sorted(set.intersection(*(set(m) for m in maps)))
 if not common:return None
 for t in common:assert len({m[t]['updates'] for m in maps})==1
 y=np.array([[m[t]['success']*100 for t in common] for m in maps]);return np.array(common)/1e6,y.mean(0),y.std(0,ddof=1)

for kind in ['pilot','comparison']:
 fig,axs=plt.subplots(2,4,figsize=(17,8),sharey=True)
 for ax,(family,task) in zip(axs.flat,TASKS):
  if kind=='pilot':groups=[(f'T={t:g}',[r for r in snapshot['rows'] if r['job']['family']==family and r['job']['task']==task and r['job']['phase']=='pilot' and r['job']['temperature']==t],[5,6]) for t in [.05,.1,.25,.5,1.]]
  else:groups=[(label,[r for r in snapshot['rows'] if r['job']['family']==family and r['job']['task']==task and r['job']['phase']==phase],list(range(5))) for label,phase in [('iBOLT (selected T)','confirm'),('JAX SAC','baseline')]]
  ready=[(label,series(rr,seeds)) for label,rr,seeds in groups];ready=[(a,b) for a,b in ready if b is not None]
  # Method comparison is clipped to the exact common evaluation grid of both five-seed groups.
  if kind=='comparison' and len(ready)!=2:ready=[]
  if kind=='comparison' and ready:
   shared=set(ready[0][1][0])&set(ready[1][1][0]);ready=[(l,tuple(np.asarray(v)[np.isin(s[0],list(shared))] for v in s)) for l,s in ready]
  for label,(x,y,sd) in ready:
   line=ax.plot(x,y,label=label,linewidth=1.4)[0];ax.fill_between(x,np.maximum(0,y-sd),np.minimum(100,y+sd),color=line.get_color(),alpha=.13)
  if not ready:ax.text(.5,.5,'Waiting for matched seeds',ha='center',va='center',transform=ax.transAxes,color='#777777')
  ax.set_title(f'{family.title()} {task.title()}'+(' (20 cm)' if family=='fetch' and task=='slide' else ''));ax.set_ylim(0,100);ax.set_xlim(0,5);ax.grid(alpha=.18);ax.set_xlabel('Environment steps (M)')
  if ready:ax.legend(fontsize=8,loc='lower right')
 for ax in axs[:,0]:ax.set_ylabel('Success rate (%)')
 fig.suptitle(('Pilot: seeds 5 and 6, mean ± sample SD' if kind=='pilot' else 'Independent confirmation: seeds 0–4, matched-step mean ± sample SD')+' | DACER on | 256×2',fontsize=15)
 fig.tight_layout(rect=[0,0,.999,.95]);fig.savefig(P/(kind+'-curves.png'),dpi=170);fig.savefig(P/(kind+'-curves.pdf'));plt.close(fig)
print(json.dumps(dict(hosts=snapshot['hosts'],selections=snapshot['selections'],curves=str(P/'pilot-curves.png')),ensure_ascii=False))
