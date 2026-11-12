"""Refresh the existing campaign, then report the sequential entropy comparison."""
from pathlib import Path
import json,subprocess,sys,collections
P=Path(__file__).resolve().parent;OLD=P.parent/'fetch-adroit-f1-a1-temp-20261112'
subprocess.run([sys.executable,str(OLD/'refresh.py')],check=True,capture_output=True,text=True)
code="""from pathlib import Path
import json,time
p=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp')
print(json.dumps(dict(at=time.time(),selections={f.stem:json.loads(f.read_text()) for f in (p/'entropy-selections').glob('*.json')},temperatures={f.stem:json.loads(f.read_text()) for f in (p/'entropy-temperature-selections').glob('*.json')},coordinator=json.loads((p/'entropy-coordinator-status.json').read_text()))))
"""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code,text=True,capture_output=True,timeout=30);assert r.returncode==0,r.stderr
metadata=json.loads(r.stdout);snapshot=json.loads((OLD/'snapshot.json').read_text());protocol=json.loads((P/'protocol.json').read_text())
snapshot['entropy']=metadata;(P/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def series(rows,seeds):
 if sorted(r['job']['seed'] for r in rows)!=seeds:return None
 maps=[{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in rows]
 shared=sorted(set.intersection(*(set(m) for m in maps)))
 if not shared:return None
 assert all(len({m[t]['updates'] for m in maps})==1 for t in shared)
 y=np.array([[m[t]['success']*100 for t in shared] for m in maps]);return np.array(shared),y.mean(0),y.std(0,ddof=1)
tasks=protocol['task_keys'];report=[]
for kind in ['pilot','comparison']:
 fig,axs=plt.subplots(2,4,figsize=(17,8),sharey=True)
 for ax,key in zip(axs.flat,tasks):
  family,task=key.split('-',1);selected=metadata['temperatures'].get(key) or snapshot['selections'].get(key);ent=metadata['selections'].get(key)
  rows=[r for r in snapshot['rows'] if r['job']['family']==family and r['job']['task']==task]
  groups=[]
  if kind=='pilot' and selected:
   for h in protocol['target_entropy_per_dim']:
    matches=[r for r in rows if r['job']['temperature']==selected['temperature'] and (r['job']['phase']=='pilot' if h==-.9 else r['job']['phase']=='entropy_pilot' and r['job']['dacer_target_entropy_per_dim']==h)]
    groups.append((f'H/d={h:g}',series(matches,[5,6])))
  if kind=='comparison' and ent:
   original=snapshot['selections'].get(key)
   phase='confirm' if ent['target_entropy_per_dim']==-.9 and original and original['temperature']==ent['temperature'] else 'entropy_confirm'
   groups=[('iBOLT (selected T, H)',series([r for r in rows if r['job']['phase']==phase],list(range(5)))),('JAX SAC',series([r for r in rows if r['job']['phase']=='baseline'],list(range(5))))]
   if any(v is None for _,v in groups):groups=[]
  ready=[(label,s) for label,s in groups if s is not None]
  if ready:
   shared=set.intersection(*(set(s[0]) for _,s in ready))
   for label,(x,y,sd) in ready:
    mask=np.isin(x,list(shared));x,y,sd=x[mask]/1e6,y[mask],sd[mask]
    line=ax.plot(x,y,label=label,lw=1.4)[0];ax.fill_between(x,np.maximum(0,y-sd),np.minimum(100,y+sd),color=line.get_color(),alpha=.12)
   ax.legend(fontsize=7)
  else:ax.text(.5,.5,'Waiting for selected T / matched runs',ha='center',va='center',transform=ax.transAxes,fontsize=9)
  ax.set_title(key+(' (20 cm)' if task=='slide' else '')+(f" | T={selected['temperature']:g}" if selected else ''))
  ax.set_xlim(0,5);ax.set_ylim(0,100);ax.set_xlabel('Environment steps (M)');ax.grid(alpha=.15)
  if kind=='pilot':report.append(dict(task=key,temperature=selected['temperature'] if selected else None,entropy_target=ent['target_entropy_per_dim'] if ent else None,entropy_status=dict(collections.Counter(r['job']['status'] for r in rows if r['job']['phase'].startswith('entropy_')))))
 for ax in axs[:,0]:ax.set_ylabel('Success rate (%)')
 fig.suptitle('DACER target entropy | '+('Pilot seeds 5/6' if kind=='pilot' else 'Independent seeds 0–4 versus JAX SAC')+' | mean ± sample SD')
 fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(P/(kind+'-curves.png'),dpi=170);fig.savefig(P/(kind+'-curves.pdf'));plt.close(fig)
result=dict(at=metadata['at'],tasks=report,coordinator=metadata['coordinator']);(P/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
