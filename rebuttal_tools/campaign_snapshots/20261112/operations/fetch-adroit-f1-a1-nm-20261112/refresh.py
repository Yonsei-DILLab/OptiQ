"""Read centralized evidence; compare only exact common regular evaluation steps."""
from pathlib import Path
import json,ast,subprocess,collections,time
P=Path(__file__).resolve().parent;OLD=P.parent/'fetch-adroit-f1-a1-temp-20261112'
tree=ast.parse((OLD/'refresh.py').read_text());remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=remote,text=True,capture_output=True,timeout=50);assert not r.returncode,r.stderr
snap=json.loads(r.stdout);(P/'snapshot.json').write_text(json.dumps(snap,indent=2));proto=json.loads((P/'protocol.json').read_text());rows={r['job']['id']:r for r in snap['rows']}
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axs=plt.subplots(2,4,figsize=(17,8),sharey=True);report=[]
keys=['fetch-'+t for t in ['reach','push','pick-and-place','slide']]+['adroit-'+t for t in ['door','hammer','pen','relocate']]
for ax,key in zip(axs.flat,keys):
 fam,task=key.split('-',1);groups=[]
 for n,m in [(64,64),(64,256),(16,64)]:
  refs=proto['baseline_references'] if (n,m)==(64,64) else [j for j in proto['rows'] if (j['num_policy_samples'],j['num_reference_actions'])==(n,m)]
  rr=[rows[j['id']] for j in refs if j['family']==fam and j['task']==task and j['id'] in rows]
  groups.append((f'iBOLT N={n}, M={m} (s5/6)',rr,[5,6]))
 sac=[r for r in rows.values() if r['job']['family']==fam and r['job']['task']==task and r['job']['phase']=='baseline']
 groups.append(('JAX SAC (s0–4)',sac,list(range(5))))
 valid=len(groups)==4 and all(sorted(r['job']['seed'] for r in rr)==seeds for _,rr,seeds in groups)
 maps=[[{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in rr] for _,rr,_ in groups]
 shared=sorted(set.intersection(*(set(m) for mm in maps for m in mm))) if valid else []
 stats=[]
 if shared:
  for t in shared:assert len({m[t]['updates'] for mm in maps for m in mm})==1
  for (label,rr,seeds),mm in zip(groups,maps):
   y=np.array([[m[t]['success']*100 for t in shared] for m in mm]);x=np.array(shared)/1e6
   mean=y.mean(0);sd=y.std(0,ddof=1);line=ax.plot(x,mean,lw=1.4,label=label)[0];ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=line.get_color(),alpha=.12)
   stats.append(dict(label=label,latest_common_step=shared[-1],success_mean=mean[-1],success_sd=sd[-1],seeds=seeds))
  ax.legend(fontsize=6.5,loc='lower right')
 else:ax.text(.5,.5,'Waiting for all seed pairs',ha='center',va='center',transform=ax.transAxes)
 ax.set_title(key+(' (20 cm)' if task=='slide' else '')+f" | T={proto['temperatures'][key]:g}");ax.set_xlim(0,5);ax.set_ylim(0,100);ax.grid(alpha=.15);ax.set_xlabel('Environment steps (M)')
 report.append(dict(task=key,statistics=stats))
for ax in axs[:,0]:ax.set_ylabel('Success rate (%)')
fig.suptitle('N/M ablation | DACER on, H/d = −0.9 | mean ± sample SD\niBOLT seeds 5/6; SAC seeds 0–4: exploratory comparison, not paired-seed confirmation',fontsize=12)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(P/'nm-comparison.png',dpi=170);fig.savefig(P/'nm-comparison.pdf');plt.close(fig)
new=[rows[j['id']] for j in proto['rows'] if j['id'] in rows];counts=collections.Counter(r['job']['status'] for r in new)
result=dict(at=snap['at'],expected_new_runs=32,seen=len(new),counts=dict(counts),tasks=report)
(P/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
