from pathlib import Path
import json,ast,subprocess,collections
P=Path(__file__).resolve().parent;proto=json.loads((P/'protocol.json').read_text())
tree=ast.parse((P.parent/'fetch-adroit-f1-a1-temp-20261112/refresh.py').read_text())
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=remote,text=True,capture_output=True,timeout=50)
assert r.returncode==0,r.stderr
snap=json.loads(r.stdout);wanted={j['id'] for j in proto['rows']}|{j['baseline_run_id'] for j in proto['rows']}
snap['rows']=[r for r in snap['rows'] if r['job']['id'] in wanted];(P/'snapshot.json').write_text(json.dumps(snap,indent=2))
rows={r['job']['id']:r for r in snap['rows']}
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axs=plt.subplots(2,4,figsize=(16,7.5),sharey=True);report=[]
keys=[('fetch',x) for x in ['reach','push','pick-and-place','slide']]+[('adroit',x) for x in ['door','hammer','pen','relocate']]
for ax,(family,task) in zip(axs.flat,keys):
 jobs=sorted([j for j in proto['rows'] if j['family']==family and j['task']==task],key=lambda j:j['seed'])
 groups=[[rows.get(j['id']) for j in jobs],[rows.get(j['baseline_run_id']) for j in jobs]]
 stats=[];finals=[]
 if all(r is not None for g in groups for r in g):
  maps=[[{e['steps']:e for e in r['curve'] if e['episodes']==10} for r in g] for g in groups]
  shared=sorted(set.intersection(*(set(m) for g in maps for m in g)))
  for step in shared:
   assert len({m[step]['updates'] for g in maps for m in g})==1
   assert len({m[step]['primary_mode'] for g in maps for m in g})==1
  for label,mm,g in zip(['SAC alpha=0','SAC automatic alpha'],maps,groups):
   if shared:
    y=np.array([[m[t]['success']*100 for t in shared] for m in mm]);mean=y.mean(0);sd=y.std(0,ddof=1)
    line=ax.plot(np.array(shared)/1e6,mean,label=label,lw=1.4)[0];ax.fill_between(np.array(shared)/1e6,np.maximum(0,mean-sd),np.minimum(100,mean+sd),alpha=.15,color=line.get_color())
    stats.append(dict(method=label,step=shared[-1],seed_success=y[:,-1].tolist(),mean=float(mean[-1]),sample_sd=float(sd[-1]),last5_mean=float(y[:,-5:].mean())))
   fe=[next((e for e in r['curve'] if e['steps']==5000192 and e['episodes']==100),None) for r in g]
   if all(e is not None for e in fe):
    y=np.array([e['success']*100 for e in fe]);ret=np.array([e['return_'] for e in fe])
    finals.append(dict(method=label,step=5000192,episodes_per_seed=100,seed_success=y.tolist(),success_mean=float(y.mean()),success_sample_sd=float(y.std(ddof=1)),raw_return_mean=float(ret.mean()),raw_return_sample_sd=float(ret.std(ddof=1))))
 if not stats:ax.text(.5,.5,'Waiting for paired evaluations',ha='center',va='center',transform=ax.transAxes)
 else:ax.legend(fontsize=8,loc='lower right')
 ax.set_title(f'{family.title()} {task.title()}'+(' (20 cm)' if task=='slide' else ''));ax.set_xlim(0,5);ax.set_ylim(0,100);ax.grid(alpha=.18);ax.set_xlabel('Environment steps (million)')
 report.append(dict(family=family,task=task,common_regular=stats,final=finals))
for ax in axs[:,0]:ax.set_ylabel('Success rate (%)')
fig.suptitle('JAX SAC fixed alpha=0 versus automatic alpha | F1 / A1 | seeds 0, 1\nExact common evaluation steps; mean ± sample SD (two-seed exploratory comparison)',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(P/'alpha0-comparison.png',dpi=180);fig.savefig(P/'alpha0-comparison.pdf')
new=[rows[j['id']] for j in proto['rows'] if j['id'] in rows]
summary=dict(at=snap['at'],expected=16,seen=len(new),counts=dict(collections.Counter(r['job']['status'] for r in new)),tasks=report)
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
