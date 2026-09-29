from pathlib import Path
import json,importlib.util,sys,csv
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('routes',REPO/'artifacts/antmaze_dacer_off_t1/report_completed.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[];panels={}
for label in ['training','audit/runs','watch']:
 for p in sorted((ROOT/'data'/label).glob('*/**/evaluations/*/policy-*/rollouts.npz')):
  # Training is one level shallower than checkpoint-specific evaluation folders.
  summary=json.loads(p.with_name('summary.json').read_text());data=np.load(p)
  task=next(part[:2] for part in p.parts if part.startswith(('v3-optiq','v4-optiq')))
  reset='random' if label=='training' else 'fixed'
  if reset=='fixed':
   np.testing.assert_array_equal(data['initial_full_state'],np.repeat(data['initial_full_state'][:1],len(data['lengths']),axis=0))
   np.testing.assert_array_equal(data['initial_full_state'][:,:2],np.zeros((len(data['lengths']),2)))
  paths=[xy[:int(n)+1] for xy,n in zip(data['xy'],data['lengths'])];families=[mod.family(task,xy) for xy in paths]
  goals=data['goals'];detail={}
  for f in sorted(set(families)):
   inds=[i for i,x in enumerate(families) if x==f]
   detail[f]=dict(count=len(inds),successes=int(sum(goals[i]>0 for i in inds)),mean_length=float(np.mean(data['lengths'][inds])),mean_return=float(np.mean(data['returns'][inds])))
  r=dict(task=task,step=summary['step'],reset=reset,episodes=len(paths),corridors=dict(Counter(families)),successes=int(sum(goals>0)),groups=detail,input=str(p.relative_to(ROOT)),unique_initial_states=len(np.unique(data['initial_full_state'],axis=0)))
  rows.append(r);panels[task,r['step'],reset]=(paths,goals,families,r)
rows=sorted({(r['task'],r['step'],r['reset']):r for r in rows}.values(),key=lambda x:(x['task'],x['reset'],x['step']))
logs={}
for task in ['v3','v4']:
 p=ROOT/'data/training'/f'{task}-optiq-progress100_geodesic_no_bonus-T1-s0/learner/progress.csv'
 with p.open() as f:rr=list(csv.DictReader(f))
 keys=['train/temperature','train/backup_entropy_term','train/actor_std_mean','train/actor_std_at_max_fraction','train/source_ess_absolute','train/max_source_weight','train/source_q_std','train/actor_between_mean_variance']
 logs[task]=[dict(step=int(float(r['time/total_timesteps'])),**{k:float(r[k]) for k in keys}) for r in rr]
 assert all(r['train/temperature']==1 and r['train/backup_entropy_term']==0 for r in logs[task])
result=dict(runs=rows,learner_logs=logs,geometry=json.loads((ROOT/'geometry_probe.json').read_text()),notes=['Direct policy random z + conditional sigma; 40 episodes per saved checkpoint.','Current reward is 100*nearest-goal geodesic progress - 1, B=0.','Fixed origin/full state matches v3/v4 learning resets.','Random resets are historical supplementary only.','First stored checkpoint ~250k; no claim about earlier policy behavior.','Current training replay/XY are not yet finalized; old dense reward forgetting evidence is a different experiment.'])
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2))
for r in rows:print(r['task'],r['step'],r['reset'],r['corridors'],r['successes'])
selected={}
for task in ['v3','v4']:
 fixed=sorted([r for r in rows if r['task']==task and r['reset']=='fixed'],key=lambda x:x['step'])
 if len(fixed)<3:continue
 mid=min(fixed,key=lambda r:abs(r['step']-(750000 if task=='v3' else 1000000)))
 selected[task]=[fixed[0],mid,fixed[-1]]
if len(selected)==2:
 fig,axs=plt.subplots(2,3,figsize=(14,9));fig.subplots_adjust(top=.87,bottom=.11,hspace=.32,wspace=.24)
 for i,task in enumerate(['v3','v4']):
  for ax,r in zip(axs[i],selected[task]):
   mod.audit.decorate(ax,task);paths,goals,families,_=panels[task,r['step'],'fixed']
   for path,goal,f in zip(paths,goals,families):
    ax.plot(path[:,0],path[:,1],color=mod.COLORS[f],alpha=.43,lw=.8)
    if not goal:ax.scatter(*path[-1],c='#bd2942',marker='x',s=12,alpha=.7)
   ax.scatter(0,0,c='black',marker='*',s=70,zorder=10)
   ax.set_title(f"{task.upper()} | {r['step']/1e6:.2f}M | success {r['successes']}/40\n"+', '.join(f'{k}: {v}' for k,v in r['corridors'].items()),fontsize=10)
 fig.suptitle('Did both routes persist from the same start?\nOptiQ | 100 x geodesic progress - 1 | B=0, T=1 | fixed original full state',fontsize=15,y=.97)
 fig.text(.5,.025,'Direct policy: random z + conditional sigma. Seed 0; 40 episodes per checkpoint; failures included (red crosses).\nEarliest checkpoint is 250k. Historical random-start branches are not evidence of same-state route diversity.',ha='center',fontsize=10)
 fig.savefig(ROOT/'fixed_history.png',dpi=150);plt.close(fig)
