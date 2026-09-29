"""Read-only audit of saved random-start direct-policy progress100 rollouts."""
import json,sys,importlib.util
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
from antmaze_experiments.progress_reward import distance,maze_geometry,bonus_enabled
spec=importlib.util.spec_from_file_location('details',ROOT.parent/'antmaze_dacer_off_t1/report_completed.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
snapshot=Path((ROOT/'latest-report-path.txt').read_text().strip())
load=lambda p:json.loads(p.read_text())
profiles=['progress100_euclidean','progress100_euclidean_no_bonus','progress100_geodesic','progress100_geodesic_no_bonus']
labels=['Euclidean + bonus','Euclidean, no bonus','Geodesic + bonus','Geodesic, no bonus']
rows=[];arrays={}
for host in ['vast-heechan-180','vast-heechan-199','vast1']:
 for run in sorted((snapshot/host/'runs').glob('*')):
  if not (run/'config.json').exists():continue
  cfg=load(run/'config.json');task=cfg['task'];profile=cfg['reward_profile'];hist=[]
  assert cfg['source_commit']=='0751e86a547dc8f1d82e4861c2931cc6f9cf04b8'
  for raw in sorted(run.glob('evaluations/*/policy-natural/rollouts.npz')):
   data=np.load(raw);summary=load(raw.parent/'summary.json')
   paths=[xy[:int(n)+1] for xy,n in zip(data['xy'],data['lengths'])]
   assert len(paths)==summary['episodes'] and all(np.isfinite(x).all() for x in paths)
   assert not summary['fixed'];assert np.unique(data['initial_full_state'],axis=0).shape[0]>1
   assert np.isclose(np.mean(data['goals']>0),summary['success_rate'])
   families=[mod.family(task,p) for p in paths]
   goals=maze_geometry(task)[1];min_dist=[];max_travel=[];err=[]
   for p,n,g,ret in zip(paths,data['lengths'],data['goals'],data['returns']):
    bonus=(20 if task=='v2' and g==1 else 10) if g and bonus_enabled(profile) else 0
    expected=100*(distance(p[0],task,profile)-distance(p[-1],task,profile))-n+bonus
    err.append(abs(float(ret-expected)))
    min_dist.append(float(np.linalg.norm(p[:,None,:]-goals,axis=-1).min()))
    max_travel.append(float(np.linalg.norm(p-p[0],axis=-1).max()))
   assert max(err)<.01,(raw,max(err))
   hist.append(dict(step=summary['step'],episodes=len(paths),successes=int(sum(data['goals']>0)),corridors=dict(Counter(families)),successful_corridors=dict(Counter(f for f,g in zip(families,data['goals']) if g)),goal_counts={str(int(k)):int(v) for k,v in Counter(data['goals']).items()},mean_closest_goal=float(np.mean(min_dist)),mean_max_displacement=float(np.mean(max_travel)),max_reward_sum_error=max(err),mean_return=float(np.mean(data['returns'])),raw=str(raw)))
  if hist:arrays[task,profile]=(paths,families,data['goals'])
  progress=load(run/'progress.json') if (run/'progress.json').exists() else {}
  rows.append(dict(task=task,profile=profile,progress=progress,budget=cfg['steps'],latest=hist[-1] if hist else None,history=hist))
rows.sort(key=lambda x:(x['task'],profiles.index(x['profile'])))
tasks=sorted({r['task'] for r in rows if r['latest']})
fig,axes=plt.subplots(len(tasks),4,figsize=(16,4.3*len(tasks)+.5),squeeze=False)
fig.subplots_adjust(top=.86,bottom=.10,hspace=.37,wspace=.22)
for task in tasks:
 for profile in profiles:
  ax=axes[tasks.index(task),profiles.index(profile)];mod.audit.decorate(ax,task)
  r=next((r for r in rows if (r['task'],r['profile'])==(task,profile)),None)
  if not r or not r['latest']:
   ax.set_title(f'{task.upper()} | {labels[profiles.index(profile)]}\nAwaiting first saved evaluation',fontsize=10);continue
  paths,families,goals=arrays[task,profile]
  for p,fam,goal in zip(paths,families,goals):
   color=mod.COLORS[fam];ax.plot(p[:,0],p[:,1],c=color,lw=.7,alpha=.4)
   ax.scatter(*p[0],c=color,marker='^',s=10,alpha=.55);ax.scatter(*p[-1],c='#bd2942' if not goal else color,marker='x',s=14,alpha=.6)
  m=r['latest'];ax.set_title(f'{task.upper()} | {labels[profiles.index(profile)]}\n{m["step"]/1e6:.2f}M | success {m["successes"]}/{m["episodes"]}\n'+', '.join(f'{k}: {v}' for k,v in m['corridors'].items()),fontsize=9)
fig.suptitle('OptiQ: 100 x distance progress - 1 + unscaled goal bonus\nLatest direct-policy rollouts | T=1, DACER OFF, NovelD OFF | seed 0',fontsize=15,y=.975)
fig.text(.5,.022,'40 random-start episodes per panel; random z + conditional sigma; every failure included.\nTriangles = starts; red crosses = unsuccessful endpoints. Panels may use different training steps.',ha='center',fontsize=10)
fig.savefig(snapshot/'latest_trajectories.png',dpi=150)
plt.close(fig)
for task in tasks:
 fig,axs=plt.subplots(1,4,figsize=(16,5))
 fig.subplots_adjust(top=.76,bottom=.12,left=.055,right=.99,wspace=.24)
 for ax,profile in zip(axs,profiles):
  mod.audit.decorate(ax,task)
  r=next((r for r in rows if (r['task'],r['profile'])==(task,profile)),None)
  if not r or not r['latest']:
   ax.set_title(labels[profiles.index(profile)]+'\nAwaiting first evaluation',fontsize=10);continue
  paths,families,goals=arrays[task,profile]
  for p,fam,goal in zip(paths,families,goals):
   color=mod.COLORS[fam];ax.plot(p[:,0],p[:,1],c=color,lw=.7,alpha=.4)
   ax.scatter(*p[0],c=color,marker='^',s=10,alpha=.55)
   ax.scatter(*p[-1],c='#bd2942' if not goal else color,marker='x',s=14,alpha=.6)
  m=r['latest'];ax.set_title(labels[profiles.index(profile)]+f'\n{m["step"]/1e6:.2f}M | success {m["successes"]}/40\n'+', '.join(f'{k}: {v}' for k,v in m['corridors'].items()),fontsize=9)
 fig.suptitle(f'{task.upper()} | OptiQ progress x100 | T=1, DACER/NovelD OFF, seed0\n40 random-start direct-policy rollouts; every failure included; different training steps',fontsize=12,y=.98)
 fig.savefig(snapshot/f'{task}_trajectories.png',dpi=150);plt.close(fig)
result=dict(source_commit='0751e86a547dc8f1d82e4861c2931cc6f9cf04b8',snapshot=str(snapshot),trajectory_reward_sum_verified=True,runs=rows)
(snapshot/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(json.dumps({k:r[k] for k in ('task','profile','progress','latest')},ensure_ascii=False))
