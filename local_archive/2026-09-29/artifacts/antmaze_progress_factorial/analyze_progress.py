"""Post-hoc progress reward trajectory audit; never alters training."""
import json,sys,importlib.util
from pathlib import Path
from collections import Counter
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
profiles=['progress_euclidean','progress_euclidean_no_bonus','progress_geodesic','progress_geodesic_no_bonus']
labels=['Euclidean + bonus','Euclidean, no bonus','Geodesic + bonus','Geodesic, no bonus']
rows=[];arrays={}
for host in ['vast-heechan-180','vast-heechan-199']:
 for run in sorted((snapshot/host/'runs').glob('*')):
  cfg=load(run/'config.json');task=cfg['task'];profile=cfg['reward_profile'];hist=[]
  assert cfg['source_commit']=='155fda49f525b5fb71c697c58e0e99e0021e2ca1'
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
    expected=distance(p[0],task,profile)-distance(p[-1],task,profile)-.01*n+bonus
    err.append(abs(float(ret-expected)))
    min_dist.append(float(np.linalg.norm(p[:,None,:]-goals,axis=-1).min()))
    max_travel.append(float(np.linalg.norm(p-p[0],axis=-1).max()))
   assert max(err)<1e-4,(raw,max(err))
   m=dict(step=summary['step'],episodes=len(paths),successes=int(sum(data['goals']>0)),corridors=dict(Counter(families)),mean_closest_goal=float(np.mean(min_dist)),mean_max_displacement=float(np.mean(max_travel)),max_reward_sum_error=max(err),mean_return=float(np.mean(data['returns'])))
   hist.append(m)
  if not hist:continue
  arrays[task,profile]=(paths,families,data['goals'])
  progress=load(run/'progress.json')
  modes={p.parent.name:load(p) for p in sorted(raw.parent.parent.glob('*/summary.json'))}
  rows.append(dict(task=task,profile=profile,progress=progress,budget=cfg['steps'],latest=hist[-1],history=hist,modes=modes))
rows.sort(key=lambda x:(x['task'],profiles.index(x['profile'])))
fig,axes=plt.subplots(2,4,figsize=(16,8.8));fig.subplots_adjust(top=.86,bottom=.10,hspace=.36,wspace=.22)
for r in rows:
 task,profile=r['task'],r['profile'];ax=axes[['v3','v4'].index(task),profiles.index(profile)]
 mod.audit.decorate(ax,task);paths,families,goals=arrays[task,profile]
 for p,fam,goal in zip(paths,families,goals):
  color=mod.COLORS[fam];ax.plot(p[:,0],p[:,1],c=color,lw=.65,alpha=.35)
  ax.scatter(*p[0],c=color,marker='^',s=10,alpha=.55);ax.scatter(*p[-1],c='#bd2942' if not goal else color,marker='x',s=14,alpha=.55)
 m=r['latest'];ax.set_title(f'{task.upper()} | {labels[profiles.index(profile)]}\n{m["step"]/1e6:.2f}M | success {m["successes"]}/{m["episodes"]}\n'+', '.join(f'{k}: {v}' for k,v in m['corridors'].items()),fontsize=9)
fig.suptitle('OptiQ progress reward factorial | latest saved direct-policy rollouts\nT=1, DACER OFF, NovelD OFF | one training seed (0)',fontsize=16,y=.97)
fig.text(.5,.025,'40 random-start episodes per panel; random z + conditional sigma. All failures included.\nTriangles: starts; red crosses: failed endpoints. Different training steps; v1/v2 are still queued.',ha='center',fontsize=10)
fig.savefig(snapshot/'latest_trajectories.png',dpi=150);plt.close(fig)
result=dict(source_commit='155fda49f525b5fb71c697c58e0e99e0021e2ca1',snapshot=str(snapshot),trajectory_reward_sum_verified=True,runs=rows)
(snapshot/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(json.dumps({k:r[k] for k in ('task','profile','latest')},ensure_ascii=False))
