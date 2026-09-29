from pathlib import Path
import json,sys,importlib.util,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('routes',ROOT.parent/'antmaze_dacer_off_t1/report_completed.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
out=Path((ROOT/'latest-report-path.txt').read_text().strip())
records=[];panels={}
for host in ['vast-heechan-180','vast-heechan-199','vast1']:
 for resultfile in (out/host/'runs').glob('*/step_*/result.json'):
  folder=resultfile.parent;prov=json.loads((folder/'provenance.json').read_text());verification=json.loads((folder/'verification.json').read_text());assert verification['passed']
  for name,digest in verification['raw_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
  step=prov['checkpoint_steps'];task=prov['task'];job=folder.parent.name
  row=dict(task=task,job=job,step=step,evaluation_source=prov['evaluation_source'],training_source=prov['training_source'],modes={})
  for label,path in [('fixed',folder/'evaluations'/f'{step:010d}'/'policy-fixed'),('random',folder/'paired_random'),('native_fixed',folder/'evaluations'/f'{step:010d}'/'native-fixed')]:
   if not (path/'summary.json').exists():continue
   summary=json.loads((path/'summary.json').read_text());data=np.load(path/'rollouts.npz')
   xy=[p[:int(n)+1] for p,n in zip(data['xy'],data['lengths'])];goals=data['goals'];families=[mod.family(task,p) for p in xy]
   unique=len(np.unique(data['initial_full_state'],axis=0))
   if label!='random':
    assert unique==1
    np.testing.assert_array_equal(data['initial_full_state'][:,:2],np.zeros((len(xy),2)))
   else:assert unique>1
   metric=dict(episodes=len(xy),successes=int(sum(goals>0)),corridors=dict(Counter(families)),successful_corridors=dict(Counter(f for f,g in zip(families,goals) if g)),unique_initial_states=unique,goal_counts={str(int(k)):int(v) for k,v in Counter(goals).items()})
   assert metric['successes']/len(xy)==summary['success_rate']
   row['modes'][label]=metric;panels[job,label]=(xy,goals,families,metric)
  records.append(row)
selected=sorted([r for r in records if 'no-step' in r['job']],key=lambda r:r['task'])
if selected:
 fig,axs=plt.subplots(2,len(selected),figsize=(4.5*len(selected),9),squeeze=False)
 fig.subplots_adjust(top=.86,bottom=.10,hspace=.36,wspace=.19)
 for col,r in enumerate(selected):
  for i,label in enumerate(['fixed','random']):
   ax=axs[i,col];mod.audit.decorate(ax,r['task'])
   if (r['job'],label) not in panels:ax.set_title('Saved random evaluation pending');continue
   paths,goals,families,metric=panels[r['job'],label]
   for p,g,f in zip(paths,goals,families):
    color=mod.COLORS[f];ax.plot(p[:,0],p[:,1],c=color,lw=.85,alpha=.45)
    ax.scatter(*p[0],c=color,marker='^',s=9,alpha=.5)
    ax.scatter(*p[-1],c=color if g else '#c43150',marker='x',s=14,alpha=.7)
   if label=='fixed':ax.scatter(0,0,c='black',marker='*',s=65,zorder=10)
   title='FIXED: original full state' if label=='fixed' else 'RANDOM: historical supplementary'
   ax.set_title(f"{r['task'].upper()} | {r['step']/1e6:.2f}M | {title}\nSuccess {metric['successes']}/{metric['episodes']}\n"+', '.join(f'{k} {v}' for k,v in metric['corridors'].items()),fontsize=9)
 fig.suptitle('OptiQ geodesic progress x100 | B=0, step penalty=0\nSame checkpoint: fixed vs random reset | Direct policy (random z + conditional sigma)',fontsize=13,y=.97)
 fig.text(.5,.025,'40 episodes per panel; seed0. All failures included. Stars = fixed origin; triangles = starts; red crosses = failed endpoints.\nV1 retains native random resets and is not included in this reset correction.',ha='center',fontsize=9)
 fig.savefig(out/'fixed_vs_random.png',dpi=155);plt.close(fig)
(out/'analysis.json').write_text(json.dumps(dict(runs=records,all_fixed_starts_verified=True,source_eval='061e9888e1f3d0e59b5037441fa146c1eff25e2a'),indent=2))
for r in records:print(json.dumps(r,ensure_ascii=False))
print(out)
