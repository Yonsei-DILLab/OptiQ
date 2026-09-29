from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,str(Path('tmp/reward-progress-worktree').resolve()))
from antmaze_experiments.progress_reward import distance,EUCLIDEAN_NO_COST_PROFILE
root=Path(__file__).resolve().parent
rows=[]
for file in sorted(root.glob('rollouts-*.npz')):
 data=np.load(file);step=int(data['env_steps']);episodes=[]
 for xy,n,goal,ret in zip(data['xy'],data['lengths'],data['goals'],data['returns']):
  p=xy[:int(n)+1].astype(float);d=distance(p,'v3',EUCLIDEAN_NO_COST_PROFILE);r=100*(d[:-1]-d[1:]);g=float(np.sum(r*.99**np.arange(int(n))))
  identity=100*(d[0]-.01*np.sum(.99**np.arange(int(n)-1)*d[1:-1])-.99**(int(n)-1)*d[-1]);assert abs(g-identity)<1e-8
  assert abs(r.sum()-ret)<.001
  left=(p[:,0]<-8).any();right=(p[:,0]>8).any();side='both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
  episodes.append(dict(side=side,success=bool(goal),steps=int(n),return_undiscounted=float(ret),return_gamma099=g,start_distance=float(d[0]),end_distance=float(d[-1]),distance_at100=float(d[min(100,len(d)-1)]),positive_reward_sum=float(r[r>0].sum()),negative_reward_sum=float(r[r<0].sum())))
 groups={}
 for side in sorted(set(e['side'] for e in episodes)):
  group=[e for e in episodes if e['side']==side]
  groups[side]={'n':len(group),'successes':sum(e['success'] for e in group)}
  for key in ['steps','return_undiscounted','return_gamma099','start_distance','end_distance','distance_at100','positive_reward_sum','negative_reward_sum']:
   values=[e[key] for e in group];groups[side][key]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else 0.}
 rows.append(dict(step=step,groups=groups,episodes=episodes))
(root/'discounted_return_analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
 print(row['step'])
 for side,g in row['groups'].items(): print(side,json.dumps(g))
