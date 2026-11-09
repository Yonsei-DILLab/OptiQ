from pathlib import Path
import json,math,subprocess
import numpy as np
from pointmaze_rebuttal.envs.drac_paper import PaperPointMaze
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='heechan-rebuttal'
base=Path('/workspace/iBOLT/outputs/pointmaze-rebuttal/utd1-b256-sequential-20261107T161144Z')
reports=[]
for maze_dir in base.iterdir():
 if not maze_dir.is_dir() or maze_dir.name not in ['medium','hard']:continue
 env=PaperPointMaze(maze_dir.name)
 try:
  for path in sorted(maze_dir.glob('*-ibolt-*')):
   cfg=json.loads((path/'config.json').read_text());seed=cfg['seed']
   with np.load(path/'checkpoints/final-replay.npz') as d:
    size,pos=int(d['size']),int(d['position']);lost=cfg['steps']-size
    assert lost==192 and pos==192 and cfg['num_envs']==256 and size==1000000
    actions=np.random.default_rng(seed).uniform(-1.,1.,(256,2)).astype(np.float32)
    initial=[];following=[];reward=[];terminal=[]
    for i,action in enumerate(actions):
     obs,_=env.reset(seed=seed+100004*i)
     next_obs,r,term,trunc,_=env.step(action)
     initial.append(np.asarray(obs,np.float32));following.append(np.asarray(next_obs,np.float32))
     reward.append(r);terminal.append(float(term or trunc))
    initial=np.array(initial);following=np.array(following);reward=np.array(reward,np.float32)
    for key,value in [('observations',initial),('actions',actions),('next_observations',following),('rewards',reward),('terminals',np.array(terminal,np.float32))]:
     np.testing.assert_array_equal(d[key][lost:256],value[lost:256])
    raw=d['rewards'].reshape(-1);ordered=np.concatenate((raw[pos:],raw[:pos]))
    full=np.concatenate((reward[:lost],ordered));assert len(full)==cfg['steps']
    hits=np.flatnonzero(full>0)
    first=int(math.ceil((int(hits[0])+1)/256)*256) if len(hits) else None
    indices=(pos+(hits-lost))%size
    assert np.all(hits>=lost)
    xy=d['next_observations'][indices,:2]
    distances=np.max(np.abs(xy[:,None,:]-env.goal_positions[None,:,:]),axis=2)
    labels=distances.argmin(axis=1);assert np.all(distances[np.arange(len(labels)),labels]<=.50001)
    counts=np.bincount(labels,minlength=len(env.goal_positions))
    final=json.loads((path/'evaluations/001000192_summary.json').read_text())
    m=final['mu_only'];prob=np.array(m['goals'],float)/sum(m['goals'])
    entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0])))
    reports.append(dict(maze=cfg['maze'],seed=seed,initial_missing_transitions=lost,prefix_rewards_nonzero=int(np.count_nonzero(reward[:lost])),
     prefix_reconstruction_verified_against_retained_first_batch=True,total_transitions=len(full),
     reward_discovered=bool(len(hits)),first_reward_at_collection_end=first,
     warmup_reward_events=int(np.count_nonzero(full[:cfg['warmup']]>0)),
     training_reward_events=len(hits),training_reward_transition_fraction=float(len(hits)/len(full)),
     training_goal_counts=counts.tolist(),training_goal_coverage=int(np.count_nonzero(counts)),
     evaluation_mode='fresh-z mu_only; conditional Gaussian noise disabled',
     evaluation_success=m['success'],evaluation_goals=m['goals'],evaluation_coverage=m['reachable_goals'],
     evaluation_normalized_goal_entropy=entropy/math.log(len(prob)),evaluation_effective_goals=math.exp(entropy),
     evaluation_sr5_removal=m['sr5_removal']))
 finally:env.close()
report=dict(source='completed iBOLT final replay and 500-episode final evaluation',reconstruction='Regenerated the initial random collection using the original seed and physics; all five fields match the 64 retained rows exactly. Restored the overwritten first 192 transitions. First-reward time is expressed as total transitions at end of a 256-env collection.',runs=reports)
out=base/f"review-metrics-audit-20261108T045742Z-{reports[0]['maze']}.json"
assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

