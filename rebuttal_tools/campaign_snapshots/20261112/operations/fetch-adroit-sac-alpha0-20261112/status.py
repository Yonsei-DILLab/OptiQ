from pathlib import Path
import sys,json,time,concurrent.futures,collections,math
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-nm-20261112'));from preflight import ssh,ROOT
CODE='ROOT='+repr(ROOT)+'\n'+r'''
from pathlib import Path
import json,time,sys
p=Path(ROOT);sys.path.insert(0,str(p));import controller
plan=controller.read(p/'plan.json');rows=[]
for j in plan['jobs']:
 if j['phase']!='sac_alpha0_pilot':continue
 o=Path(j['output']);c=controller.read(o/'config.json',{});pr=controller.read(o/'progress.json',{})
 verified=False
 if 'entropy_coefficient' in c and (j['family']=='adroit' or (o/'architecture-audit.json').exists()):
  assert c['entropy_coefficient']==0. and not c['automatic_entropy_tuning']
  assert c['batch_size']==c['num_envs']==256 and c['utd']==(.125 if j['family']=='fetch' else .25)
  if j['family']=='adroit':
   assert c['critic_layer_norm'] and c['actor_hidden']==c['critic_hidden']==[256,256]
   assert c['demonstration_dataset_sha256']==j['demo_sha256']
   proof=controller.read(o/'demonstration-validation.json');assert proof['passed'] and proof['replayed'] and proof['successful_trajectories']==25
  else:
   a=controller.read(o/'architecture-audit.json',{})
   assert a.get('backend')=='gpu' and a['actor_hidden_dims']==a['critic_hidden_dims']==[256,256] and not a['layer_norm']
   if j['task']=='slide':assert c['distance_threshold_m']==.2
  verified=True
 alpha_evidence=None
 if pr.get('updates',0)>0:
  if 'train/ent_coef' in pr.get('metrics',{}):
   assert pr['metrics']['train/ent_coef']==0.
   alpha_evidence='live_training_metric'
  else:
   # Successful completion replaces progress.json with a terminal summary;
   # Adroit intentionally omits the transient training metrics in that summary.
   assert verified and pr.get('completed') and pr['steps']==5000192
   assert pr['updates']==int((5000192-12800)*c['utd'])
   prefix='final' if j['family']=='adroit' else 'policy_005000192'
   metadata=controller.read(o/'checkpoints'/(prefix+'.jax.json'))
   assert metadata['steps']==pr['steps'] and metadata['updates']==pr['updates'] and metadata['backend']=='gpu'
   assert (o/'checkpoints'/(prefix+'.jax.msgpack')).is_file()
   alpha_evidence='completed_fixed_alpha_config_and_checkpoint_metadata'
 live=j.get('pid') is not None and controller.identity(j['pid'])==j.get('start_ticks')
 rows.append(dict(id=j['id'],host=j['host'],task=j['task'],seed=j['seed'],status=j['status'],gpu=j.get('gpu'),pid=j.get('pid'),live=live,verified=verified,steps=pr.get('steps',0),updates=pr.get('updates',0),metrics=pr.get('metrics',{}),alpha_evidence=alpha_evidence,failure=controller.read(o/'failure.json')))
print(json.dumps(dict(at=time.time(),host=plan['host'],runs=rows,controller_failure=controller.read(p/'controller-failure.json'))))
'''
def one(h):
 d=ssh(h,CODE);(P/f'status-host{h}.json').write_text(json.dumps(d,indent=2));return d
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:hosts=list(pool.map(one,[3,4]))
rows=[r for h in hosts for r in h['runs']]
result=dict(at=time.time(),expected=16,seen=len(rows),counts=dict(collections.Counter(r['status'] for r in rows)),verified=sum(r['verified'] for r in rows),updating=sum(r['updates']>0 for r in rows),runs=rows)
(P/'status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
