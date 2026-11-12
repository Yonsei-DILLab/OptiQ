from pathlib import Path
import json,time,concurrent.futures,collections,math
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent
CODE='ROOT='+repr(ROOT)+'\n'+r'''
from pathlib import Path
import json,time,subprocess
p=Path(ROOT);plan=json.loads((p/'plan.json').read_text());rows=[]
def read(q):return json.loads(q.read_text()) if q.exists() else None
for j in plan['jobs']:
 if j['phase']!='nm_pilot':continue
 o=Path(j['output']);s=Path('/proc',str(j.get('pid',0)),'stat')
 rows.append(dict(job=j,stat=s.read_text() if s.exists() else None,progress=read(o/'progress.json'),config=read(o/'config.json'),cfg=read(o/('mujoco-config.json' if j['family']=='fetch' else 'ibolt-config.json')),demo=read(o/'demonstration-validation.json'),failure=read(o/'failure.json')))
print(json.dumps(dict(at=time.time(),rows=rows,controller_failure=read(p/'controller-failure.json'),counts=plan['counts'],gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True))))
'''
def one(h):
 for attempt in range(3):
  try:d=ssh(h,CODE);break
  except RuntimeError:
   if attempt==2:raise
 d['host']=h;d['runs']=[]
 for r in d.pop('rows'):
  j=r['job'];c=r['config'];cfg=r['cfg'];proof=r['demo'];pr=r['progress'] or {};verified=False
  if c and cfg:
   assert cfg['dacer']['enabled'] and cfg['dacer']['target_entropy_per_dim']==-.9
   assert cfg['alg']['actor']['num_policy_samples']==j['num_policy_samples'] and cfg['alg']['actor']['proposals_per_policy_sample']==4
   assert c['num_reference_actions']==j['num_reference_actions'] and cfg['temperature']==cfg['alg']['actor']['temperature']==j['temperature']
   assert c['num_envs']==c['batch_size']==256 and c['utd']==(.125 if j['family']=='fetch' else .25)
   assert cfg['alg']['actor']['hidden_dims']==cfg['alg']['critic']['hs']==[256,256]
   assert bool(cfg['alg']['critic'].get('layer_norm',False))==(j['family']=='adroit')
   if j['family']=='adroit':
    assert c['demonstration_dataset_sha256']==j['demo_sha256']
    assert proof and proof['passed'] and proof['replayed'] and proof['successful_trajectories']==25
   if j.get('protocol_variant')=='F1-slide-20cm':assert c['distance_threshold_m']==.2
   verified=True
  metrics=pr.get('metrics',{});assert all(not isinstance(v,(float,int)) or math.isfinite(v) for v in metrics.values())
  st=r['stat'].rsplit(')',1)[1].split() if r['stat'] else None;live=bool(st and st[0]!='Z' and int(st[19])==j['start_ticks'])
  d['runs'].append(dict(id=j['id'],status=j['status'],gpu=j.get('gpu'),pid=j.get('pid'),live=live,config_verified=verified,steps=pr.get('steps',0),updates=pr.get('updates',0),metrics=metrics,failure=r['failure']))
 (P/f'status-host{h}.json').write_text(json.dumps(d,indent=2));return d
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:hosts=list(pool.map(one,[1,2,3]))
 runs=[r for h in hosts for r in h['runs']];assert len({r['id'] for r in runs})==len(runs)
 out=dict(at=time.time(),seen=len(runs),counts=dict(collections.Counter(r['status'] for r in runs)),verified=sum(r['config_verified'] for r in runs),updating=sum(r['updates']>0 for r in runs),hosts=[dict(host=h['host'],counts=dict(collections.Counter(r['status'] for r in h['runs'])),gpu_counts=dict(collections.Counter(r['gpu'] for r in h['runs'] if r['status']=='running')),gpus=h['gpus'],steps=[min((r['steps'] for r in h['runs']),default=0),max((r['steps'] for r in h['runs']),default=0)],controller_failure=h['controller_failure'],failures=[r for r in h['runs'] if r['failure'] or r['status']=='failed']) for h in hosts])
 (P/'status.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
