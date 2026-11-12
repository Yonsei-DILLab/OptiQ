from pathlib import Path
import json,itertools,hashlib,shutil
P=Path(__file__).resolve().parent;REMOTE=Path('/workspace/heechan-unified-f1-a1-20261112')
TEMPS=[.05,.1,.25,.5,1.];TASKS={'fetch':['reach','push','pick-and-place','slide'],'adroit':['door','hammer','pen','relocate']}
PYTHONS={3:{'fetch':'/workspace/heechan-fetch-dacer/venv/bin/python','adroit':'/workspace/heechan-adroit/venv-capacity-repeat-20261111/bin/python'},4:{'fetch':'/workspace/heechan-fetch-sigma/venv-sac-cu128-20261110/bin/python','adroit':'/workspace/heechan-adroit/venv-capacity-repeat-20261111/bin/python'},5:{'adroit':'/workspace/heechan-adroit/hammer-pen-utd025-20261111/venv/bin/python'}}
data=json.loads((P/'datasets-original.json').read_text())
for task,d in data.items():
 for key in ['demo_path','demonstration_manifest']:d[key]=str(REMOTE/'datasets'/task/Path(d[key]).name)
proof=Path('outputs/adroit-preparation-20261110/adroit-jax-sac-validation-20261112/reports/adroit-validation.json')
shutil.copyfile(proof,P/'jax-validation.json');v=json.loads(proof.read_text());assert v['passed']
assert v['module_sha256']==hashlib.sha256((P/'fetch-source/baselines/jax_sac.py').read_bytes()).hexdigest()
plans={n:dict(host=n,root=str(REMOTE),python=py,jobs=[],status='prepared',selection_rule='mean of seed 5/6 last five common regular primary success evaluations; tie uses ascending temperature',max_jobs_per_gpu=3,min_free_gib=5 if n==5 else 12 if n==3 else 8,source_manifests={f:hashlib.sha256((P/(f+'-source')/'source-sha256.json').read_bytes()).hexdigest() for f in TASKS},datasets=data,expected_steps=5000192,profile=dict(actor=[256,256],critic=[256,256],vec=256,batch=256,warmup=12800,learning_rate=.0003,gamma=.99,tau=.005,dacer=True)) for n,py in PYTHONS.items()}
rows=[]
def add(family,task,method,seed,temp,phase,host=None):
 if host is None:
  hosts=[3,4] if family=='fetch' else [3,4,5,5]
  host=hosts[(TASKS[family].index(task)+seed+(TEMPS.index(temp) if temp is not None else 0))%len(hosts)]
 suffix='' if temp is None else '-t'+str(temp).replace('.','p')
 name=f'{family}-{task}-{method}-s{seed}{suffix}-{phase}'
 spec=dict(id=name,family=family,task=task,method=method,seed=seed,temperature=temp if temp is not None else .25,phase=phase,host=host,depth=2,steps=5000192,status='waiting_selection' if phase=='confirm' else 'queued',output=str(REMOTE/('preflights' if phase=='smoke' else 'runs')/name),source=str(REMOTE/(family+'-source')),source_manifest=str(REMOTE/(family+'-source')/'source-sha256.json'),ibolt_reference_config=str(REMOTE/'fetch-reference.json'),validation_manifest=str(REMOTE/'jax-validation.json'))
 if family=='adroit':spec.update(utd=.25,critic_layer_norm=True,paired_final_evaluation=True,**data[task])
 plans[host]['jobs'].append(spec);rows.append(spec)
# Simulator integration checks for each corrected-HER Fetch task and each method.
for task,method in itertools.product(TASKS['fetch'],['ibolt','sac']):add('fetch',task,method,105,.25 if method=='ibolt' else None,'smoke',3 if method=='ibolt' else 4)
# Pilot seeds are disjoint from confirmation and SAC comparison seeds.
for ti,temp in enumerate(TEMPS):
 for family,tasks in TASKS.items():
  for task,seed in itertools.product(tasks,[5,6]):add(family,task,'ibolt',seed,temp,'pilot')
for family,tasks in TASKS.items():
 for seed,task in itertools.product(range(5),tasks):add(family,task,'sac',seed,None,'baseline')
 # Exactly one temperature per task will be released by the central selector.
 for temp,seed,task in itertools.product(TEMPS,range(5),tasks):add(family,task,'ibolt',seed,temp,'confirm')
# Reuse only the original completed paired pilots whose corrected-HER protocol matches F1.
runs=json.loads(Path('outputs/fetch-adroit-inventory-20261112/registry.json').read_text())['runs']
for job in rows:
 if job['family']!='fetch' or job['task']!='slide' or job['phase']!='pilot' or job['temperature'] not in [.1,.25,.5]:continue
 campaign='fetch-slide-her-normalization-5m-20261110T182512Z' if job['temperature']==.25 else 'fetch-slide-her-temperature-5m-20261110T205349Z'
 candidates=[r for r in runs if r['campaign']==campaign and r['method']=='ibolt' and r['seed']==job['seed'] and r['temperature']==job['temperature']]
 assert len(candidates)==1
 r=candidates[0];assert r['state']=='completed' and r['steps']==5000192 and r['utd']==.125 and r['actor']==r['critic']==[256,256] and r['dacer']['enabled']
 # Put source-linked evidence on host3 where the original run lives; no new learner.
 plans[job['host']]['jobs'].remove(job);job.update(host=3,status='reused',reuse_output=r['output']);plans[3]['jobs'].append(job)
for host,plan in plans.items():
 smokes=[j for j in plan['jobs'] if j['phase']=='smoke']; pilots=[j for j in plan['jobs'] if j['phase']=='pilot']; baselines=[j for j in plan['jobs'] if j['phase']=='baseline']; confirms=[j for j in plan['jobs'] if j['phase']=='confirm']
 plan['jobs']=smokes+[j for pair in itertools.zip_longest(pilots,baselines) for j in pair if j is not None]+confirms
 (P/f'plan-host{host}.json').write_text(json.dumps(plan,indent=2))
protocol=dict(campaign=P.name,root=str(REMOTE),central='heechan@dil.yonsei.ac.kr:2200:/data1/heechan/iBOLT/20261112-f1-a1-temp',temperatures=TEMPS,pilot_seeds=[5,6],confirmation_seeds=list(range(5)),expected_steps=5000192,dacer_on=True,sac='shared baselines.jax_sac.JaxSAC; automatic entropy; no alpha grid',f1=dict(actor=[256,256],critic=[256,256],critic_layer_norm=False,utd=.125,num_envs=256,batch=256,horizon=50,her='future 204/256; corrected completed-episode observation and goal statistics',logstd=[-5,1],dacer_noise_scale=2.,primary_evaluation='fresh stochastic z, conditional mean; final-step success'),a1=dict(actor=[256,256],critic=[256,256],critic_layer_norm=True,utd=.25,num_envs=256,batch=256,horizon=200,demonstrations='25 unique expert episodes per task; on-disk Sparse-v2 replay verified',demo_online=[128,128],bc_updates=0,her=False,observation_normalization=False,logstd=[-5,-1],dacer_noise_scale=.1,primary_evaluation='z=0, conditional mean; any episode success'),selection_rule=plans[3]['selection_rule'],reuse='Six completed F1 Slide pilots at T=.1/.25/.5, seeds5/6; fresh remaining74 pilots and all40 JAX SAC runs',rows=rows)
(P/'protocol.json').write_text(json.dumps(protocol,indent=2))
print({h:{p:sum(j['phase']==p and j['status']!='reused' for j in x['jobs']) for p in ['smoke','pilot','baseline','confirm']} for h,x in plans.items()})
