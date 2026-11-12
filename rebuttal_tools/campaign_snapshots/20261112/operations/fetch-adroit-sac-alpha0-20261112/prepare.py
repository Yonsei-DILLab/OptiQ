from pathlib import Path
import json,hashlib,shutil,copy,time,ast,tarfile,difflib
P=Path(__file__).resolve().parent; W=P.parents[1]
T=P.parent/'fetch-adroit-f1-a1-temp-20261112'
R='/workspace/heechan-unified-f1-a1-20261112'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
sources={}
for variant in ['fetch','fetch-slide-20cm','adroit']:
 old=T/('slide-20cm-20261112/fetch-slide-20cm-source' if variant=='fetch-slide-20cm' else variant+'-source')
 new=P/('alpha0-'+variant+'-source');new.mkdir(exist_ok=False)
 manifest=json.loads((old/'source-sha256.json').read_text())
 for f,h in manifest.items():
  assert sha(old/f)==h,f
  (new/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/f,new/f)
 for f in ['baselines/jax_sac.py','rebuttal_tools/check_fetch_jax_sac.py']:
  (new/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(W/f,new/f)
 worker='adroit_rebuttal/run.py' if variant=='adroit' else 'rebuttal_tools/fetch_jax_sac_worker.py'
 f=new/worker;s=f.read_text()
 if variant=='adroit':
  s=replace(s,'assert np.isfinite(ent_coef) and ent_coef > 0','assert np.isfinite(ent_coef) and ent_coef == 0')
  s=replace(s,"ent_coef = spec.get('sac_ent_coef', 'auto')","ent_coef = float(spec['sac_ent_coef']); assert ent_coef == 0.")
 else:
  s=replace(s,"    temperature = spec['temperature']","    temperature = spec['temperature']\n    ent_coef = float(spec['sac_ent_coef']); assert ent_coef == 0.")
  s=replace(s,"ent_coef='auto', policy_kwargs",'ent_coef=ent_coef, policy_kwargs')
  s=replace(s,"entropy_coefficient='auto'","entropy_coefficient=ent_coef, automatic_entropy_tuning=False")
  s=replace(s,'float(np.exp(np.asarray(model.jax_state.log_alpha))[0])','0.0')
  s=replace(s,"entropy_lr=3e-4","entropy_lr=None")
  s=replace(s,"update_order='alpha (old alpha reused), critic, actor with updated critic, Polyak target'","update_order='fixed alpha=0; critic, actor with updated critic, Polyak target'")
  s=replace(s,"entropy='automatic'","entropy='fixed', fixed_alpha=ent_coef")
  s=replace(s,"        primary_success='Final-step", "        entropy_coefficient=ent_coef, automatic_entropy_tuning=False,\n        primary_success='Final-step")
  marker="            from baselines.jax_sac import export_actor, export_critic"
  s=replace(s,marker,"            assert model.jax_state.zero_alpha and model.jax_state.alpha_opt is None\n            assert model.ent_coef_optimizer is None and float(model.ent_coef_tensor)==0.\n"+marker)
  marker="            metrics={}"
  s=replace(s,marker,"            assert model.jax_state.zero_alpha and model.jax_state.alpha_opt is None\n            assert model.logger.name_to_value['train/ent_coef'] == 0.\n"+marker)
 f.write_text(s);ast.parse(s)
 manifest={f:sha(new/f) for f in set(manifest)|{'baselines/jax_sac.py','rebuttal_tools/check_fetch_jax_sac.py'}}
 (new/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 sources[variant]=dict(manifest_sha256=sha(new/'source-sha256.json'),parent_manifest_sha256=sha(old/'source-sha256.json'))
 changed=[f for f in manifest if not (old/f).exists() or sha(old/f)!=sha(new/f)]
 (P/(variant+'.diff')).write_text(''.join(''.join(difflib.unified_diff((old/f).read_text().splitlines(True),(new/f).read_text().splitlines(True),fromfile='old/'+f,tofile='new/'+f)) for f in changed if (old/f).exists()))
plans=[json.loads((P/f'before-host{h}.json').read_text())['plan'] for h in [3,4]]
# Completed controls may be on a different host; original deployment snapshots preserve their IDs.
plans += [json.loads(p.read_text())['plan'] for p in (P.parent/'fetch-adroit-f1-a1-nm-20261112').glob('preflight-host*.json')]
jobs=[]
for family,tasks in [('fetch',['reach','push','pick-and-place','slide']),('adroit',['door','hammer','pen','relocate'])]:
 for task in tasks:
  for seed in [0,1]:
   refs={j['id']:j for p in plans for j in p['jobs'] if j['family']==family and j['task']==task and j['method']=='sac' and j['seed']==seed and j['phase'] not in ['smoke'] and (task!='slide' or j.get('distance_threshold_m')==.2)}
   assert len(refs)==1,(family,task,seed,list(refs))
   base=next(iter(refs.values()));h=4 if family=='fetch' else 3;variant='fetch-slide-20cm' if task=='slide' else family
   rid=f'{family}-{task}'+('-r20cm' if task=='slide' else '')+f'-jax-sac-alpha0-s{seed}-pilot'
   drop={'pid','start_ticks','gpu','gpu_uuid','learner_cpus','worker_cpus','started_unix','progress','result','config_verified','placement_history','exit_code','failure','reused_from','reuse_output'}
   j={k:copy.deepcopy(v) for k,v in base.items() if k not in drop}
   j.update(id=rid,host=h,phase='sac_alpha0_pilot',campaign=P.name,status='queued',sac_ent_coef=0.,source=R+'/alpha0-'+variant+'-source',source_manifest=R+'/alpha0-'+variant+'-source/source-sha256.json',source_manifest_sha256=sources[variant]['manifest_sha256'],output=R+'/runs/'+rid,validation_manifest=R+'/sac-alpha0-20261112/validation-'+family+'.json',baseline_run_id=base['id'],baseline_host=base['host'])
   assert j['steps']==5000192 and j['depth']==2
   jobs.append(j)
proto=dict(campaign=P.name,created_unix=time.time(),steps=5000192,seeds=[0,1],new_runs=16,alpha=0.,automatic_entropy_tuning=False,sources=sources,rows=jobs,selection='Exploratory fixed-alpha ablation; no seed replacement or winner selection',unchanged='F1/A1, same expert demos, LN setting, HER, normalization, UTD, evaluation and network as matching automatic-alpha SAC controls')
(P/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
for h in [3,4]:(P/f'additions-host{h}.json').write_text(json.dumps([j for j in jobs if j['host']==h],indent=2)+'\n')
print('Prepared',len(jobs),'runs',sources)
