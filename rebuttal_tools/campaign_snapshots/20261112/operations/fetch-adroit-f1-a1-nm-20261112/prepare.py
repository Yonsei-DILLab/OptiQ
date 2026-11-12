from pathlib import Path
import json,hashlib,shutil,ast,difflib,copy,time
P=Path(__file__).resolve().parent; E=P.parent/'fetch-adroit-f1-a1-entropy-20261112'; T=P.parent/'fetch-adroit-f1-a1-temp-20261112'
R='/workspace/heechan-unified-f1-a1-20261112'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def replace_once(s,old,new):
 assert s.count(old)==1,(old,s.count(old));return s.replace(old,new)
proofs={}
for variant in ['fetch','fetch-slide-20cm','adroit']:
 old=E/('entropy-'+variant+'-source');new=P/('nm-'+variant+'-source');new.mkdir(exist_ok=False)
 manifest=json.loads((old/'source-sha256.json').read_text())
 for f,h in manifest.items():
  assert sha(old/f)==h,f
  (new/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/f,new/f)
 worker='adroit_rebuttal/run.py' if variant=='adroit' else 'rebuttal_tools/fetch_suite_5m_worker.py'
 p=new/worker;s=p.read_text()
 validation="n, m = int(spec['num_policy_samples']), int(spec['num_reference_actions'])\n    assert (n, m) in [(64, 256), (16, 64)] and m % n == 0\n    repeats = m // n"
 if variant!='adroit':
  s=replace_once(s,"    target = float(spec['dacer_target_entropy_per_dim'])",'    '+validation+"\n    target = float(spec['dacer_target_entropy_per_dim'])")
  s=replace_once(s,"assert target in [-.9, 0., .25, .5, .75]","assert target == -.9")
  s=replace_once(s,'settings.update(temperature=temperature, log_std_min=-5.', 'settings.update(temperature=temperature, num_policy_samples=n, num_reference_actions=m, proposals_per_policy_sample=repeats, log_std_min=-5.')
  s=s.replace("parameter_selection='Frozen selected T; DACER entropy pilot seeds 5/6; independent confirmation seeds 0-4'", "parameter_selection='5M temperature-pilot selected T; DACER -0.9; N/M ablation seeds 5/6'")
  s=replace_once(s,'            assert cfg.alg.actor.num_policy_samples==64 and cfg.alg.actor.proposals_per_policy_sample==1','            cfg.alg.actor.num_policy_samples=n; cfg.alg.actor.proposals_per_policy_sample=repeats\n            assert cfg.alg.actor.num_policy_samples * cfg.alg.actor.proposals_per_policy_sample == m')
  s=replace_once(s,"'wandb.group','dacer.target_entropy_per_dim'}", "'wandb.group','dacer.target_entropy_per_dim','alg.actor.num_policy_samples','alg.actor.proposals_per_policy_sample'}")
 else:
  s=replace_once(s,"            target = float(spec['dacer_target_entropy_per_dim'])", '            '+validation.replace('\n    ','\n            ')+"\n            target = float(spec['dacer_target_entropy_per_dim'])")
  s=replace_once(s,"assert target in [-.9, 0., .25, .5, .75]","assert target == -.9")
  s=replace_once(s,'            cfg.dacer.target_entropy_per_dim = target', '            cfg.dacer.target_entropy_per_dim = target\n            cfg.alg.actor.num_policy_samples=n\n            cfg.alg.actor.proposals_per_policy_sample=repeats')
  s=replace_once(s,'settings.update(temperature=temperature, num_policy_samples=64, proposals_per_policy_sample=1,','settings.update(temperature=temperature, num_policy_samples=n, num_reference_actions=m, proposals_per_policy_sample=repeats,')
  oldline="                        observed_unix=time.time(), metrics=metrics, discovery=self.model.replay_buffer.discovery()))"
  newline=oldline+"\n                    if self.n_calls % 40 == 0:\n                        with (output/'training-metrics.jsonl').open('a') as stream:\n                            stream.write(json.dumps(dict(steps=self.num_timesteps, updates=int(self.model._n_updates), metrics=metrics), allow_nan=False)+'\\n')"
  s=replace_once(s,oldline,newline)
 p.write_text(s);ast.parse(s)
 p=new/'ibolt/algorithm.py';s=p.read_text()
 s=replace_once(s,'                actor_std_max=std.max())','                actor_std_max=std.max(),\n                importance_ess_mean=(1. / jnp.sum(weights ** 2, axis=-1)).mean(),\n                importance_max_weight_mean=weights.max(axis=-1).mean(),\n                proposal_q_std=q.std(axis=-1).mean(),\n                proposal_log_density_std=density.std(axis=-1).mean())')
 p.write_text(s);ast.parse(s)
 updated={f:sha(new/f) for f in manifest}
 changed=[f for f in manifest if manifest[f]!=updated[f]]
 assert set(changed)=={worker,'ibolt/algorithm.py'}
 # No update formula, randomness or action sampling changes: metrics-only algorithm diff.
 (new/'source-sha256.json').write_text(json.dumps(updated,indent=2)+'\n')
 (P/(variant+'.diff')).write_text(''.join(''.join(difflib.unified_diff((old/f).read_text().splitlines(True),(new/f).read_text().splitlines(True),fromfile='old/'+f,tofile='new/'+f)) for f in changed))
 proofs[variant]=dict(parent_manifest_sha256=sha(old/'source-sha256.json'),manifest_sha256=sha(new/'source-sha256.json'),changed_files=changed)
pre={h:json.loads((P/f'preflight-host{h}.json').read_text()) for h in range(1,6)}
alljobs=[j for d in pre.values() for j in d['plan']['jobs']]
selects=pre[1]['selects']; rows=[];references=[]
for family,tasks in [('fetch',['reach','push','pick-and-place','slide']),('adroit',['door','hammer','pen','relocate'])]:
 for task in tasks:
  key=family+'-'+task;temp=selects[key]['temperature'];variant='fetch-slide-20cm' if key=='fetch-slide' else family
  for seed in [5,6]:
   base=next(j for j in alljobs if j['family']==family and j['task']==task and j['method']=='ibolt' and j['phase']=='pilot' and j['seed']==seed and j['temperature']==temp)
   assert base['status']=='completed'
   references.append(dict(id=base['id'],host=base['host'],family=family,task=task,seed=seed,temperature=temp,N=64,M=64))
   for n,m in [(64,256),(16,64)]:
    if family=='fetch':host=1 if seed==5 else 2
    else:host=2 if task=='relocate' else 3
    rid=f'{family}-{task}'+('-r20cm' if key=='fetch-slide' else '')+f'-ibolt-n{n}-m{m}-s{seed}-pilot'
    drop=['pid','start_ticks','gpu','gpu_uuid','learner_cpus','worker_cpus','started_unix','progress','result','config_verified','placement_history','exit_code','failure']
    j={k:copy.deepcopy(v) for k,v in base.items() if k not in drop}
    j.update(id=rid,host=host,phase='nm_pilot',campaign='fetch-adroit-f1-a1-nm-20261112',status='queued',source=R+'/nm-'+variant+'-source',source_manifest=R+'/nm-'+variant+'-source/source-sha256.json',source_manifest_sha256=proofs[variant]['manifest_sha256'],output=R+'/runs/'+rid,temperature=temp,dacer_target_entropy_per_dim=-.9,num_policy_samples=n,num_reference_actions=m,proposals_per_policy_sample=4,baseline_run_id=base['id'],temperature_selection=selects[key])
    assert j['steps']==5000192 and j['depth']==2
    if family=='adroit':assert pre[host]['plan']['datasets'][task]['sha256']==j['demo_sha256'] if 'sha256' in pre[host]['plan']['datasets'][task] else True
    rows.append(j)
assert len(rows)==32 and len({j['id'] for j in rows})==32
protocol=dict(campaign='fetch-adroit-f1-a1-nm-20261112',created_unix=time.time(),root=R,central=json.loads((T/'protocol.json').read_text())['central'],dacer_on=True,dacer_target_entropy_per_dim=-.9,N_M=[[64,256],[16,64]],pilot_seeds=[5,6],expected_steps=5000192,new_runs=32,temperature_rule='Original complete 5M temperature-pilot last-five common regular evaluation selection; unchanged per-task winner',temperatures={k:v['temperature'] for k,v in selects.items()},temperature_selections=selects,baseline_references=references,unchanged={k:json.loads((T/'protocol.json').read_text())[k] for k in ['f1','a1']},sources=proofs,diagnostics='Read-only ESS, maximum importance weight, proposal Q and log-density spread; no extra random draws or update-rule change',selection='Exploratory N/M comparison only; no automatic best-setting selection or confirmation launch authorized',existing_runs='Preserve existing running learners and all prior results; DACER rollback applies to these new runs',rows=rows)
save('protocol.json',protocol)
for h in [1,2,3]:
 save(f'additions-host{h}.json',[j for j in rows if j['host']==h])
 s=pre[h]['controller']
 marker="   assert cfg['dacer']['enabled'] and cfg['temperature']==cfg['alg']['actor']['temperature']==j['temperature']"
 addition="   if j['phase']=='nm_pilot':\n    assert cfg['dacer']['target_entropy_per_dim']==-.9\n    assert cfg['alg']['actor']['num_policy_samples']==j['num_policy_samples']\n    assert cfg['alg']['actor']['proposals_per_policy_sample']==4\n    assert cfg['alg']['actor']['num_policy_samples']*4==j['num_reference_actions']\n    assert c['num_reference_actions']==j['num_reference_actions']\n"
 s=replace_once(s,marker,addition+marker)
 marker="  def gate(j):\n"
 addition="   if j['phase']=='nm_pilot':\n    proof=read(ROOT/'nm-archive-ready.json',{})\n    if not proof.get('passed') or proof.get('source_manifests',{}).get(Path(j['source']).name)!=j['source_manifest_sha256']:return False\n"
 s=replace_once(s,marker,marker+addition)
 s=replace_once(s," if j['phase'].startswith('entropy_'):assert hashlib.sha256", " if j['phase'].startswith('entropy_') or j['phase']=='nm_pilot':assert hashlib.sha256")
 ast.parse(s);(P/f'controller-host{h}.py').write_text(s)
save('preparation-check.json',dict(passed=True,run_count=32,reference_count=16,source_diffs=proofs,counts={h:sum(j['host']==h for j in rows) for h in [1,2,3]}))
print(json.dumps(dict(passed=True,runs=32,temperatures=protocol['temperatures'],hosts={h:sum(j['host']==h for j in rows) for h in [1,2,3]})))
