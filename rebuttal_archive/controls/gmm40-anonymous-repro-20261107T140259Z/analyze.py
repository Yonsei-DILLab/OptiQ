"""Objective comparisons of two fresh runs, with profile-transfer and GPU-placement checks kept separate."""
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from gmm40.metrics import mmd2
C=json.loads((ROOT/'campaign.json').read_text())
OLD=ROOT.parent/'gmm40-hidden-init-m35-20261107T133528Z'/'orthogonal_sqrt2'
CACHE={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_run(p):
 if p in CACHE:return CACHE[p]
 if not (p/'completed.json').exists():return None
 cfg=json.loads((p/'config.json').read_text())
 assert json.loads((p/'completed.json').read_text())==dict(actor_updates=100000,completed=True)
 for k,v in C['settings'].items():assert cfg[k]==v,(p,k,cfg[k])
 rows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
 assert len(rows)==16 and rows[-1]['actor_updates']==100000
 final=p/'evaluations/step_0100000'
 # The pre-publication backup stores the final evaluation in a final/ directory.
 if not final.exists():final=p/'final'
 reference=np.load(p/'reference.npy')
 modes={}
 for mode in ('full_policy','mu_only'):
  metric=json.loads((final/f'metrics_{mode}.json').read_text())
  assert metric==rows[-1][mode]
  x=np.load(final/f'samples_{mode}.npy');assert x.shape==(10000,2) and np.isfinite(x).all()
  value,_=mmd2(x,reference)
  assert np.isclose(np.sqrt(max(value,0)),metric['mmd'],atol=1e-12,rtol=0)
  modes[mode]=dict(samples=x,metric=metric)
 result=dict(config=cfg,reference=reference,modes=modes)
 CACHE[p]=result;return result

def compare(before,after,kind,seed):
 a,b=read_run(before),read_run(after)
 if a is None or b is None:return None
 for k,v in a['config'].items():
  if k!='runtime':assert b['config'][k]==v,(kind,seed,k)
 row=dict(kind=kind,seed=seed,before=str(before.relative_to(ROOT.parent)),after=str(after.relative_to(ROOT.parent)),
          initial_checkpoint_identical=sha(before/'checkpoints/step_0000000.bin')==sha(after/'checkpoints/step_0000000.bin'),
          final_checkpoint_identical=sha(before/'checkpoints/step_0100000.bin')==sha(after/'checkpoints/step_0100000.bin'),
          reference_identical=bool(np.array_equal(a['reference'],b['reference'])))
 assert row['reference_identical']
 for mode in ('full_policy','mu_only'):
  u,v=a['modes'][mode],b['modes'][mode]
  diff=v['samples'].astype(float)-u['samples'].astype(float)
  row[mode]=dict(samples_bitwise_identical=bool(np.array_equal(u['samples'],v['samples'])),max_abs_sample_difference=float(np.abs(diff).max()),rms_sample_difference=float(np.sqrt(np.mean(diff**2))),
                 mmd_before=u['metric']['mmd'],mmd_after=v['metric']['mmd'],mmd_difference=v['metric']['mmd']-u['metric']['mmd'],
                 coverage_before=u['metric']['mode_coverage'],coverage_after=v['metric']['mode_coverage'],metrics_identical=u['metric']==v['metric'])
 if kind!='different_gpu_same_commit':
  for filename,label in [('checkpoint-comparison.json','checkpoints'),('parameter-comparison.json','parameters')]:
   if (after/filename).exists():row[label]=json.loads((after/filename).read_text())
 return row

transfer=[];strict=[];cross=[]
for seed in range(5):
 a=compare(OLD/f'seed-{seed}',ROOT/'same_gpu'/f'seed-{seed}','profile_transfer_different_commits',seed)
 if a:transfer.append(a)
 a=compare(ROOT/'same_gpu'/f'seed-{seed}',ROOT/'repeat_same_gpu'/f'seed-{seed}','same_commit_same_gpu',seed)
 if a:strict.append(a)
for seed in range(3):
 a=compare(ROOT/'same_gpu'/f'seed-{seed}',ROOT/'other_gpu'/f'seed-{seed}','different_gpu_same_commit',seed)
 if a:cross.append(a)
 a=compare(ROOT/'other_gpu'/f'seed-{seed}',ROOT/'repeat_other_gpu'/f'seed-{seed}','same_commit_same_gpu_extra_placement',seed)
 if a:strict.append(a)

def describe(rows):
 return dict(comparisons=len(rows),initial_checkpoints_identical=sum(r['initial_checkpoint_identical'] for r in rows),final_checkpoints_identical=sum(r['final_checkpoint_identical'] for r in rows),full_policy_sample_arrays_identical=sum(r['full_policy']['samples_bitwise_identical'] for r in rows),full_policy_metric_dicts_identical=sum(r['full_policy']['metrics_identical'] for r in rows),max_abs_mmd_difference=max([abs(r['full_policy']['mmd_difference']) for r in rows],default=None))
def aggregate(role):
 runs=[read_run(ROOT/role/f'seed-{s}') for s in range(5)]
 if any(r is None for r in runs):return None
 a=np.array([r['modes']['full_policy']['metric']['mmd'] for r in runs])
 return dict(mean_mmd=float(a.mean()),sample_sd_mmd=float(a.std(ddof=1)),coverage=[r['modes']['full_policy']['metric']['mode_coverage'] for r in runs])
primary=[r for r in strict if r['kind']=='same_commit_same_gpu']
summary=dict(published_revision=C['revision'],complete_primary_pairs=len(primary)==5,complete_all_pairs=len(strict)==8,primary_same_commit_same_gpu=describe(primary),all_same_commit_same_gpu=describe(strict),different_gpu_same_commit=describe(cross),profile_transfer_different_commits=describe(transfer),first_published_run=aggregate('same_gpu'),second_published_run=aggregate('repeat_same_gpu'))
environments=[]
for host in ('ibolt-server-1','ibolt-server-2'):
 a=json.loads((ROOT/'environments'/host/'environment.json').read_text())
 b=json.loads((ROOT/'environments'/host/'environment-repeat.json').read_text())
 assert a==b,(host,'Environment changed between two runs')
 assert a['source_revision']==C['revision'] and a['source_archive_sha256']==sha(ROOT/'published-source.tgz')
 integrity=json.loads((ROOT/'environments'/host/'source-integrity.json').read_text())
 assert integrity['revision']==C['revision'] and integrity['files_verified']==18
 environments.append(dict(host=host,source_and_packages_and_gpu_driver_unchanged=True,packages=len(a['packages']),gpus=a['gpus'],source_archive_sha256=a['source_archive_sha256'],source_files_verified=integrity['files_verified']))
summary['environment_verification']=environments
report=dict(summary=summary,strict_reexecution=strict,different_gpu=cross,prior_experiment_comparison=transfer,notes=['Two published-code runs use the identical source archive, numerical settings, evaluation seeds, physical GPU placements and package/driver versions. No resume or warm start.','The pre-publication experiment uses a different source commit and is not the primary evidence for pure rerun determinism.','Checkpoint hashes include model weights, Adam state, training RNG and update count. Timing fields are not compared.','MMD and coverage are the original evaluation protocol. No changes were made to training code or seeds for this test.'])
(ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'same-code-rerun.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['seed','initial_checkpoint_identical','final_checkpoint_identical','samples_identical','mmd_first','mmd_second','mmd_difference','coverage_first','coverage_second'])
 writer.writeheader()
 for r in primary:
  m=r['full_policy'];writer.writerow(dict(seed=r['seed'],initial_checkpoint_identical=r['initial_checkpoint_identical'],final_checkpoint_identical=r['final_checkpoint_identical'],samples_identical=m['samples_bitwise_identical'],mmd_first=m['mmd_before'],mmd_second=m['mmd_after'],mmd_difference=m['mmd_difference'],coverage_first=m['coverage_before'],coverage_second=m['coverage_after']))
print(json.dumps(summary,indent=2))
