"""Unchanged full-policy evaluation; mean and sample SD across training seeds."""
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from gmm40.metrics import mmd2
C=json.loads((ROOT/'campaign.json').read_text());BASE=Path(C['baseline_local'])
KEYS=['mmd','sliced_wasserstein2','mode_coverage','high_density_fraction','mode_mass_tv','histogram_js']

def read(p):
 cfg=json.loads((p/'config.json').read_text())
 assert json.loads((p/'completed.json').read_text())==dict(actor_updates=100000,completed=True)
 records=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
 assert len(records)==16 and records[-1]['actor_updates']==100000
 final=p/'evaluations/step_0100000'
 if not final.exists():final=p/'final'
 ref=np.load(p/'reference.npy');result={}
 for mode in ('full_policy','mu_only'):
  metrics=json.loads((final/f'metrics_{mode}.json').read_text());assert metrics==records[-1][mode]
  samples=np.load(final/f'samples_{mode}.npy')
  assert samples.shape==(10000,2) and np.isfinite(samples).all() and np.abs(samples).max()<=40.00001
  squared,_=mmd2(samples,ref)
  assert np.isclose(np.sqrt(max(squared,0)),metrics['mmd'],atol=1e-12,rtol=0)
  result[mode]=metrics
 return cfg,ref,result

baseline=[];floor_on=[];checks=[]
for seed in range(5):
 cfg,ref,result=read(BASE/f'seed-{seed}')
 assert cfg['teacher_std_floor'] is None and cfg['hidden_init']=='orthogonal_sqrt2'
 baseline.append(dict(seed=seed,metrics=result))
for j in C['jobs']:
 p=ROOT/j['role']/f"seed-{j['seed']}"
 if not (p/'completed.json').exists():continue
 cfg,ref,result=read(p)
 old,oldref,_=read(BASE/f"seed-{j['seed']}")
 for key,value in old.items():
  if key!='teacher_std_floor':assert cfg[key]==value,(j,key)
 assert cfg['teacher_std_floor']==j['teacher_std_floor']
 assert cfg['runtime']['revision']==C['revision']
 np.testing.assert_array_equal(ref,oldref)
 a=(p/'checkpoints/step_0000000.bin').read_bytes();b=(BASE/f"seed-{j['seed']}"/'checkpoints/step_0000000.bin').read_bytes()
 assert hashlib.sha256(a).digest()==hashlib.sha256(b).digest()
 row=dict(seed=j['seed'],same_initial_checkpoint=True,metrics=result)
 (floor_on if j['role']=='floor_on' else checks).append(row)
floor_on.sort(key=lambda r:r['seed']);checks.sort(key=lambda r:r['seed'])

def aggregate(rows):
 result={}
 for mode in ('full_policy','mu_only'):
  result[mode]={}
  for key in KEYS:
   values=np.array([r['metrics'][mode][key] for r in rows])
   result[mode][key]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)) if len(values)>1 else None)
 return result
summary=dict(source_revision=C['revision'],settings=C['settings'],baseline_floor_off=aggregate(baseline),floor_on_seeds=[r['seed'] for r in floor_on],floor_on_complete=len(floor_on)==5,additional_floor_off_checks_complete=len(checks)==3)
if floor_on:summary['floor_on']=aggregate(floor_on)
if len(floor_on)==5:
 difference=np.array([floor_on[s]['metrics']['full_policy']['mmd']-baseline[s]['metrics']['full_policy']['mmd'] for s in range(5)])
 summary['paired_mmd_difference_on_minus_off']=dict(mean=float(difference.mean()),sample_sd=float(difference.std(ddof=1)),per_seed=difference.tolist())
report=dict(summary=summary,baseline_cohort='Pre-specified original c2b5e59 orthogonal_sqrt2 five-seed floor-OFF campaign. Extra checks are not pooled into it.',baseline_runs=baseline,floor_on_runs=floor_on,additional_floor_off_checks=checks,evaluation='Original metrics: full-policy primary; first 2048 samples per distribution for sqrt(max(mean unbiased MMD2 over RBF bandwidths 1,2,5,10,20),0). 10000 evaluation samples. No checkpoint selection or evaluator changes.',aggregation='Mean and sample SD (ddof=1), five independent training seeds per primary condition. Overlapping SD bars alone are not an equivalence test.')
(ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'results.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['condition','seed','mode']+KEYS);writer.writeheader()
 for label,rows in [('floor_off_baseline',baseline),('floor_on_0.05',floor_on),('floor_off_additional_check',checks)]:
  for r in rows:
   for mode in ('full_policy','mu_only'):
    writer.writerow(dict(condition=label,seed=r['seed'],mode=mode,**{k:r['metrics'][mode][k] for k in KEYS}))
print(json.dumps(summary,indent=2))
