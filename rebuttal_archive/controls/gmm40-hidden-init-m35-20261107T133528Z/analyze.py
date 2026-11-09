"""Compare complete 100K-update runs under the original evaluation protocol."""
import json,csv,sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from gmm40.metrics import mmd2,assignments
c=json.loads((ROOT/'campaign.json').read_text())
baseline=ROOT.parent/'gmm40-no-floor-N256-20261107T123651Z'
rows=[]
for j in c['jobs']:
 p=ROOT/j['hidden_init']/f"seed-{j['seed']}"
 if not (p/'completed.json').exists(): continue
 cfg=json.loads((p/'config.json').read_text())
 old=json.loads((baseline/f"seed-{j['seed']}"/'config.json').read_text())
 for key in old.keys()-{'runtime','hidden_init'}:
  assert cfg[key]==old[key],(j['hidden_init'],j['seed'],key)
 assert cfg['hidden_init']==j['hidden_init'] and cfg['backend']=='gpu'
 assert cfg['runtime']['revision']==c['revision']
 assert cfg['runtime']['packages']==old['runtime']['packages']
 assert cfg['teacher_std_floor'] is None and cfg['log_std_min']==-5 and cfg['log_std_max']==-3.5
 done=json.loads((p/'completed.json').read_text()); assert done==dict(actor_updates=100000,completed=True)
 records=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
 r=records[-1]; assert r['actor_updates']==100000 and len(records)==16
 v=r['full_policy']; final=p/'evaluations/step_0100000'
 assert json.loads((final/'metrics_full_policy.json').read_text())==v
 x=np.load(final/'samples_full_policy.npy'); assert x.shape==(10000,2) and np.isfinite(x).all() and np.abs(x).max()<=40.00001
 reference=np.load(p/'reference.npy')
 np.testing.assert_array_equal(reference,np.load(baseline/'reference.npy'))
 squared,_=mmd2(x,reference)
 assert np.isclose(np.sqrt(max(squared,0)),v['mmd'],atol=1e-12,rtol=0)
 target=SimpleNamespace(means=np.asarray(cfg['target']['means']),std=np.asarray(cfg['target']['std']))
 _,_,counts=assignments(x,target)
 _,_,refcounts=assignments(reference,target)
 assert int((counts>=np.maximum(10,.1*refcounts)).sum())==v['mode_coverage']
 np.testing.assert_array_equal(counts,v['mode_counts_3sigma'])
 row=dict(hidden_init=j['hidden_init'],seed=j['seed'],mmd=v['mmd'],coverage=v['mode_coverage'],missing=sorted(set(range(40))-set(v['covered_modes'])),sliced_wasserstein2=v['sliced_wasserstein2'])
 rows.append(row)
summary={}
for init in sorted({r['hidden_init'] for r in rows}):
 sub=sorted([r for r in rows if r['hidden_init']==init],key=lambda r:r['seed'])
 a=np.array([r['mmd'] for r in sub]); full=[r['seed'] for r in sub]==list(range(5))
 summary[init]=dict(seeds=[r['seed'] for r in sub],mean_mmd=float(a.mean()),sample_sd_mmd=float(a.std(ddof=1)) if len(a)>1 else None,coverage=[r['coverage'] for r in sub],complete_five_seeds=full,meets_goal=bool(full and all(r['coverage']==40 for r in sub) and .021<=a.mean()<=.031))
report=dict(rows=rows,summary=summary,evaluation='Unchanged physical-coordinate five-RBF kernel, unbiased MMD squared averaged then sqrt(max(0,value)), first 2048 samples per distribution. Final checkpoint at 100000 updates, full policy only. SD across training seeds (ddof=1).',criterion=c['success_criterion'])
(ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'results.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['hidden_init','seed','mmd','coverage','missing','sliced_wasserstein2']); writer.writeheader();writer.writerows(rows)
print(json.dumps(report,indent=2))
