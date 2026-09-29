import json,hashlib,sys
from pathlib import Path
import numpy as np
from flax import serialization
sys.path.insert(0,'/home/heechan/OptiQ-v5-gmm40')
from gmm40.evaluation import assignments,mmd2
from gmm40.target import Target
root=Path(sys.argv[1]);t=Target();report={}
for mode in ['fixed','fresh']:
 for seed in [0,1]:
  p=root/f'{mode}_seed{seed}';result=json.loads((p/'metrics.json').read_text());m=result['manifest'];row=result['rows'][-1]
  assert m['batch']==256 and m['n']==m['m']==64 and m['temperature']==1 and row['step']==100000
  ck=serialization.msgpack_restore((p/'checkpoint.msgpack').read_bytes());assert int(ck['step'])==int(ck['state']['step'])==100000
  bank=ck['bank'];assert bank.shape==(64,2)
  for step in [0,1000,5000,10000,20000,50000,100000]:
   data=np.load(p/f'step{step}.npz');assert np.isfinite(data['samples']).all();np.testing.assert_array_equal(data['z'],bank)
  data=np.load(p/'step100000.npz');x=data['samples'];ref=np.load(p/'reference.npy')
  _,near,counts=assignments(x,t);_,_,rc=assignments(ref,t)
  assert int((counts>=np.maximum(10,.1*rc)).sum())==row['coverage'];assert near.mean()==row['near_fraction'];np.testing.assert_array_equal(counts,row['counts'])
  expected,_=mmd2(x,ref);np.testing.assert_allclose(expected,row['mmd2'],rtol=1e-12,atol=1e-12)
  report[p.name]={'status':'passed','checkpoint_step':int(ck['state']['step']),'original_gmm40_metrics_match':True,'source_commit':m['source_commit']}
for seed in [0,1]:
 f=np.load(root/f'fixed_seed{seed}'/'step0.npz');r=np.load(root/f'fresh_seed{seed}'/'step0.npz')
 for key in ['mu','log_sigma','z']:np.testing.assert_array_equal(f[key],r[key])
report['paired_initialization']='identical';print(json.dumps(report,indent=2))
