"""Compare fixed-origin direct-policy retention with the archived T1 control."""
from pathlib import Path
import hashlib,importlib.util,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
modules=[]
for folder in (ROOT.parent/'antmaze_geodesic_v1_origin_supplement',ROOT):
 p=folder/'report_results.py';s=importlib.util.spec_from_file_location(folder.name,p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 modules.append(m)
records=[[r for r in m.collect() if r['row']['mode']=='policy'] for m in modules]
steps=sorted({r['row']['step'] for r in records[0]} & {r['row']['step'] for r in records[1]})
assert steps==[500224,750080,1008384]
fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
first=None;rows=[]
for i,(m,entries,name) in enumerate(zip(modules,records,('Fixed teacher T=1','Linear teacher T: 1 to 3'))):
 for j,step in enumerate(steps):
  entry=next(r for r in entries if r['row']['step']==step)
  files=list((m.ROOT/'results/vast-heechan-180').glob(f'*/evaluations/{step:010d}/policy-fixed/rollouts.npz'))
  assert len(files)==1
  with np.load(files[0],allow_pickle=False) as d:
   start=d['initial_full_state']
   if first is None:first=start[0].copy()
   np.testing.assert_array_equal(start,np.repeat(first[None],100,axis=0))
  modules[1].base.draw(axes[i,j],'v1',entry,name)
  rows.append(dict(condition=name,**entry['row']))
fig.suptitle('v1 | Same full starting state | 100 direct-policy rollouts per checkpoint\nRandom latent + conditional sigma; no external DACER noise. One training seed; supplementary to native random-start evaluation.',fontsize=11)
out=ROOT/'report';out.mkdir(exist_ok=True)
fig.savefig(out/'fixed_vs_increasing_temperature.png',dpi=150);plt.close(fig)
proof=dict(rows=rows,identical_full_start_across_conditions=True,training_seed_count=1,
 direct_policy=True,primary_random_evaluation_preserved=True,
 result='T1 collapses to lower100/100; T1-to3 retains successful upper86/lower10 at1M. Retention beyond1M and other seeds remains untested.',
 source_hashes={str(m.ROOT):hashlib.sha256((m.ROOT/'report_results.py').read_bytes()).hexdigest() for m in modules},
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'retention-comparison.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'image':str(out/'fixed_vs_increasing_temperature.png')}))
