"""Small dispatch checks: no duplicates, preserve failures, disjoint CPU slots."""
import json,tempfile
from pathlib import Path
from queue import select_next,free_cpus
p=json.loads((Path(__file__).parent/'plan.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
 roots={v:Path(tmp)/v for v in ('v1','v2','v3')}
 active=[dict(version='v1',seed=0,cpus=[0,1])]
 assert select_next(p,'ant',active,set(),{},roots)[0]=='ant_v2_s0'
 assert free_cpus(p,'ant',active)==[10,11]
 assert select_next(p,'ant',active,{'ant_v1_s1'},{},roots)[0]=='ant_v2_s0'
 path=roots['v1']/'runs/ant_s1';path.mkdir(parents=True);(path/'FAILED.json').write_text('{}')
 assert select_next(p,'ant',active,set(),{},roots)[0]=='ant_v2_s0'
 assert free_cpus(p,'ant',[*active,dict(cpus=[10,11])]) is None
 assert {c for e in p['cpus'].values() for pair in e for c in pair}.isdisjoint({6,7,8,9})
 # Pending v2 seeds precede every unstarted v1 seed.
 for n in range(4):
  attempted={f'ant_v2_s{i}' for i in range(n)}
  assert select_next(p,'ant',active,attempted,{},roots)[0]==f'ant_v2_s{n}'
 for n in range(4):
  attempted={f'ant_v2_s{i}' for i in range(4)}|{f'ant_v3_s{i}' for i in range(n)}
  assert select_next(p,'ant',active,attempted,{},roots)[0]==f'ant_v3_s{n}'
 attempted={f'ant_{v}_s{i}' for v in ('v2','v3') for i in range(4)}
 assert select_next(p,'ant',active,attempted,{},roots)[0]=='ant_v1_s2'
print('PASS: v2 then v3 then v1 priority, duplicate/failed exclusion, disjoint CPU slots')
