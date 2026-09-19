"""Small dispatch checks: no duplicates, preserve failures, disjoint CPU slots."""
import json,tempfile
from pathlib import Path
from queue import select_next,free_cpus
p=json.loads((Path(__file__).parent/'plan.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
 roots={v:Path(tmp)/v for v in ('v1','v2')}
 active=[dict(version='v1',seed=0,cpus=[0,1])]
 assert select_next(p,'ant',active,set(),{},roots)[0]=='ant_v1_s1'
 assert free_cpus(p,'ant',active)==[10,11]
 assert select_next(p,'ant',active,{'ant_v1_s1'},{},roots)[0]=='ant_v2_s0'
 path=roots['v1']/'runs/ant_s1';path.mkdir(parents=True);(path/'FAILED.json').write_text('{}')
 assert select_next(p,'ant',active,set(),{},roots)[0]=='ant_v2_s0'
 assert free_cpus(p,'ant',[*active,dict(cpus=[10,11])]) is None
 assert {c for e in p['cpus'].values() for pair in e for c in pair}.isdisjoint({6,7,8,9})
print('PASS: duplicate exclusion, pending selection, failed exclusion, disjoint CPU slots')
