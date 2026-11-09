import json,sys,hashlib,subprocess
import numpy as np
import flax.serialization
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text()); rows=[]
for j in c['jobs']:
 if j['host']!=sys.argv[2]:continue
 p=Path(j['output']);old=Path(j['baseline_output'])
 row={k:j[k] for k in ('host','gpu','seed','role','program')}
 row['completed']=(p/'completed.json').exists()
 if (p/'config.json').exists():
  cfg=json.loads((p/'config.json').read_text())
  for key,value in c['settings'].items(): assert cfg[key]==value,(j['seed'],key,cfg[key])
  assert cfg['seed']==j['seed'] and cfg['backend']=='gpu'
  if (old/'config.json').exists():
   baseline=json.loads((old/'config.json').read_text())
   assert all(cfg[k]==v for k,v in baseline.items() if k!='runtime')
  row['config_valid']=True
 if (p/'metrics.jsonl').exists():
  lines=(p/'metrics.jsonl').read_text().splitlines()
  if lines:
   r=json.loads(lines[-1]);row.update(updates=r['actor_updates'],coverage=r['full_policy']['mode_coverage'],mmd=r['full_policy']['mmd'])
 matches=0;compared=0;first=None;details=[]
 for f in sorted((p/'checkpoints').glob('step_*.bin')):
  b=old/'checkpoints'/f.name
  if not b.exists():continue
  step=int(f.stem.split('_')[-1])
  if step>row.get('updates',-1):continue
  before=hashlib.sha256(b.read_bytes()).hexdigest();after=hashlib.sha256(f.read_bytes()).hexdigest()
  equal=before==after
  details.append(dict(updates=step,before_sha256=before,after_sha256=after,equal=equal))
  compared+=1;matches+=int(equal)
  if not equal and first is None:first=int(f.stem.split('_')[-1])
 row.update(checkpoints_compared=compared,checkpoints_identical=matches,first_checkpoint_difference=first)
 if p.exists(): (p/'checkpoint-comparison.json').write_text(json.dumps(details,indent=2)+'\n')
 if row['completed'] and (old/'checkpoints/step_0100000.bin').exists():
  a=flax.serialization.msgpack_restore((p/'checkpoints/step_0100000.bin').read_bytes())
  b=flax.serialization.msgpack_restore((old/'checkpoints/step_0100000.bin').read_bytes())
  differences=[]
  for layer,fields in a['state']['params'].items():
   for name,value in fields.items():
    differences.append(float(np.max(np.abs(value-b['state']['params'][layer][name]))))
  row.update(final_parameters_identical=all(v==0 for v in differences),final_parameter_max_abs_difference=max(differences),final_rng_identical=bool(np.array_equal(a['key'],b['key'])))
  (p/'parameter-comparison.json').write_text(json.dumps({k:row[k] for k in ('final_parameters_identical','final_parameter_max_abs_difference','final_rng_identical')},indent=2)+'\n')
 row['supervisor']=subprocess.run(['sudo','supervisorctl','status',j['program']],capture_output=True,text=True).stdout.strip()
 if 'RUNNING' not in row['supervisor'] and not row['completed']:
  log=Path(j['output']+'.log')
  if log.exists():row['log_tail']=log.read_text()[-1600:]
 rows.append(row)
print('RESULT '+json.dumps(rows),flush=True)
