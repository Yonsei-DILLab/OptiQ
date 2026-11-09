import json,sys,hashlib,subprocess
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());rows=[]
for j in c['jobs']:
 if j['host']!=sys.argv[2]:continue
 p=Path(j['output']);old=Path(c['baseline_root'])/f"seed-{j['seed']}"
 row={k:j[k] for k in ('host','gpu','seed','role')};row['completed']=(p/'completed.json').exists()
 if (p/'config.json').exists():
  cfg=json.loads((p/'config.json').read_text())
  for k,v in c['settings'].items():assert cfg[k]==v,(j,k,cfg[k])
  assert cfg['seed']==j['seed'] and cfg['teacher_std_floor']==j['teacher_std_floor'] and cfg['backend']=='gpu'
  assert cfg['runtime']['revision']==c['revision']
  if (old/'config.json').exists():
   before=json.loads((old/'config.json').read_text())
   for k,v in before.items():
    if k!='teacher_std_floor':assert cfg[k]==v,(j,k)
  row['config_valid']=True
 if (p/'metrics.jsonl').exists():
  lines=(p/'metrics.jsonl').read_text().splitlines()
  if lines:
   r=json.loads(lines[-1]);v=r['full_policy'];row.update(updates=r['actor_updates'],coverage=v['mode_coverage'],mmd=v['mmd'],sw2=v['sliced_wasserstein2'])
 a=p/'checkpoints/step_0000000.bin';b=old/'checkpoints/step_0000000.bin'
 if a.exists() and b.exists():row['same_initial_checkpoint']=hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
 row['supervisor']=subprocess.run(['sudo','supervisorctl','status',j['program']],capture_output=True,text=True).stdout.strip()
 if not row['completed'] and 'RUNNING' not in row['supervisor']:
  log=Path(j['output']+'.log')
  if log.exists():row['log_tail']=log.read_text()[-2000:]
 rows.append(row)
print('RESULT '+json.dumps(rows),flush=True)
