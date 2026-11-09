import json,sys,subprocess
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
rows=[]
for j in c['jobs']:
 if j['host']!=sys.argv[2]: continue
 p=Path(j['output']); row={k:j[k] for k in ('host','gpu','hidden_init','seed','program')}
 row['completed']=(p/'completed.json').exists()
 if (p/'config.json').exists():
  cfg=json.loads((p/'config.json').read_text())
  for key in ('log_std_min','log_std_max','initial_log_std','teacher_std_floor','n','m','batch','steps','eval_samples','mean_output_init_scale'):
   assert cfg[key]==c['settings'][key],(key,cfg[key])
  assert cfg['hidden_init']==j['hidden_init'] and cfg['seed']==j['seed']
  assert cfg['backend']=='gpu' and cfg['runtime']['revision']==c['revision']
  row['config_valid']=True
 if (p/'metrics.jsonl').exists():
  lines=(p/'metrics.jsonl').read_text().splitlines()
  if lines:
   r=json.loads(lines[-1]); v=r['full_policy']; row.update(updates=r['actor_updates'],coverage=v['mode_coverage'],mmd=v['mmd'])
 row['supervisor']=subprocess.run(['sudo','supervisorctl','status',j['program']],capture_output=True,text=True).stdout.strip()
 rows.append(row)
print('RESULT '+json.dumps(rows),flush=True)
print(subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)
