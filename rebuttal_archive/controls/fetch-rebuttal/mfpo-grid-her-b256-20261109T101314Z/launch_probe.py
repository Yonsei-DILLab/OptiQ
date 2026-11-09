import pathlib,json,time,os,sys
b=pathlib.Path('/workspace/iBOLT/outputs/fetch-rebuttal/mfpo-grid-her-b256-20261109T101314Z/host2');d=json.loads((b/'plan.json').read_text());sys.path.insert(0,d['controller_source'])
from fetch_rebuttal.campaign import process_live
rows=[]
for j in d['jobs']:
 if j['phase']!='mfpo-hyperparameter-sweep':continue
 o=pathlib.Path(j['output']);cfg=json.loads((o/'config.json').read_text()) if (o/'config.json').exists() else {};native=json.loads((o/'native-config.json').read_text()) if (o/'native-config.json').exists() else {};progress=json.loads((o/'progress.json').read_text()) if (o/'progress.json').exists() else {}
 row={k:j.get(k) for k in ('id','seed','target_entropy_coeff','initial_temperature','gpu','pid','status')};row.update(live=process_live(j),steps=progress.get('steps',0),updates=progress.get('updates',0),config_ready=bool(cfg),native_ready=bool(native),failure=(o/'failure.json').exists())
 row['config_match']=bool(cfg) and cfg['batch_size']==256 and cfg['utd']==1 and cfg['HER'] and cfg['normalize_observations'] and cfg['mfpo_target_entropy_coeff']==j['target_entropy_coeff'] and cfg['mfpo_initial_temperature']==j['initial_temperature']
 row['native_match']=bool(native) and native['temp']==j['initial_temperature'] and native['target_entropy_coeff']==j['target_entropy_coeff']
 if process_live(j):
  env=dict(x.split(b'=',1) for x in pathlib.Path(f'/proc/{j["pid"]}/environ').read_bytes().split(b'\0') if b'=' in x)
  row['gpu_match']=env.get(b'CUDA_VISIBLE_DEVICES')==str(j['gpu']).encode();row['cpu_match']=sorted(os.sched_getaffinity(j['pid']))==sorted(j['cpus'])
 rows.append(row)
handoff=json.loads((b/'handoff-verification.json').read_text());preserved=all(process_live(j) for j in handoff['adopted'])
r=dict(time=time.time(),status=d['status'],counts=d.get('counts'),grid=rows,other_training_pids_preserved=preserved,gpu_preflights=len(d['mfpo_preflights']),selected_jobs_waiting=sum(j['phase']=='mfpo-selected-5seed' and j['status']=='waiting_for_selection' for j in d['jobs']))
print(json.dumps(r))
