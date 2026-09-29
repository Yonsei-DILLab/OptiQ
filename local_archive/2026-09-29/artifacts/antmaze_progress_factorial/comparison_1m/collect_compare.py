import json,sys,subprocess,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/Users/yunheechan/Documents/ChatGPT/OptiQ');sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('details',R/'artifacts/antmaze_dacer_off_t1/report_completed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
files=subprocess.check_output(['rg','--files','artifacts'],cwd=R,text=True).splitlines();seen=set();out=[]
for file in sorted(files,reverse=True):
 if '/0001000192/policy-natural/rollouts.npz' not in file or 'optiq' not in file:continue
 if not any('/'+p+'/' in file for p in ('antmaze_progress_factorial','antmaze_dacer_off_t1','antmaze_dacer_off_temperature','antmaze_dense_off_16_current')):continue
 p=R/file;cfg=json.loads((p.parents[3]/'config.json').read_text());task=cfg['task']
 if task not in ('v3','v4'):continue
 key=(task,cfg['source_commit'],cfg['temperature'],cfg['reward_profile'])
 if key in seen:continue
 seen.add(key);d=np.load(p);s=json.loads(p.with_name('summary.json').read_text());paths=[xy[:int(n)+1] for xy,n in zip(d['xy'],d['lengths'])]
 fam=Counter(m.family(task,x) for x in paths);goals=np.array(m.audit.GOALS[task]);
 row=dict(task=task,T=cfg['temperature'],dacer=cfg.get('dacer_enabled',cfg['native']['dacer']['enabled']),reward=cfg['reward_profile'],step=s['step'],episodes=len(paths),success=int(sum(d['goals']>0)),corridors=dict(fam),mean_max_displacement=float(np.mean([np.linalg.norm(x-x[0],axis=-1).max() for x in paths])),mean_closest_goal=float(np.mean([np.linalg.norm(x[:,None,:]-goals,axis=-1).min() for x in paths])),input=str(p),config=cfg,starts_sha=str(__import__('hashlib').sha256(d['initial_full_state'].tobytes()).hexdigest()))
 out.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('config','input','starts_sha')}))
Path('/tmp/antmaze-1m-comparison.json').write_text(json.dumps(out,indent=2))
