"""Read-only process/source/config audit for the current four-policy launch."""
import concurrent.futures,json,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
SOURCE='f953d28456d3800860dddb9b9cb91b6bd520ae00'
CAMPAIGN='antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2'
CODE=r"""import json,psutil,subprocess,time
from pathlib import Path
root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
source=Path('/home/heechan/OptiQ-ops/sources')/SOURCE
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
processes=[]
for p in psutil.process_iter(['pid','ppid','cmdline','status']):
 try:
  c=p.info['cmdline'] or [];s=' '.join(c)
  if 'antmaze' in s and ('geodesic' in s or ('antmaze_experiments.run' in c and str(root) in s)):
   processes.append(p.info)
 except(psutil.NoSuchProcess,psutil.AccessDenied):pass
result={'time':time.time(),'processes':processes,'geodesic_remaining':[p for p in processes if 'geodesic' in ' '.join(p['cmdline'])],
 'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
 'tracked_source_clean':not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()}
if root.exists():
 m=read(root/'manifest.json');result['status']=read(root/'status.json');result['failure']=read(root/'failure.json');result['jobs']={}
 for job in m['jobs']:
  run=root/'runs'/job['id'];pre=root/'preflight'/job['id']
  c=read(run/'config.json');proof=read(pre/'result.json')
  result['jobs'][job['id']]={'preflight_complete':bool(proof and proof['completed']),'preflight_proof':proof,
   'job':read(root/'jobs'/(job['id']+'.json')),'config':c,'progress':read(run/'progress.json'),'wandb':read(run/'wandb.json')}
print(json.dumps(result))
"""
def one(host):
 code='SOURCE='+repr(SOURCE)+'\nCAMPAIGN='+repr(CAMPAIGN)+'\n'+CODE
 p=subprocess.run(['ssh',host,'python3','-'],input=code,text=True,capture_output=True,check=True)
 d=json.loads(p.stdout);(OUT/('live-'+host+'.json')).write_text(json.dumps(d,indent=2)+'\n')
 rows=[]
 for name,j in d.get('jobs',{}).items():
  c=j['config'] or {};progress=j['progress'] or {};proof=j['preflight_proof'] or {}
  rows.append(dict(id=name,phase=(j['job'] or {}).get('phase'),preflight_complete=j['preflight_complete'],
    replay_verified=(proof.get('checkpoint') or {}).get('progress_replay_verified'),source=c.get('source_commit'),
    eval_starts=c.get('effective_eval_starts'),steps=progress.get('step',progress.get('steps')),updates=progress.get('updates'),wandb=j['wandb']))
 return dict(host=host,geodesic_remaining=len(d['geodesic_remaining']),source_clean=d['tracked_source_clean'],
   failure=d.get('failure'),learner_processes=[p['pid'] for p in d['processes'] if 'antmaze_experiments.run' in p['cmdline'] and p['ppid'] not in {x['pid'] for x in d['processes']}],jobs=rows)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 result=list(pool.map(one,['vast-heechan-180','vast-heechan-199','vast1']))
(OUT/'launch-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
