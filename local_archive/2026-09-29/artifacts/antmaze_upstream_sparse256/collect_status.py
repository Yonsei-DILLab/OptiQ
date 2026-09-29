"""Read-only snapshot of the registered official AntMaze campaign."""
import concurrent.futures, json, subprocess
from pathlib import Path
ROOT='/home/heechan/optiq-experiments/antmaze-upstream-sparse256-nativebudget-s0-20260923'
OUT=Path(__file__).resolve().parent
REMOTE='''from pathlib import Path
import json,subprocess,importlib.metadata
root=Path(%r)
result={}
for name in ['manifest','status','failure','result','environment-audit','registration','wandb-sync-status']:
 p=root/(name+'.json')
 if p.exists():result[name]=json.loads(p.read_text())
result['jobs']={p.stem:json.loads(p.read_text()) for p in (root/'jobs').glob('*.json')}
result['preflight']={}
for d in (root/'preflight').glob('*'):
 p=d/'result.json'
 if p.exists():result['preflight'][d.name]=json.loads(p.read_text())
result['runs']={} 
for p in (root/'runs').glob('*'):
 d={}
 for n in ['config','progress','failure','result','wandb','parameter-audit','checkpoint-verification']:
  f=p/(n+'.json')
  if f.exists():d[n]=json.loads(f.read_text())
 result['runs'][p.name]=d
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
children={};commands={}
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  status=(process/'stat').read_text().rsplit(')',1)[1].split()
  parent=int(status[1]);pid=int(process.name)
  children.setdefault(parent,[]).append(pid)
  command=(process/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if str(root) in command and 'antmaze_experiments.run ' in command:commands[pid]=command
 except (OSError,ValueError,IndexError):continue
result['learner_processes']=[dict(pid=pid,child_count=len(children.get(pid,[])),command=command)
 for pid,command in commands.items() if len(children.get(pid,[]))>=256]
result['canonical_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/heechan/OptiQ-direct-gmm-trg',text=True).strip()
print(json.dumps(result))
''' % ROOT

def collect(host):
 result=subprocess.run(['ssh',host,'python3 -'],input=REMOTE,text=True,capture_output=True,check=True)
 state=json.loads(result.stdout)
 (OUT/(host+'.json')).write_text(json.dumps(state,indent=2)+'\n')
 s=state.get('status',{})
 return dict(host=host,running=len(s.get('running',[])),pending=len(s.get('pending',[])),
  completed=len(s.get('completed',[])),failed=s.get('failed'),
  progress={k:v.get('progress',{}).get('step') for k,v in state['runs'].items()})

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor() as pool:
  for r in pool.map(collect,['vast-heechan-180','vast-heechan-199']):print(json.dumps(r))
