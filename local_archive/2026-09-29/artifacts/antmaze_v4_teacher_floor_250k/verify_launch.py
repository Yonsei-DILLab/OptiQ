"""Read-only proof of actual committed v4 jobs; never starts or changes learning."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT.parent/'antmaze_geodesic_gamma999_250k/results/vast-heechan-180'
SHA = '555bb7e3c0101ee939545cc35d4d0729291781ec'
CAMPAIGN = 'antmaze-optiq-v4-teacherfloor-250k-s0-20260925'
manifest = json.loads((CONTROL/'manifest.json').read_text())
control_job = next(j for j in manifest['jobs'] if j['task'] == 'v4')
folder = CONTROL/'preflight'/control_job['id']
control_cfg = json.loads((folder/'config.json').read_text())
control_initial = json.loads((folder/'parameter-audit.json').read_text())['initial']

CODE = r'''
import json,sys,subprocess,os,time,hashlib
from pathlib import Path
source=Path('/home/heechan/OptiQ-ops/sources')/SHA
sys.path.insert(0,str(source))
from antmaze_experiments.register_horizon_temperature import CORE_FILES,PARENT
root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
read=lambda p:json.loads(p.read_text())
manifest=read(root/'manifest.json');status=read(root/'status.json')
assert manifest['source_commit']==status['source_commit']==SHA
assert not status['failed'] and not status['pending']
if not all((root/'preflight'/j['id']/'result.json').exists()
           and (root/'runs'/j['id']/'progress.json').exists() for j in manifest['jobs']):
 print(json.dumps({'ready':False,'reason':'Waiting for real preflights/main progress'}))
 sys.exit(0)
subprocess.run(['git','diff','--exit-code',PARENT,'--',*CORE_FILES],cwd=source,check=True)
compute={int(x.strip()) for x in subprocess.check_output(
 ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()}
def diff(a,b,prefix=''):
 out={}
 for k in set(a)|set(b):
  left,right=a.get(k),b.get(k);name=prefix+k
  if isinstance(left,dict) and isinstance(right,dict):out.update(diff(left,right,name+'.'))
  elif left!=right:out[name]=[left,right]
 return out
items=[]
for j in manifest['jobs']:
 pre=root/'preflight'/j['id'];run=root/'runs'/j['id'];proof=read(pre/'result.json')
 assert proof['completed'] and proof['steps']==8448 and proof['updates']==8
 assert proof['source_commit']==SHA and proof['checkpoint']['readback_verified']
 assert proof['checkpoint']['environment_reward_verified'] and proof['checkpoint']['progress_replay_verified']
 initial=read(pre/'parameter-audit.json')['initial']
 assert initial==CONTROL_INITIAL
 cfg=read(run/'config.json');assert cfg['source_commit']==SHA
 changes=diff(CONTROL_CFG['native']['alg'],cfg['native']['alg'])
 assert set(changes)=={'actor.teacher_std_floor','actor.proposal_std','actor.proposal_std_pretanh'},changes
 assert all(pair[1]==j['teacher_std_floor'] for pair in changes.values())
 assert cfg['reward_specification']==CONTROL_CFG['reward_specification']
 assert cfg['native']['dacer']==CONTROL_CFG['native']['dacer']
 probe=read(run/'teacher-proposal-verification.json')
 assert probe['verified'] and probe['finite_bounded_samples'] and probe['learner_rng_untouched']
 assert probe['exact_sampling_and_density_scales'] and probe['default_floor_identity']
 progress=read(run/'progress.json');assert progress['step']>8192 and progress['updates']>0
 pids=[]
 for pid in compute:
  proc=Path('/proc')/str(pid)
  args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(run)+' ' in args:
   assert 'antmaze_experiments.run ' in args and (proc/'cwd').resolve()==source
   pids.append(pid)
 assert len(pids)==1,pids
 items.append(dict(id=j['id'],learner_pid=pids[0],process_group=os.getpgid(pids[0]),
   progress=progress,preflight_updates=8,preflight_initial_parameters_match=True,
   alg_differences=changes,teacher_probe=probe,
   teacher_preflight_update_verification=proof['teacher_update_verification'],
   wandb=read(run/'wandb.json')))
result=dict(time=time.time(),source_commit=SHA,verified=True,
 algorithm_files_unchanged_from=PARENT,control_source=CONTROL_CFG['source_commit'],
 verification_script_sha256=VERIFIER_SHA256,jobs=items)
(root/'actual-launch-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
code = ('SHA='+repr(SHA)+'\nCAMPAIGN='+repr(CAMPAIGN)+
        '\nCONTROL_CFG='+repr(control_cfg)+'\nCONTROL_INITIAL='+repr(control_initial)+
        '\nVERIFIER_SHA256='+repr(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())+'\n'+CODE)
result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
    'vast-heechan-199','/home/heechan/.venv-ddiffpg-native/bin/python','-'],
    input=code,text=True,capture_output=True,timeout=45)
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
data=json.loads(result.stdout)
if data.get('verified'):
    (ROOT/'actual-launch-verification.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'verified':True,'source_commit':SHA,
        'jobs':[{'id':j['id'],'learner_pid':j['learner_pid'],
                 'step':j['progress']['step'],'wandb':j['wandb']['url']} for j in data['jobs']]}))
else:
    print(json.dumps(data))
