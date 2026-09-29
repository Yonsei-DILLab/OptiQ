"""Inspect actual fixed64 preflights/main processes; never start or alter learning."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SHA = '9b733bad9d52ba73fad01f03521f18a211d21a1e'
CAMPAIGN = 'antmaze-optiq-fixed64-v34-250k-s0-20260925'
CONTROL_ROOTS = {
    'v3': ROOT.parent/'antmaze_teacher_floor_250k_r2/results/vast-heechan-180',
    'v4': ROOT.parent/'antmaze_v4_teacher_floor_250k/results/vast-heechan-199',
}
read = lambda p: json.loads(p.read_text())
controls = {}
for task, root in CONTROL_ROOTS.items():
    floor = 1. if task == 'v3' else .5
    job = next(j for j in read(root/'manifest.json')['jobs']
               if j['task'] == task and j.get('teacher_std_floor') == floor)
    controls[task] = dict(job=job, config=read(root/'runs'/job['id']/'config.json'),
                         initial=read(root/'preflight'/job['id']/'parameter-audit.json')['initial'])

CODE = r'''
import json,sys,subprocess,os,time,hashlib
from pathlib import Path
import numpy as np
source=Path('/home/heechan/OptiQ-ops/sources')/SHA
sys.path.insert(0,str(source))
from antmaze_experiments.register_fixed_latent_screen import PINNED,PARENT
root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
read=lambda p:json.loads(p.read_text())
manifest=read(root/'manifest.json');status=read(root/'status.json')
assert manifest['source_commit']==status['source_commit']==SHA
assert not status['failed'] and not status['pending']
assert not (root/'failure.json').exists()
subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
file_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in PINNED}
assert file_hashes==manifest['algorithm_file_sha256']
if not all((root/'preflight'/j['id']/'result.json').exists()
           and (root/'runs'/j['id']/'progress.json').exists()
           and read(root/'runs'/j['id']/'progress.json').get('updates',0)>0
           for j in manifest['jobs']):
 print(json.dumps({'ready':False,'reason':'Waiting for real preflights/main learner progress'}))
 sys.exit(0)
compute={int(x.strip()) for x in subprocess.check_output(
 ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()}
def diff(a,b,prefix=''):
 out={}
 for k in sorted(set(a)|set(b)):
  left,right=a.get(k),b.get(k);name=prefix+k
  if isinstance(left,dict) and isinstance(right,dict):out.update(diff(left,right,name+'.'))
  elif left!=right:out[name]=[left,right]
 return out
items=[]
for j in manifest['jobs']:
 control=CONTROLS[j['task']];previous=control['config']
 pre=root/'preflight'/j['id'];run=root/'runs'/j['id'];proof=read(pre/'result.json')
 assert proof['completed'] and proof['steps']==8448 and proof['updates']==8
 assert proof['source_commit']==SHA and proof['checkpoint']['readback_verified']
 assert proof['checkpoint']['environment_reward_verified'] and proof['checkpoint']['progress_replay_verified']
 assert set(proof['summaries'])=={'native-fixed','policy-fixed','component0_mu-fixed'}
 assert read(pre/'parameter-audit.json')['initial']==control['initial']
 codebook_checks=[]
 for folder,stage in ((pre,'initial'),(pre,'final'),(run,'initial')):
  v=read(folder/f'latent-profile-{stage}-verification.json')
  assert v['verified'] and v['stage']==stage
  assert v['direct_sampler_verified'] and v['native_sampler_verified']
  assert v['full_stratified_components'] and v['serialization_roundtrip_verified']
  assert v['model_optimizer_rng_unchanged'] and v['latent_components']==64
  assert v['latent_codebook_seed']==20260911 and v['latent_prior']=='finite'
  if stage=='initial':
   assert v['actor_updates']==0 and v['parameters']==control['initial']
  else:assert v['actor_updates']==8
  codes=np.load(folder/'latent-codebook.npy',allow_pickle=False)
  assert codes.shape==(64,8) and np.isfinite(codes).all()
  assert hashlib.sha256(codes.tobytes()).hexdigest()==v['codebook_sha256']
  assert np.min(v['sample_counts'])>0 and sum(v['sample_counts'])==8192
  codebook_checks.append(dict(phase=folder.parent.name,stage=stage,sha256=v['codebook_sha256']))
 assert len({c['sha256'] for c in codebook_checks})==1
 cfg=read(run/'config.json');assert cfg['source_commit']==SHA
 changes=diff(previous['native'],cfg['native'])
 assert set(changes)=={'alg.actor.latent_prior','alg.actor.latent_components',
  'alg.actor.latent_codebook_seed','dual_mu_eval','output_root'},changes
 assert changes['alg.actor.latent_prior']==[None,'finite']
 assert changes['alg.actor.latent_components']==[None,64]
 assert changes['alg.actor.latent_codebook_seed']==[None,20260911]
 assert changes['dual_mu_eval']==[True,False]
 assert cfg['native']['output_root']==str(run)
 other={k:v for k,v in diff(previous,cfg).items() if not k.startswith('native.')}
 assert set(other)=={'source_commit','evaluation','latent_profile','latent_prior',
  'latent_components','latent_codebook_seed','latent_sampling','deterministic_control'},other
 assert cfg['steps']==258304 and cfg['expected_updates']==7816
 assert cfg['num_envs']==256 and cfg['updates_per_vector_step']==8 and cfg['batch_size']==4096
 assert cfg['reward_specification']==previous['reward_specification']
 assert cfg['latent_profile']=='fixed64' and cfg['native']['dacer']==previous['native']['dacer']
 probe=read(run/'teacher-proposal-verification.json')
 assert probe['verified'] and probe['finite_bounded_samples'] and probe['learner_rng_untouched']
 assert probe['exact_sampling_and_density_scales'] and probe['default_floor_identity']
 progress=read(run/'progress.json');assert progress['step']>8192 and progress['updates']>0
 pids=[]
 for pid in compute:
  proc=Path('/proc')/str(pid)
  try:args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except FileNotFoundError:continue
  if str(run)+' ' in args:
   assert 'antmaze_experiments.run ' in args and (proc/'cwd').resolve()==source
   pids.append(pid)
 finished=(run/'result.json').is_file() and read(run/'result.json').get('completed',False)
 if not finished:assert len(pids)==1,pids
 else:
  final=read(run/'result.json')
  assert final['source_commit']==SHA and final['updates']==7816
  assert final['checkpoint']['readback_verified'] and final['checkpoint']['progress_replay_verified']
 items.append(dict(id=j['id'],task=j['task'],learner_pids=pids,
  process_groups=[os.getpgid(p) for p in pids],progress=progress,
  preflight_updates=8,preflight_and_main_initial_parameters_match=True,
  native_differences=changes,metadata_differences=other,
  codebook_verifications=codebook_checks,main_completed=finished,
  teacher_preflight_update_verification=proof['teacher_update_verification'],
  wandb=read(run/'wandb.json')))
result=dict(time=time.time(),source_commit=SHA,verified=True,
 algorithm_files_unchanged_from=PARENT,algorithm_file_sha256=file_hashes,
 verification_script_sha256=VERIFIER_SHA256,jobs=items,
 controls={t:c['config']['source_commit'] for t,c in CONTROLS.items()},
 provenance='Post-launch read-only observations; metadata sidecar only. Frozen training source unchanged.')
(root/'actual-launch-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
code = ('SHA='+repr(SHA)+'\nCAMPAIGN='+repr(CAMPAIGN)+
        '\nCONTROLS='+repr(controls)+
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
        'jobs':[{'id':j['id'],'learner_pids':j['learner_pids'],
                 'step':j['progress']['step'],'wandb':j['wandb']['url']} for j in data['jobs']]}))
else:
    print(json.dumps(data))
