"""Read-only runtime/config verification; only a provenance sidecar is written."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
SHA='b111a993b1e1bf895966dac4469abdfdb2c37c05'
CAMPAIGN='antmaze-optiq-sigmacap-higher-v34-250k-s0-20260925'
CONTROLS={}
for task,name,host in [('v3','antmaze_teacher_floor_250k_r2','vast-heechan-180'),
                       ('v4','antmaze_v4_teacher_floor_250k','vast-heechan-199')]:
    root=ROOT.parent/name/'results'/host
    manifest=json.loads((root/'manifest.json').read_text())
    job=next(j for j in manifest['jobs'] if j['teacher_std_floor']==(1. if task=='v3' else .5))
    run=root/'runs'/job['id']
    CONTROLS[task]=dict(config=json.loads((run/'config.json').read_text()),
                        initial=json.loads((run/'parameter-audit.json').read_text())['initial'])

CODE=r'''
from pathlib import Path
import json,hashlib,sys,subprocess,time,os,math
source=Path('/home/heechan/OptiQ-ops/sources')/SHA
sys.path.insert(0,str(source))
from antmaze_experiments.register_actor_sigma_screen import PINNED,PARENT
from antmaze_experiments.actor_sigma_profile import settings
root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
read=lambda p:json.loads(p.read_text())
manifest=read(root/'manifest.json');status=read(root/'status.json')
assert manifest['source_commit']==status['source_commit']==SHA
assert not status['failed'] and not (root/'failure.json').exists(),status
subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
file_hashes={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in PINNED}
assert file_hashes==manifest['algorithm_file_sha256']
started=[j for j in manifest['jobs'] if j['id'] not in status['pending']]
assert len(started)>=4
if not all((root/'preflight'/j['id']/'result.json').exists()
           and (root/'runs'/j['id']/'progress.json').exists()
           and read(root/'runs'/j['id']/'progress.json')['updates']>0 for j in started):
 print(json.dumps(dict(ready=False,reason='Waiting for actual per-job preflight/main progress',
     pending=status['pending'],running=[j['id'] for j in status['running']])))
 sys.exit(0)
assert len(status['pending'])<=1
compute={int(x.strip()) for x in subprocess.check_output(
 ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()}
def diff(a,b,prefix=''):
 out={}
 for k in sorted(set(a)|set(b)):
  left,right=a.get(k),b.get(k);name=prefix+k
  if isinstance(left,dict) and isinstance(right,dict):out.update(diff(left,right,name+'.'))
  elif left!=right:out[name]=[left,right]
 return out
records=[]
for job in started:
 pre=root/'preflight'/job['id'];run=root/'runs'/job['id']
 control=CONTROLS[job['task']];wanted=settings(job['actor_sigma_profile'])
 proof=read(pre/'result.json')
 assert proof['completed'] and proof['source_commit']==SHA
 assert proof['steps']==8448 and proof['updates']==8
 assert all(proof['checkpoint'][k] for k in ['readback_verified','environment_reward_verified','progress_replay_verified'])
 assert proof['checkpoint']['simulator_count']==256
 assert set(proof['summaries'])=={'policy-fixed','native-fixed','zero_z-fixed'}
 for folder in [pre,run]:
  initial=read(folder/'actor-sigma-initial-verification.json')
  assert initial['verified'] and initial['profile']==job['actor_sigma_profile']
  assert initial['settings']==wanted and initial['actor_updates']==initial['critic_updates']==0
  assert initial['parameters']==control['initial']
  assert initial['upper_bound_removed']==(job['actor_sigma_profile']=='uncapped-initm1')
  assert len(initial['scratch_bound_checks'])==4
  assert all(c['finite_marginal_nll_and_gradients'] for c in initial['scratch_bound_checks'])
  assert initial['control_actor_after_restoring_only_initial_sigma_bias']==control['initial']['actor']
  assert all(initial[k] for k in ['direct_sampler_verified','native_sampler_verified','serialization_verified','model_optimizer_rng_unchanged'])
 final=read(pre/'actor-sigma-final-verification.json')
 assert final['verified'] and final['settings']==wanted
 assert final['actor_updates']==final['critic_updates']==8
 assert proof['teacher_update_verification']['actor_std_mean'] <= math.exp(wanted['log_std_max'])+1e-6
 cfg=read(run/'config.json');old=control['config']
 native_changes=diff(old['native'],cfg['native'])
 assert set(native_changes)=={'output_root','alg.actor.log_std_max'},native_changes
 metadata_changes={k:v for k,v in diff(old,cfg).items() if not k.startswith('native.')}
 assert set(metadata_changes)=={'source_commit','actor_sigma_profile','actor_sigma_upper_bound_removed','actor_sigma_upper_bound'},metadata_changes
 assert cfg['source_commit']==SHA and cfg['actor_sigma_profile']==job['actor_sigma_profile']
 assert cfg['reward_specification']==old['reward_specification']
 assert cfg['num_envs']==256 and cfg['batch_size']==4096 and cfg['updates_per_vector_step']==8
 assert cfg['steps']==258304 and cfg['expected_updates']==7816
 assert cfg['native']['dacer']==old['native']['dacer']
 assert cfg['native']['alg']['actor'].get('latent_prior','normal')=='normal'
 for k,v in wanted.items():assert cfg['native']['alg']['actor'][k]==v
 progress=read(run/'progress.json')
 assert progress['updates']>0 and progress['updates']==(progress['step']-8192)//32
 pids=[]
 for pid in compute:
  proc=Path('/proc')/str(pid)
  try:args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except FileNotFoundError:continue
  if str(run)+' ' in args:
   assert 'antmaze_experiments.run ' in args and (proc/'cwd').resolve()==source
   pids.append(pid)
 completed=(run/'result.json').exists() and read(run/'result.json').get('completed',False)
 if not completed:assert len(pids)==1,pids
 records.append(dict(id=job['id'],task=job['task'],profile=job['actor_sigma_profile'],
   settings=wanted,learner_pids=pids,process_groups=[os.getpgid(p) for p in pids],
   progress=progress,preflight_updates=8,preflight_and_main_control_initial_agreement=True,
   native_differences=native_changes,metadata_differences=metadata_changes,
   main_completed=completed,wandb=read(run/'wandb.json')))
record=dict(time=time.time(),host=HOST,source_commit=SHA,verified=True,jobs=records,pending=status['pending'],
  algorithm_unchanged_from=PARENT,algorithm_file_sha256=file_hashes,
  verification_script_sha256=SCRIPT_SHA,
  provenance='Post-launch read-only observations; metadata sidecar only; frozen source unchanged')
(root/'actual-launch-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',choices=['vast-heechan-180','vast-heechan-199'],required=True)
    host=parser.parse_args().host
    code='SHA='+repr(SHA)+'\nCAMPAIGN='+repr(CAMPAIGN)+'\nCONTROLS='+repr(CONTROLS)+\
        '\nHOST='+repr(host)+'\nSCRIPT_SHA='+repr(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())+'\n'+CODE
    result=subprocess.run(['ssh','-o','BatchMode=yes',host,
        '/home/heechan/.venv-ddiffpg-native/bin/python','-'],input=code,text=True,
        capture_output=True,timeout=60)
    if result.returncode:
        print(result.stderr);raise SystemExit(result.returncode)
    data=json.loads(result.stdout.splitlines()[-1])
    if data.get('verified'):
        (ROOT/('actual-launch-verification-'+host+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(host=host,verified=True,source_commit=SHA,pending=data['pending'],jobs=[dict(
            id=j['id'],pids=j['learner_pids'],step=j['progress']['step'],url=j['wandb']['url']) for j in data['jobs']])))
    else:print(json.dumps(dict(host=host,**data)))
