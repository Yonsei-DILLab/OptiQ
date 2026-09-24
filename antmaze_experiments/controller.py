"""Independent per-GPU preflight/train jobs; failure holds pending jobs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .run import write, CAMPAIGN
from .settings import REWARD, DENSE_REWARD, NUM_ENVS, PREFLIGHT_STEPS, total_budget, expected_updates
from .dependencies import verify_dependencies


def job(root, identifier, gpu, phases=('preflight','runs')):
    manifest=json.loads((root/'manifest.json').read_text())
    entry=next(j for j in manifest['jobs'] if j['id']==identifier)
    source=Path(manifest['source'])
    verify_dependencies(source, (entry['method'],))
    for phase in phases:
        target=root/phase/identifier
        cmd=['bash',str(source/'antmaze_experiments/launch.sh'),
             '--method',entry['method'],'--task',entry['task'],'--output',str(target)]
        if 'temperature' in entry:
            cmd.extend(['--temperature',str(entry['temperature'])])
        if 'dacer_target_entropy_per_dim' in entry:
            cmd.extend(['--dacer-target-entropy-per-dim',str(entry['dacer_target_entropy_per_dim'])])
        if 'temperature_schedule' in entry:
            schedule=entry['temperature_schedule']
            cmd.extend(['--temperature-final',str(schedule['final_temperature']),
                        '--temperature-anneal-steps',str(schedule['anneal_steps']),
                        '--temperature-decay',schedule['decay']])
        if 'steps' in entry and phase != 'preflight':
            cmd.extend(['--budget-steps',str(entry['steps'])])
        if 'final_eval_episodes' in entry:
            cmd.extend(['--final-eval-episodes',str(entry['final_eval_episodes'])])
        for key in ('reward_profile','noveld','eval_starts','interim_eval_episodes'):
            if key in entry:cmd.extend(['--'+key.replace('_','-'),str(entry[key])])
        if entry.get('save_intermediate_policy',False):
            cmd.append('--save-intermediate-policy')
        if phase=='preflight':cmd.append('--preflight')
        write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,pid=os.getpid(),phase=phase,status='running'))
        with (root/'logs'/f'{identifier}-{phase}.log').open('x') as log:
            result=subprocess.run(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,phase=phase,status='failed',returncode=result.returncode))
            return result.returncode
        proof=json.loads((target/'result.json').read_text())
        assert proof['completed'] and proof['source_commit']==manifest['source_commit']
        expected=PREFLIGHT_STEPS if phase=='preflight' else entry.get('steps',total_budget(entry['task']))
        assert proof['steps']==expected and proof['updates']==expected_updates(expected)
        assert proof['checkpoint']['environment_reward_verified']
        config=json.loads((target/'config.json').read_text())
        reward=REWARD if entry.get('reward_profile','sparse')=='sparse' else DENSE_REWARD
        assert config['reward']==reward and config['num_envs']==NUM_ENVS and config['batch_size']==4096
        assert config['noveld_enabled']==(entry.get('noveld','on')=='on')
        assert config['eval_starts']==entry.get('eval_starts','upstream')
        assert proof['rnd_updates']==(proof['updates'] if config['noveld_enabled'] else 0)
        assert config['interim_eval_episodes']==entry.get('interim_eval_episodes',20)
        if 'optiq_profile' in entry:
            profile=json.loads((target/'optiq-profile-verification.json').read_text())
            assert profile['verified']
            assert profile['actor_hidden_dims']==entry['optiq_profile']['actor_hidden_dims']
            assert profile['critic_hidden_dims']==entry['optiq_profile']['critic_hidden_dims']
            assert profile['optimizers']['actor']['expected_lr']==entry['optiq_profile']['actor_lr']
            assert profile['optimizers']['critic']['expected_lr']==entry['optiq_profile']['critic_lr']
            expected_rnd_lrs = [1e-4] if config['noveld_enabled'] else []
            assert profile['rnd_lrs']==expected_rnd_lrs and profile['tau']==.005
            assert config['wandb_project']==manifest['wandb_project']=='antmaze'
        if 'temperature' in entry:
            actor=config['native']['alg']['actor']
            assert actor['temperature']==entry['temperature']
            assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'])==(-5.,-1.,-1.)
        if 'temperature_schedule' in entry:
            schedule=entry['temperature_schedule']
            assert config['temperature_schedule']==schedule
            assert config['native']['alg']['actor']['temperature_schedule']==schedule
            progress=min(max((expected-config['warmup_transitions'])/schedule['anneal_steps'],0.),1.)
            wanted=(1-progress)*entry['temperature']+progress*schedule['final_temperature']
            if schedule['decay']=='log_linear':
                wanted=math.exp((1-progress)*math.log(entry['temperature'])+progress*math.log(schedule['final_temperature']))
            assert abs(proof['final_temperature']-wanted)<1e-9
        if 'dacer_target_entropy_per_dim' in entry:
            target_entropy=entry['dacer_target_entropy_per_dim']
            verification=json.loads((target/'dacer-target-verification.json').read_text())
            assert verification['verified'] and verification['action_dim']==8
            assert verification['target_entropy_per_dim']==target_entropy
            assert config['dacer_target_entropy_per_dim']==target_entropy
            assert config['native']['dacer']['target_entropy_per_dim']==target_entropy
            assert config['native']['dacer']['enabled'] and config['native']['dacer']['behavior_only']
            assert config['temperature_schedule'] is None
            regulator=json.loads((target/'dacer_regulator.json').read_text())
            assert math.isclose(regulator['target_entropy'],target_entropy*8,abs_tol=1e-12)
            assert regulator['updates']>=1 and math.isfinite(regulator['entropy_proxy'])
            assert proof['dacer_target_entropy_per_dim']==target_entropy
        if entry.get('save_intermediate_policy',False):
            assert config['save_intermediate_policy']
            snapshots=sorted((target/'policy-checkpoints').glob('*/verification.json'))
            expected_count=1 if phase=='preflight' else (expected-1)//250000
            assert len(snapshots)==expected_count,(len(snapshots),expected_count)
            for saved in snapshots:
                verification=json.loads(saved.read_text())
                assert verification['readback_verified'] and verification['restored_policy_state_exact']
                assert verification['source_commit']==manifest['source_commit']
    write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,status='completed'))
    return 0


def controller(root):
    lock=(root/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text())
    source=Path(manifest['source'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==manifest['source_commit']
    verify_dependencies(source, {entry['method'] for entry in manifest['jobs']})
    pending=list(manifest['jobs']);live={};completed=[];failed=[]
    for directory in ('jobs','logs','preflight','runs'): (root/directory).mkdir(exist_ok=True)
    if (root/'status.json').exists():raise RuntimeError('Existing controller state: do not restart automatically')
    while pending or live:
        for gpu,(proc,entry) in list(live.items()):
            code=proc.poll()
            if code is None:continue
            del live[gpu]
            if code==0:completed.append(entry['id'])
            else:
                failed.append(dict(id=entry['id'],returncode=code,gpu=gpu))
                write(root/'failure.json',dict(failed=failed,pending_held=True,time=time.time()))
        if not failed:
            for gpu in range(4):
                if gpu in live or not pending:continue
                lock_path=Path(f'/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock')
                with lock_path.open('a') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                entry=pending.pop(0)
                env=dict(os.environ,OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu))
                cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',
                     '/home/heechan/.venv-ddiffpg-native/bin/python','-m','antmaze_experiments.controller',
                     '--root',str(root),'--job',entry['id'],'--gpu',str(gpu)]
                with (root/'logs'/f"{entry['id']}-job.log").open('x') as log:
                    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                live[gpu]=(proc,entry)
        state=dict(source_commit=manifest['source_commit'],controller_pid=os.getpid(),time=time.time(),
            pending=[j['id'] for j in pending],running=[dict(**entry,gpu=gpu,pid=proc.pid) for gpu,(proc,entry) in live.items()],
            completed=completed,failed=failed,pending_held=bool(failed))
        write(root/'status.json',state)
        if failed and not live: return 1
        time.sleep(2)
    results={key:json.loads((root/'runs'/key/'result.json').read_text()) for key in completed}
    write(root/'result.json',dict(completed=True,source_commit=manifest['source_commit'],results=results))
    return 0


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--job');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    sys.exit(job(a.root,a.job,a.gpu) if a.job else controller(a.root))


if __name__=='__main__':main()
