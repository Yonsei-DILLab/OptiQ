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
from .settings import reward_description, REWARD, DENSE_REWARD, NUM_ENVS, PREFLIGHT_STEPS, total_budget, expected_updates
from .dependencies import verify_dependencies


def priority_queue_ready(manifest):
    """Wait only for older queued jobs to be assigned, never for all to finish."""
    predecessor=manifest.get('priority_campaign')
    if predecessor is None:return True
    path=Path(predecessor)/'status.json'
    if not path.exists():return False
    state=json.loads(path.read_text())
    return 'pending' in state and not state['pending'] and not state.get('failed') and not state.get('pending_held')


def priority_reserved_gpus(manifest):
    """An assigned predecessor job owns its slot even before run-gpu takes flock."""
    predecessor=manifest.get('priority_campaign')
    if predecessor is None:return set()
    path=Path(predecessor)/'status.json'
    if not path.exists():return set(range(4))
    return {int(job['gpu']) for job in json.loads(path.read_text()).get('running',[])}


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
        if 'nm' in entry:
            cmd.extend(['--nm',str(entry['nm'])])
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
        for key in ('reward_profile','noveld','eval_starts','train_starts','interim_eval_episodes','dacer','dynamics_profile','eval_interval','dacer_interval_updates','discount','teacher_std_floor','latent_profile','collection_profile','actor_sigma_profile','optiq_config_profile'):
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
        from .collection_profile import get_profile as collection_settings, expected_updates as collection_updates
        collection=collection_settings(entry.get('collection_profile'))
        assert proof['steps']==expected and proof['updates']==collection_updates(expected,entry.get('collection_profile'))
        assert proof['checkpoint']['environment_reward_verified']
        config=json.loads((target/'config.json').read_text())
        reward=reward_description(entry.get('reward_profile','sparse'))
        assert config['reward']==reward and config['num_envs']==collection['num_envs'] and config['batch_size']==4096
        assert config['updates_per_vector_step']==collection['updates_per_vector_step']
        if 'collection_profile' in entry:
            verification=proof['collection_profile_verification']
            assert verification['verified'] and verification['profile']==entry['collection_profile']
            assert verification['settings']==collection and verification['actual_updates']==proof['updates']
            assert verification['simulator_count']==proof['checkpoint']['simulator_count']==collection['num_envs']
            initial=json.loads((target/'collection-profile-initial-verification.json').read_text())
            assert initial['verified'] and initial['actor_updates']==initial['critic_updates']==0
            assert initial['actual_envs']==collection['num_envs']
            assert initial['actual_updates_per_collection']==collection['updates_per_vector_step']
            if phase!='preflight':
                actual_steps=sorted(int(p.name) for p in (target/'evaluations').glob('*') if int(p.name)<expected)
                assert actual_steps==verification['expected_eval_steps']
        if 'reward_specification' in entry:
            assert config['reward_profile']==entry['reward_profile']
            assert config['reward_specification']==entry['reward_specification']
            if entry['reward_profile']=='dense':
                assert proof['checkpoint']['dense_replay_verified']
            else:
                assert proof['checkpoint']['progress_replay_verified']
        assert config['noveld_enabled']==(entry.get('noveld','on')=='on')
        assert config['eval_starts']==entry.get('eval_starts','upstream')
        assert config['train_starts']==entry.get('train_starts','upstream')
        if entry.get('train_starts')=='random':
            assert config['random_init'] and config['effective_eval_starts']=='random'
            actor=config['native']['alg']['actor']
            assert actor['num_policy_samples']==64 and actor['proposals_per_policy_sample']==1
            assert config['native']['experiment']['components']==64
            assert config['native']['experiment']['candidates']==64
            starts=json.loads((target/'train-starts-verification.json').read_text())
            assert starts['verified'] and starts['upstream_random_init']
            assert starts['observed_envs']==collection['num_envs']
            assert min(starts['initial_xy_peak_to_peak'])>1.
            natural=proof['summaries']['native-natural']
            assert natural['episodes']>=2 and not natural['identical_initial_full_state']
        if entry.get('train_starts')=='fixed':
            assert config['fixed_full_state_start'] and not config['random_init']
            starts=json.loads((target/'train-starts-verification.json').read_text())
            assert starts['verified'] and starts['profile']=='fixed-full-state'
            assert starts['observed_envs']==collection['num_envs']
            assert starts['identical_initial_full_state'] and starts['goal_count']==2
            for summary_path in target.glob('evaluations/*/*/summary.json'):
                summary=json.loads(summary_path.read_text())
                assert summary['fixed'] and summary['identical_initial_full_state']
                assert sum(summary['route_counts'].values())==summary['episodes']
                assert abs(sum(summary['route_proportions'].values())-1.)<1e-9
        assert proof['rnd_updates']==(proof['updates'] if config['noveld_enabled'] else 0)
        assert config['interim_eval_episodes']==entry.get('interim_eval_episodes',20)
        if 'optiq_profile' in entry:
            profile=json.loads((target/'optiq-profile-verification.json').read_text())
            assert profile['verified']
            assert profile['config_profile']==config['optiq_config_profile']==entry.get('optiq_config_profile','basic')
            assert profile['actor_hidden_dims']==entry['optiq_profile']['actor_hidden_dims']
            assert profile['critic_hidden_dims']==entry['optiq_profile']['critic_hidden_dims']
            assert profile['optimizers']['actor']['expected_lr']==entry['optiq_profile']['actor_lr']
            assert profile['optimizers']['critic']['expected_lr']==entry['optiq_profile']['critic_lr']
            expected_rnd_lrs = [1e-4] if config['noveld_enabled'] else []
            assert profile['rnd_lrs']==expected_rnd_lrs and profile['tau']==entry['optiq_profile'].get('tau',.005)
            assert config['wandb_project']==manifest['wandb_project']=='antmaze'
        if 'default_nm' in entry:
            assert config['nm']==entry['default_nm']
            actor=config['native']['alg']['actor']
            assert actor['num_policy_samples']==entry['default_nm']
            assert actor['proposals_per_policy_sample']==1
            assert config['native']['experiment']['components']==entry['default_nm']
            assert config['native']['experiment']['candidates']==entry['default_nm']
        if 'temperature' in entry:
            actor=config['native']['alg']['actor']
            assert actor['temperature']==entry['temperature']
            from .actor_sigma_profile import settings as sigma_settings
            sigma=sigma_settings(entry.get('actor_sigma_profile'))
            assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'])==tuple(sigma.values())
        if 'nm' in entry:
            n = entry['nm']
            actor = config['native']['alg']['actor']
            assert (actor['num_policy_samples'], actor['proposals_per_policy_sample']) == (n, 1)
            assert (config['nm'], config['native']['experiment']['components'],
                    config['native']['experiment']['candidates']) == (n, n, n)
            initial = json.loads((target/'nm-initial-verification.json').read_text())
            final = proof['nm_verification']
            assert initial['verified'] and final['verified']
            assert all(item['num_policy_samples'] == item['teacher_candidates'] == n and
                       item['proposals_per_policy_sample'] == 1 for item in (initial, final))
            assert all(item['actor_microbatch_size'] == (256 if n==256 else None)
                       for item in (initial, final))
            assert config['actor_microbatch_size'] == entry.get('actor_microbatch_size', 256 if n==256 else None)
            assert initial['actor_updates'] == initial['critic_updates'] == 0
            assert final['actor_updates'] == final['critic_updates'] == proof['updates']
        if 'actor_sigma_profile' in entry:
            verification=proof['actor_sigma_verification']
            initial=json.loads((target/'actor-sigma-initial-verification.json').read_text())
            assert initial['verified'] and verification['verified']
            assert initial['profile']==verification['profile']==config['actor_sigma_profile']==entry['actor_sigma_profile']
            assert initial['actor_updates']==initial['critic_updates']==0
            assert verification['actor_updates']==verification['critic_updates']==proof['updates']
            control=manifest['initial_parameter_control']
            assert initial['parameters']['critic']==control['critic']
            assert initial['control_actor_after_restoring_only_initial_sigma_bias']==control['actor']
            if initial['settings']['initial_log_std']==-1.:
                assert initial['parameters']['actor']==control['actor']
            if entry['actor_sigma_profile'].endswith('-initm1'):
                unbounded=entry['actor_sigma_profile']=='uncapped-initm1'
                assert initial['upper_bound_removed']==verification['upper_bound_removed']==unbounded
                assert config['actor_sigma_upper_bound_removed']==unbounded
                assert len(initial['scratch_bound_checks'])==len(verification['scratch_bound_checks'])==4
                assert all(c['finite_marginal_nll_and_gradients'] for p in (initial,verification)
                           for c in p['scratch_bound_checks'])
        if 'discount' in entry:
            assert config['discount']==config['native']['alg']['gamma']==proof['discount']==entry['discount']
        if 'teacher_std_floor' in entry:
            floor=entry['teacher_std_floor']
            verification=json.loads((target/'teacher-proposal-verification.json').read_text())
            assert verification['verified'] and verification['teacher_std_floor']==floor
            assert config['teacher_std_floor_override']==floor
            assert config['native']['alg']['actor']['proposal_std']==floor
            assert math.isclose(proof['teacher_std_floor'],floor,rel_tol=1e-6)
        if 'latent_profile' in entry:
            assert entry['latent_profile']==config['latent_profile']=='fixed64'
            verification=proof['latent_profile_verification']
            assert verification['verified'] and verification['latent_components']==64
            assert verification['latent_codebook_seed']==20260911
            initial=json.loads((target/'latent-profile-initial-verification.json').read_text())
            assert initial['verified'] and initial['codebook_sha256']==verification['codebook_sha256']
            assert 'component0_mu-fixed' in proof['summaries'] and 'zero_z-fixed' not in proof['summaries']
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
            assert config['temperature_schedule'] == entry.get('temperature_schedule')
            regulator=json.loads((target/'dacer_regulator.json').read_text())
            assert math.isclose(regulator['target_entropy'],target_entropy*8,abs_tol=1e-12)
            assert regulator['updates']>=1 and math.isfinite(regulator['entropy_proxy'])
            assert proof['dacer_target_entropy_per_dim']==target_entropy
            if 'dacer_interval_updates' in entry:
                interval=entry['dacer_interval_updates']
                count=(proof['updates']+interval-1)//interval
                assert config['native']['dacer']['interval_updates']==config['dacer_interval_updates']==interval
                assert verification['interval_updates']==proof['dacer_interval_updates']==interval
                assert regulator['updates']==count and proof['dacer_next_update']==count*interval
                history=[json.loads(line) for line in (target/'dacer_regulator_history.jsonl').read_text().splitlines()]
                assert [row['learner_updates'] for row in history]==list(range(0,proof['updates'],interval))
                assert len(history)==count and all(math.isfinite(row['entropy_proxy']) for row in history)
        if entry.get('dacer')=='off':
            disabled=json.loads((target/'dacer-disabled-verification.json').read_text())
            assert disabled['verified'] and disabled['train_equals_direct_policy_at_same_rng']
            assert config['dacer_enabled'] is False and config['native']['dacer']['enabled'] is False
            assert proof['dacer_enabled'] is False and proof['dacer_updates']==0 and proof['dacer_noise_std']==0.
            assert not (target/'dacer_regulator.json').exists()
        if entry.get('save_intermediate_policy',False):
            assert config['save_intermediate_policy']
            snapshots=sorted((target/'policy-checkpoints').glob('*/verification.json'))
            if phase=='preflight':
                expected_count=1
            elif 'collection_profile' in entry:
                expected_count=len(proof['collection_profile_verification']['expected_eval_steps'])
            else:
                expected_count=(expected-1)//entry.get('eval_interval',250000)
            assert len(snapshots)==expected_count,(len(snapshots),expected_count)
            for saved in snapshots:
                verification=json.loads(saved.read_text())
                assert verification['readback_verified'] and verification['restored_policy_state_exact']
                assert verification['source_commit']==manifest['source_commit']
        if 'dynamics_profile' in entry:
            from .dynamics_profiles import get_profile, expected_actor_updates
            settings=get_profile(entry['dynamics_profile'])
            verification=json.loads((target/'dynamics-verification.json').read_text())
            assert verification['verified'] and verification['settings']==settings
            assert config['dynamics_settings']==settings and config['eval_interval']==entry['eval_interval']
            assert verification['actor_updates']==expected_actor_updates(proof['updates'],entry['dynamics_profile'])
            assert verification['critic_updates']==proof['updates']
            assert verification['runtime_tau']==settings['tau']
            assert verification['runtime_policy_delay']==settings['policy_delay']
            assert verification['direct_policy_diagnostics']
            for other in (root/phase).glob('*/parameter-audit.json'):
                baseline=json.loads(other.read_text())['initial']
                for label in ('actor','critic'):
                    assert verification['initial_parameters'][label]['sha256']==baseline[label]['sha256'], (other,label)
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
    eligible_gpus=[int(gpu) for gpu in manifest.get('eligible_gpus',range(4))]
    assert eligible_gpus and all(0<=gpu<4 for gpu in eligible_gpus)
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
        priority_ready=priority_queue_ready(manifest)
        reserved=priority_reserved_gpus(manifest)
        if not failed and priority_ready:
            for gpu in eligible_gpus:
                if gpu in live or gpu in reserved or not pending:continue
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
        state.update(priority_campaign=manifest.get('priority_campaign'),priority_queue_ready=priority_ready,
                     priority_reserved_gpus=sorted(reserved))
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
