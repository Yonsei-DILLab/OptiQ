"""Register fixed-origin v3/v4 OptiQ runs with the fixed two-goal dense reward."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .collection_profile import get_profile
from .run import dense_reward_specification
from .settings import WANDB_ENTITY, WANDB_PROJECT, WARMUP


CAMPAIGN = 'antmaze-optiq-fixed-start-nearest-dense-v34-1m-s0-20260925-r2'
HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}
TASKS = {0: 'v3', 1: 'v4'}
PROFILE = 'env256-update256'
GPU = 3
TOTAL_STEPS = (1_000_000 // 256) * 256
EVAL_INTERVAL = 50_000


def campaign_manifest(source, sha, shard):
    if shard not in HOSTS:
        raise ValueError(shard)
    task = TASKS[shard]
    profile = get_profile(PROFILE)
    assert profile == dict(num_envs=256, updates_per_vector_step=256)
    goals = dense_reward_specification(task)['goals']
    return dict(
        campaign=CAMPAIGN, source=str(source), source_commit=sha,
        host=HOSTS[shard], shard=shard, eligible_gpus=[GPU],
        protocol='antmaze_experiments/FIXED_START_NEAREST_DENSE_PROTOCOL.md',
        condition='fixed_origin_fixed_two_goal_set_nearest_euclidean_dense',
        jobs=[dict(
            id=f'{task}-optiq-fixed-start-nearest-dense-T1-nm64-s0-r2',
            task=task, method='optiq', seed=0, steps=TOTAL_STEPS,
            collection_profile=PROFILE, optiq_config_profile='basic', default_nm=64,
            optiq_profile=dict(actor_hidden_dims=[256, 256],
                               critic_hidden_dims=[256, 256],
                               actor_lr=3e-4, critic_lr=3e-4, tau=.005),
            reward_profile='dense', reward_specification=dense_reward_specification(task),
            noveld='off', dacer='off', temperature=1.,
            train_starts='fixed', eval_starts='upstream',
            eval_interval=EVAL_INTERVAL, interim_eval_episodes=40,
            final_eval_episodes=100, save_intermediate_policy=True,
            start_xy=[0., 0.], fixed_goal_coordinates=goals,
            reward='-min_g ||p_{t+1} - g||_2 over the two fixed maze goals; no step penalty, bonus, or NovelD'),
        ],
        collection_profile=PROFILE, num_envs=profile['num_envs'],
        batch_size=4096, updates_per_vector_step=profile['updates_per_vector_step'],
        updates_per_transition=1., replay_capacity=1_000_000,
        warmup_transitions=WARMUP, total_transition_budget=TOTAL_STEPS,
        post_warmup_transitions=TOTAL_STEPS-WARMUP,
        temperature=1., dacer_enabled=False, noveld_enabled=False,
        optiq_actor_critic='256x2 GELU, Adam, actor/critic LR=3e-4, tau=.005',
        nm=64, log_std_bounds=[-5., -1.], initial_log_std=-1.,
        training_start='same original full state at XY origin for every reset and all vector workers',
        goal_set=dict(task=task, coordinates=goals, count=len(goals), random_selection=False),
        dense_reward=dict(formula='r_t=-min_g ||p_{t+1}-g||_2', distance='Euclidean XY to nearest of both fixed goals',
                          step_penalty=0., success_bonus=0., noveld=0.),
        evaluation=dict(starts='same original full state as training',
                        interval_transitions=EVAL_INTERVAL, episodes_per_mode=40,
                        final_episodes_per_mode=100,
                        modes=['native_random_z_mu_only', 'direct_policy_random_z_conditional_sigma'],
                        report_route_counts_and_proportions=True,
                        route_definitions=dict(v3='left x<-8 / right x>8 / both / uncommitted',
                                               v4='first x=-4 gate crossing: upper y>2 / lower y<-2 / uncommitted')),
        full_checkpoint='final only; intermediate policy-only checkpoints every evaluation',
        wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT,
        wandb_mode='online', seed=0, existing_training_preserved=True,
        eligible_gpu_policy='GPU3 only on each 5090 host; acquire OptiQ GPU lock; do not schedule on vast1',
        automatic_restart=False, failure_holds_pending=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=tuple(HOSTS), required=True)
    parser.add_argument('--host', choices=tuple(HOSTS.values()), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert HOSTS[args.shard] == args.host
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.shard)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    service = CAMPAIGN
    assert not root.exists(), root
    assert not (conf_root / f'{service}.conf').exists(), service
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (conf_root / f'{service}.conf').write_text(f'''[program:{service}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl = ['/usr/local/bin/supervisorctl', '-c',
           '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl + ['reread'], check=True)
    subprocess.run(ctl + ['update', service], check=True)
    subprocess.run(ctl + ['start', service], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), host=args.host, service=service, source_commit=sha,
        jobs=[job['id'] for job in manifest['jobs']], eligible_gpus=[GPU]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, service=service,
                          source_commit=sha, jobs=[job['id'] for job in manifest['jobs']],
                          eligible_gpus=[GPU])))


if __name__ == '__main__':
    main()
