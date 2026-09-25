"""Register the OptiQ 256-update/256-transition AntMaze comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .collection_profile import get_profile
from .progress_reward import EUCLIDEAN_NO_COST_PROFILE, specification
from .settings import BUDGETS, WANDB_ENTITY, WANDB_PROJECT, WARMUP, total_budget


HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}
TASKS = {0: ('v1', 'v3'), 1: ('v2', 'v4')}
CAMPAIGNS = {
    'control': 'antmaze-optiq-utd1-euclidean-control-v1234-s0-20260925',
    'basic': 'antmaze-optiq-utd1-basic-v1234-s0-20260925',
}
PROFILE = 'env256-update256'
EVAL_INTERVAL = 50000


def campaign_manifest(source, sha, shard, condition):
    if shard not in HOSTS or condition not in CAMPAIGNS:
        raise ValueError((shard, condition))
    control = condition == 'control'
    reward = EUCLIDEAN_NO_COST_PROFILE if control else 'sparse'
    optiq_profile = dict(actor_hidden_dims=[256] * (3 if control else 2),
                         critic_hidden_dims=[256] * (3 if control else 2),
                         actor_lr=3e-4, critic_lr=5e-4 if control else 3e-4,
                         tau=.005)
    jobs = []
    for task in TASKS[shard]:
        entry = dict(id=f'{task}-optiq-utd1-{condition}-s0', task=task,
                     method='optiq', seed=0, steps=total_budget(task),
                     collection_profile=PROFILE,
                     optiq_config_profile='legacy' if control else 'basic',
                     optiq_profile=optiq_profile.copy(),
                     reward_profile=reward, noveld='off', dacer='off',
                     temperature=1., eval_starts='upstream',
                     eval_interval=EVAL_INTERVAL, interim_eval_episodes=40,
                     final_eval_episodes=100, save_intermediate_policy=True)
        if control:
            entry['reward_specification'] = specification(task, reward)
        jobs.append(entry)
    collection = get_profile(PROFILE)
    assert collection == dict(num_envs=256, updates_per_vector_step=256)
    return dict(campaign=CAMPAIGNS[condition], source=str(source), source_commit=sha,
                shard=shard, host=HOSTS[shard], jobs=jobs,
                protocol='antmaze_experiments/UTD256_PROTOCOL.md',
                condition=condition, comparison='same condition at 8 updates per 256 transitions',
                collection_profile=PROFILE, num_envs=256, batch_size=4096,
                updates_per_vector_step=256, updates_per_transition=1.,
                warmup_transitions=WARMUP, replay_capacity=1000000,
                native_post_warmup_budgets=BUDGETS,
                temperature=1., dacer_enabled=False, noveld_enabled=False,
                optiq_config_profile='legacy' if control else 'basic',
                reward_profile=reward,
                evaluation_starts='upstream: v1 random; v2-v4 original fixed full state',
                interim_eval_interval_transitions=EVAL_INTERVAL,
                interim_eval_episodes_per_mode=40, final_eval_episodes_per_mode=100,
                eval_modes=['native_mu_only', 'direct_policy_with_sigma'],
                reporting='report each 50k environment-transition boundary per task; exact evaluation steps round up to 256',
                wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT,
                wandb_mode='online', seed=0, excluded_hosts=['vast1'],
                existing_training_preserved=True, fresh_training=True,
                resume_cancelled_jobs=False, backfill_seconds=2,
                failure_holds_pending=True, automatic_restart=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--condition', choices=tuple(CAMPAIGNS), required=True)
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
    manifest = campaign_manifest(source, sha, args.shard, args.condition)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGNS[args.condition]
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGNS[args.condition], 'controller'),
                (CAMPAIGNS[args.condition] + '-wandb-sync', 'sync_wandb')]
    assert not root.exists(), root
    for name, _ in services:
        assert not (conf_root / (name + '.conf')).exists(), name
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, module in services:
        (conf_root / (name + '.conf')).write_text(f'''[program:{name}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.{module} --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGNS[args.condition]}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/{module}.log
stderr_logfile={root}/{module}.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl = ['/usr/local/bin/supervisorctl', '-c',
           '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl + ['reread'], check=True)
    for name, _ in services:
        subprocess.run(ctl + ['update', name], check=True)
        subprocess.run(ctl + ['start', name], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), host=args.host, services=[n for n, _ in services],
        source_commit=sha, jobs=[j['id'] for j in manifest['jobs']]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
