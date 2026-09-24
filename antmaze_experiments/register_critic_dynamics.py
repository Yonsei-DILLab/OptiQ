"""Register the bounded OptiQ critic-dynamics diagnostic on the two 5090 hosts."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .dynamics_profiles import get_profile
from .progress_reward import specification
from .register_euclidean_no_step import campaign_manifest as control_manifest
from .settings import NUM_ENVS, PREFLIGHT_STEPS, WARMUP, expected_updates


CAMPAIGN = 'antmaze-optiq-critic-dynamics-500k-s0-20260925'
REFERENCE_SOURCE = 'f953d28456d3800860dddb9b9cb91b6bd520ae00'
REFERENCE_CAMPAIGN = 'antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2'
HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}
CONDITIONS = ('control', 'scale02', 'ema01', 'criticlr2', 'actordelay2', 'scale02ema01')
POST_WARMUP_BUDGET = 500000
TOTAL_STEPS = WARMUP + (POST_WARMUP_BUDGET // NUM_ENVS + 1) * NUM_ENVS
EVAL_INTERVAL = 100000

# Each condition appears once on each host, on opposite mazes. Each host gets
# three v3 and three v4 policies. The first eligible job on BOTH hosts is a
# fresh control; no maze or condition completion gates are introduced.
SHARDS = {
    0: (('v3', 'control'), ('v4', 'scale02'), ('v3', 'ema01'),
        ('v4', 'criticlr2'), ('v3', 'actordelay2'), ('v4', 'scale02ema01')),
    1: (('v4', 'control'), ('v3', 'scale02'), ('v4', 'ema01'),
        ('v3', 'criticlr2'), ('v4', 'actordelay2'), ('v3', 'scale02ema01')),
}


def campaign_manifest(source, sha, shard):
    if shard not in SHARDS:
        raise ValueError(f'Unknown shard: {shard}')
    manifest = control_manifest(source, sha, shard)
    prototypes = {entry['task']: entry for part in (0, 1)
                  for entry in control_manifest(source, sha, part)['jobs']}
    jobs = []
    for task, condition in SHARDS[shard]:
        profile = get_profile(condition)
        entry = copy.deepcopy(prototypes[task])
        entry.update(
            id=f'{task}-optiq-{condition}-500k-s0', seed=0,
            dynamics_profile=condition, dynamics_settings=copy.deepcopy(profile),
            reward_profile=profile['reward_profile'],
            reward_specification=specification(task, profile['reward_profile']),
            temperature=profile['temperature'], reward_multiplier=profile['reward_multiplier'],
            dacer='off', noveld='off', eval_starts='upstream',
            steps=TOTAL_STEPS, eval_interval=EVAL_INTERVAL,
            interim_eval_episodes=40, final_eval_episodes=100,
            save_intermediate_policy=True,
            expected_critic_updates=expected_updates(TOTAL_STEPS),
            expected_actor_updates=expected_updates(TOTAL_STEPS) // profile['policy_delay'])
        entry['optiq_profile'].update(critic_lr=profile['critic_lr'], tau=profile['tau'],
                                      policy_delay=profile['policy_delay'])
        jobs.append(entry)
    for stale in ('supersedes_failed_preflight_campaign', 'changed_learning_setting',
                  'optiq_temperature', 'optiq_critic_lr', 'optiq_tau'):
        manifest.pop(stale, None)
    manifest.update(
        campaign=CAMPAIGN, protocol='antmaze_experiments/CRITIC_DYNAMICS_PROTOCOL.md',
        host=HOSTS[shard], shard=shard, jobs=jobs,
        parent_source=REFERENCE_SOURCE, comparison_source=REFERENCE_SOURCE,
        comparison_campaign=REFERENCE_CAMPAIGN,
        campaign_kind='bounded hypothesis diagnostic, not a final performance benchmark',
        conditions={condition: get_profile(condition) for condition in CONDITIONS},
        reward_profile='per-job dynamics_settings.reward_profile',
        reward='100*Euclidean distance decrease, or 20*decrease in scale02 conditions; bonus=step_cost=0',
        post_warmup_budget=POST_WARMUP_BUDGET,
        actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS, warmup_transitions=WARMUP,
        expected_critic_updates=expected_updates(TOTAL_STEPS),
        eval_interval=EVAL_INTERVAL, evaluation_clock='total transitions including warmup, rounded up to 256',
        evaluation_starts='upstream fixed original full state for v3 and v4',
        preflight_total_transitions=PREFLIGHT_STEPS,
        preflight_critic_updates=expected_updates(PREFLIGHT_STEPS),
        preflight_required='Each job validates its own profile, real batch4096 updates, counters, reward replay and checkpoint readback before main training',
        diagnostics=dict(enabled=True, cadence='intermediate policy checkpoints and final',
                         preserve=['raw states', 'raw actions', 'critic and target Q',
                                   'teacher weights', 'Monte Carlo returns', 'route labels'],
                         full_checkpoint='final only'),
        priority='control first on each host; then listed order; independent per-slot 2-second backfill',
        assigned_hosts=HOSTS, excluded_hosts=['vast1'], existing_training_preserved=True,
        fresh_training=True, resume_cancelled_jobs=False,
        backfill_seconds=2, failure_holds_pending=True, automatic_restart=False,
        cross_job_completion_barriers=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=list(HOSTS), required=True)
    parser.add_argument('--host', choices=list(HOSTS.values()), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if HOSTS[args.shard] != args.host:
        parser.error('host and shard must match the fixed 5090-only assignment')
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
    configs = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGN, 'controller'), (CAMPAIGN + '-wandb-sync', 'sync_wandb')]
    assert not root.exists(), root
    for service, _ in services:
        assert not (configs / (service + '.conf')).exists(), service
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for service, module in services:
        (configs / (service + '.conf')).write_text(f'''[program:{service}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.{module} --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
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
    for service, _ in services:
        subprocess.run(ctl + ['update', service], check=True)
        subprocess.run(ctl + ['start', service], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), services=[s for s, _ in services], manifest=manifest), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                         jobs=[entry['id'] for entry in manifest['jobs']])))


if __name__ == '__main__':
    main()
