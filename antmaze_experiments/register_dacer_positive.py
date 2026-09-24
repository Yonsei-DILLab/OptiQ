"""Positive DACER target sweep with a 500-learner-update regulator cadence."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .register_euclidean_no_step import campaign_manifest as control_manifest
from .settings import NUM_ENVS, PREFLIGHT_STEPS, WARMUP, expected_updates

CAMPAIGN = 'antmaze-optiq-dacer-hpos-i500-500k-s0-20260925'
HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}
TARGETS = (.1, .5, .7, .9)
TASKS = ('v1', 'v2', 'v3', 'v4')
INTERVAL = 500
POST_WARMUP_BUDGET = 500000
TOTAL_STEPS = WARMUP + (POST_WARMUP_BUDGET // NUM_ENVS + 1) * NUM_ENVS
EVAL_INTERVAL = 100000
# Each host has every maze and target twice, and its first four jobs cover all
# mazes. No maze/condition barrier: a free slot immediately takes the next job.
SHARDS = {
    0: (('v1', .1), ('v2', .5), ('v3', .7), ('v4', .9),
        ('v1', .5), ('v2', .7), ('v3', .9), ('v4', .1)),
    1: (('v1', .7), ('v2', .9), ('v3', .1), ('v4', .5),
        ('v1', .9), ('v2', .1), ('v3', .5), ('v4', .7)),
}


def campaign_manifest(source, sha, shard):
    if shard not in SHARDS:
        raise ValueError(shard)
    manifest = control_manifest(source, sha, shard)
    prototypes = {entry['task']: entry for part in (0, 1)
                  for entry in control_manifest(source, sha, part)['jobs']}
    jobs = []
    for task, target in SHARDS[shard]:
        entry = copy.deepcopy(prototypes[task])
        entry.update(id=f'{task}-optiq-H{target:g}-i{INTERVAL}-500k-s0', seed=0,
                     dacer='on', dacer_target_entropy_per_dim=target,
                     dacer_interval_updates=INTERVAL, temperature=1.,
                     steps=TOTAL_STEPS, eval_interval=EVAL_INTERVAL)
        jobs.append(entry)
    for stale in ('supersedes_failed_preflight_campaign', 'changed_learning_setting',
                  'dacer_behavior_noise_std'):
        manifest.pop(stale, None)
    manifest.update(campaign=CAMPAIGN, host=HOSTS[shard], shard=shard, jobs=jobs,
        protocol='antmaze_experiments/DACER_POSITIVE_PROTOCOL.md',
        parent_source='23603a7e7696aa64e8e49a38b986d6b430b6d6f3',
        comparison_source='f953d28456d3800860dddb9b9cb91b6bd520ae00',
        comparison_campaign='antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2',
        comparison_scope='DACER-on target grid versus saved DACER-off policies at matched checkpoints; one seed',
        dacer_enabled=True, target_entropy_per_dim_grid=list(TARGETS),
        target_entropy_grid=[8*t for t in TARGETS], dacer_interval_updates=INTERVAL,
        dacer=dict(behavior_only=True, initial_alpha=.27, alpha_lr=.03, noise_scale=.1,
                   components=3, samples=200, entropy_seed=42, interval_updates=INTERVAL),
        entropy_measure='GMM joint-entropy proxy of clipped noisy conditional actions, not trajectory entropy',
        post_warmup_budget=POST_WARMUP_BUDGET,
        actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS, warmup_transitions=WARMUP,
        expected_learner_updates=expected_updates(TOTAL_STEPS),
        expected_regulator_updates=(expected_updates(TOTAL_STEPS)+INTERVAL-1)//INTERVAL,
        eval_interval=EVAL_INTERVAL,
        evaluation_clock='total transitions including warmup, rounded up to 256',
        preflight_total_transitions=PREFLIGHT_STEPS,
        preflight_required='Per-job real batch4096 updates, signed target/cadence, reward replay and checkpoint readback before main training',
        diagnostics=dict(regulator_history='every entropy update, including before/after noise and clipping',
                         policy_checkpoints='100k intervals, evaluation only; full checkpoint at final',
                         evaluation='40 episodes/mode intermediate; 100 final; no external DACER noise'),
        priority='listed order; independent per-slot 2-second backfill, no maze/target completion gates',
        assigned_hosts=HOSTS, excluded_hosts=['vast1'], existing_training_preserved=True,
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
