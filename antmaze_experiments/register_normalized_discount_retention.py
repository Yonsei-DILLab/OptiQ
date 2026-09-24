"""Bounded discount comparison for the surviving v3 normalized-reward candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_start_normalized_geodesic import campaign_manifest as screen_manifest, CAMPAIGN as SCREEN_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT
from .register_geodesic_v4_temperature import CAMPAIGN as PRIORITY_CAMPAIGN
from .progress_reward import START_NORMALIZED_PROFILE
from .settings import NUM_ENVS, WARMUP, expected_updates

CAMPAIGN = 'antmaze-optiq-v3-startnorm-discount-retention-1m-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = 'db4ca0a446f5fe8df1dda02e261fa9154c66d9c9'
POST_WARMUP_BUDGET = 1000000
TOTAL_STEPS = WARMUP + (POST_WARMUP_BUDGET // NUM_ENVS + 1) * NUM_ENVS


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['temperature'] == 3.)
    jobs = []
    for label, discount in (('999', .999), ('99999', .99999)):
        job = copy.deepcopy(control)
        job.update(id=f'v3-optiq-startnorm-geodesic-gamma{label}-T3-H0.7-i500-1m-s0',
                   hypothesis=f'normalized_gamma{label}_retention',
                   discount=discount, steps=TOTAL_STEPS)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/NORMALIZED_DISCOUNT_RETENTION_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='v3 normalized-reward T3: same-seed longer-budget control and discount-only treatment; neither is a new independent seed',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={'normalized_gamma999_retention': (.999, 3.),
                                         'normalized_gamma99999_retention': (.99999, 3.)},
        post_warmup_budget=POST_WARMUP_BUDGET,
        actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS,
        expected_learner_updates=expected_updates(TOTAL_STEPS),
        expected_regulator_updates=(expected_updates(TOTAL_STEPS)+499)//500,
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        reward_profile=START_NORMALIZED_PROFILE,
        priority='Preserve live v4 T3/T10 and every GPU lock; use independently free180 slots',
        evidence={
            'control_total_step': 258304, 'control_episodes': 100,
            'control_policy_entries': {'right': 84, 'left': 16},
            'control_policy_successful_routes': {'right': 4},
            'control_native_entries': {'right': 92, 'left': 8},
            'control_native_successful_routes': {'right': 38},
            'historical_success_path_rescoring': {
                'left_count': 74, 'right_count': 4,
                'gamma999_mean_return': {'left': 1447.6995788863285, 'right': 1326.3385454099575},
                'gamma99999_mean_return': {'left': 1644.8610211770147, 'right': 1643.2345137095922},
                'undiscounted_common_return': 1647.0562748478314,
                'limitation': 'Different historical policies/budgets, selected successful episodes only; reward-functional diagnostic, not causal Q accuracy or a new policy.'},
            'hypothesis': 'Test whether longer acquisition plus weaker discounted path-timing preferences permits both successful routes to survive.',
            'risks': ['Near-unit discount may slow or destabilize value learning.',
                      'Equal undiscounted successful returns do not guarantee equal learned Q or mode retention.']},
        screen='Inspect250k/500k/750k and final1M. Stop a candidate if the minority route remains absent across two later checkpoints with no minority success; keep failures and stopped-run evidence. No automatic extension or retry.')
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=[HOST], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', PARENT, '--', *CORE_FILES], cwd=source, check=True)
    manifest = campaign_manifest(source, sha)
    manifest['algorithm_file_sha256'] = {p: hashlib.sha256((source/p).read_bytes()).hexdigest()
                                        for p in CORE_FILES}
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
        time=time.time(), host=args.host, services=[s for s, _ in services],
        source_commit=sha, jobs=[j['id'] for j in manifest['jobs']]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                         jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
