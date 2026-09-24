"""One fresh longer v3 run for the observed gamma=.999 route-entry candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_horizon_temperature import campaign_manifest as screen_manifest
from .register_horizon_temperature import CAMPAIGN as SCREEN_CAMPAIGN, CORE_FILES, PARENT
from .settings import NUM_ENVS, WARMUP, expected_updates

CAMPAIGN = 'antmaze-optiq-v3-gamma999-retention-1m-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = 'eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45'
POST_WARMUP_BUDGET = 1000000
TOTAL_STEPS = WARMUP + (POST_WARMUP_BUDGET // NUM_ENVS + 1) * NUM_ENVS


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha, 0)
    job = copy.deepcopy(next(j for j in manifest['jobs']
                             if j['task'] == 'v3' and j['hypothesis'] == 'gamma999'))
    job.update(id='v3-optiq-gamma999-H0.7-i500-1m-s0', steps=TOTAL_STEPS,
               hypothesis='gamma999_retention', seed=0)
    manifest.update(campaign=CAMPAIGN, jobs=[job], host=HOST, shard=0,
        protocol='antmaze_experiments/GAMMA_RETENTION_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='Fresh same-seed longer-budget follow-up, not independent replication or checkpoint resume',
        priority_campaign='/home/heechan/optiq-experiments/' + SCREEN_CAMPAIGN,
        post_warmup_budget=POST_WARMUP_BUDGET,
        actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS,
        expected_learner_updates=expected_updates(TOTAL_STEPS),
        expected_regulator_updates=(expected_updates(TOTAL_STEPS)+499)//500,
        discount_temperature_conditions={'gamma999_retention': (.999, 1.)},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        evidence={'screen_total_step': 258304, 'policy_episodes': 100,
                  'policy_entries': {'left': 54, 'right': 33, 'uncommitted': 13},
                  'mu_only_entries': {'left': 54, 'right': 20, 'uncommitted': 26},
                  'successes': 0, 'policy_closest_goal_distance_mean_m': 4.7156449856,
                  'conclusion': 'Promising route-entry retention only; goal-reaching remains unproved.'},
        screen='Inspect 500k/750k and final1M for successful routes and retention; no success claim from corridor entry')
    manifest['diagnostics']['policy_checkpoints'] = '50k intervals; evaluation only; final full state'
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
