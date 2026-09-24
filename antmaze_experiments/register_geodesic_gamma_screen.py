"""Fresh reward-only controls for the gamma=.999 route-retention screen."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_horizon_temperature import campaign_manifest as screen_manifest
from .register_horizon_temperature import CAMPAIGN as SCREEN_CAMPAIGN, CORE_FILES, PARENT
from .register_gamma_retention import CAMPAIGN as PRIORITY_CAMPAIGN
from .progress_reward import NO_COST_PROFILE, specification

CAMPAIGN = 'antmaze-optiq-geodesic-gamma999-250k-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = 'eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45'


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha, 0)
    controls = {j['task']: j for shard in (0, 1)
                for j in screen_manifest(source, sha, shard)['jobs']
                if j['hypothesis'] == 'gamma999'}
    jobs = []
    for task in ('v3', 'v4', 'v1'):
        job = copy.deepcopy(controls[task])
        job.update(id=f'{task}-optiq-geodesic-gamma999-H0.7-i500-250k-s0',
                   hypothesis='geodesic_gamma999', reward_profile=NO_COST_PROFILE,
                   reward_specification=specification(task, NO_COST_PROFILE))
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/GEODESIC_GAMMA_SCREEN_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='Fresh reward-only comparison with gamma999/T1 controls at identical checkpoints; all seed0',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={'geodesic_gamma999': (.999, 1.)},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        reward_profile=NO_COST_PROFILE,
        reward='100 * nearest-goal XY geodesic distance decrease; B=0; step penalty=0',
        priority='Preserve the live1M run and its GPU reservation; independently fill other unlocked180 slots',
        evidence={'v3_gamma999_T1_250k': {'policy_entries': {'left': 54, 'right': 33, 'uncommitted': 13},
                  'successes': 0, 'closest_goal_distance_mean_m': 4.7156449856},
                  'hypothesis': 'Wall-aware progress may improve completion of retained route entries.',
                  'limitation': 'Geodesic distance is a point-agent XY potential, not full-body configuration-space distance; v3 is not geometrically symmetric.'},
        screen='250k fixed budget; inspect50k checkpoints and100 final rollouts for successful distinct routes, not only entries')
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
