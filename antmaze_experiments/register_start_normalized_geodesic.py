"""Bounded v3 tests of a fixed start-normalized geodesic reward."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_geodesic_gamma_screen import campaign_manifest as screen_manifest, CAMPAIGN as SCREEN_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT
from .register_geodesic_retention import CAMPAIGN as PRIORITY_CAMPAIGN
from .progress_reward import START_NORMALIZED_PROFILE, specification

CAMPAIGN = 'antmaze-optiq-v3-startnorm-geodesic-250k-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = '438907f3a5ef681d6cde9012a66c7336fa642546'


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['task'] == 'v3')
    jobs = []
    for temperature in (1., 3.):
        job = copy.deepcopy(control)
        job.update(id=f'v3-optiq-startnorm-geodesic-gamma999-T{temperature:g}-H0.7-i500-250k-s0',
                   hypothesis=f'startnorm_geodesic_T{temperature:g}', temperature=temperature,
                   reward_profile=START_NORMALIZED_PROFILE,
                   reward_specification=specification('v3', START_NORMALIZED_PROFILE))
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/START_NORMALIZED_GEODESIC_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='v3 original geodesic/T1 at matched budgets: T1 changes only reward; T3 additionally changes teacher temperature',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={'startnorm_geodesic_T1': (.999, 1.),
                                         'startnorm_geodesic_T3': (.999, 3.)},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        reward_profile=START_NORMALIZED_PROFILE,
        reward='100 * decrease in minimum start-normalized remaining XY geodesic distance to an existing success region; B=0; step penalty=0',
        priority='Preserve live v1/v4 geodesic1M runs and GPU locks; fill independent unlocked180 slots',
        evidence={
            'reference_xy': [0., 0.],
            'v3_geodesic_center_distances_m': [17.986460085480267, 16.97056274847832],
            'probe_next_xy': [[-.1, .1], [.1, -.1]],
            'original_geodesic_progress_reward': [-14.1421356237, 14.1421356237],
            'new_normalized_progress_reward': [12.9180051986, 14.1421356237],
            'hypothesis': 'Normalize fixed origin-to-goal remaining distances to avoid initially penalizing progress toward the farther goal.',
            'changes': ['fixed per-goal distance weights', 'distance endpoint is the existing radius0.5 success region'],
            'limitation': 'Equal undiscounted potential change is not equal discounted return, full-body geodesic symmetry, or guaranteed exploration.'},
        screen='250k fixed budget with50k checkpoints; inspect successful distinct routes and failures, not only entries; no automatic extension')
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
