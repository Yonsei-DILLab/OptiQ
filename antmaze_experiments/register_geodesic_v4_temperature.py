"""Temperature-only v4 screen after the geodesic T1 policy loses a route."""
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

CAMPAIGN = 'antmaze-optiq-v4-geodesic-temperature-250k-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = '438907f3a5ef681d6cde9012a66c7336fa642546'


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['task'] == 'v4')
    jobs = []
    for temperature in (3., 10.):
        job = copy.deepcopy(control)
        job.update(id=f'v4-optiq-geodesic-gamma999-T{temperature:g}-H0.7-i500-250k-s0',
                   hypothesis=f'geodesic_T{temperature:g}', temperature=temperature)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/GEODESIC_V4_TEMPERATURE_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='Fresh v4 original geodesic250k/T1 paired control; only teacher temperature and bookkeeping change',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={'geodesic_T3': (.999, 3.), 'geodesic_T10': (.999, 10.)},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        priority='Preserve current geodesic1M and v3 normalized-reward jobs; fill independently unlocked180 slots',
        evidence={
            'v4_origin_geodesic_distances_m': [17.65685624949359, 17.65685624949359],
            'T1_policy_350208': {'lower': 40, 'upper': 0, 'successes': 0},
            'T1_policy_400128': {'lower': 40, 'upper': 0, 'successes': 0},
            'T1_policy_450048': {'lower': 39, 'uncommitted': 1, 'upper': 0, 'successes': 0},
            'hypothesis': 'A higher teacher temperature may retain probability on actions with slightly lower learned Q, improving minority-route survival.',
            'limitation': 'Higher temperature may instead impede goal acquisition; this is not established from entropy alone.'},
        screen='250k fixed budget with50k checkpoints/100 final episodes; compare successful routes and failures; no automatic extension')
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
