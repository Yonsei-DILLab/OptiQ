"""v4 teacher-only floor screen; no learning implementation changes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_geodesic_gamma_screen import campaign_manifest as screen_manifest, CAMPAIGN as SCREEN_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT

CAMPAIGN = 'antmaze-optiq-v4-teacherfloor-250k-s0-20260925'
HOST = 'vast-heechan-199'
SCREEN_SOURCE = '438907f3a5ef681d6cde9012a66c7336fa642546'


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['task'] == 'v4')
    jobs = []
    for floor in (.5, 1.):
        job = copy.deepcopy(control)
        job.update(id=f'v4-optiq-geodesic-T1-teacherfloor{floor:g}-250k-s0',
                   hypothesis=f'teacher_floor_{floor:g}', teacher_std_floor=floor)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/V4_TEACHER_FLOOR_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='v4 original geodesic/T1/gamma.999: only existing teacher std floor changes; same seed/budget',
        priority_campaign=None,
        discount_temperature_conditions={job['hypothesis']: (.999, 1.) for job in jobs},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        teacher_floor_grid=[.5, 1.], control_teacher_floor=0.006737946999085467,
        priority='Use independent unlocked199 slots; preserve all old results and cancelled queues',
        evidence={
            'T1_control': 'At250k direct goal success0/100; by600k lower-only success20/40, with upper route lost.',
            'T1to3': 'At700k/750k/800k direct upper1/lower39, at850k/900k lower40; zero success in each. Screened early.',
            'v3_transfer_limit': 'A preliminary wider-teacher screen improves some majority-route acquisition but has no minority-route success through250k. It does not establish a v4 effect.',
            'hypothesis': 'Broader teacher candidates may improve useful action acquisition while retaining alternatives at the original T1.',
            'risks': ['Wider candidates may emphasize critic extrapolation errors.',
                      'Improved acquisition or entries alone do not establish retained successful multimodality.']},
        screen='250k post-warmup only;40episodes each50k and100final. Compare both successful routes and all failures. No automatic extension or retry.')
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
