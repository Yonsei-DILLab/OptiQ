"""Bounded teacher-only floor ablation; pinned training algorithm stays unchanged."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_start_normalized_geodesic import campaign_manifest as screen_manifest, CAMPAIGN as SCREEN_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT
from .register_increasing_temperature import CAMPAIGN as PRIORITY_CAMPAIGN

CAMPAIGN = 'antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = 'db4ca0a446f5fe8df1dda02e261fa9154c66d9c9'


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['temperature'] == 3.)
    jobs = []
    for floor in (.5, 1.):
        job = copy.deepcopy(control)
        job.update(id=f'v3-optiq-startnorm-geodesic-T3-teacherfloor{floor:g}-250k-s0',
                   hypothesis=f'teacher_floor_{floor:g}', teacher_std_floor=floor)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        failed_preflight_campaign='antmaze-optiq-v3-teacherfloor-250k-s0-20260925',
        failed_preflight_source='4607dd5c14e12fb87b395eb2d7192c742f77be22',
        preflight_correction='Accept omitted periodic diagnostics; verify runtime config and always-logged actor std. No main training started in the prior attempt.',
        protocol='antmaze_experiments/TEACHER_FLOOR_SCREEN_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        comparison_scope='v3 normalized reward/T3/gamma.999: change only existing teacher std floor; same seed and budget',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={job['hypothesis']: (.999, 3.) for job in jobs},
        assigned_hosts={0: HOST}, excluded_hosts=['vast1'],
        fresh_start=True, checkpoint_resume=False, seed=0,
        teacher_floor_grid=[.5, 1.], control_teacher_floor=0.006737946999085467,
        priority='Preserve live v1/v4 temperature-schedule runs; fill independent unlocked180 slots',
        evidence={
            'control_step': 258304, 'control_episodes': 100,
            'control_policy_entries': {'left': 16, 'right': 84},
            'control_policy_successful_routes': {'right': 4},
            'long_control': 'At450k/500k/550k both discount treatments chose only right; screened early at last logged598016.',
            'paired_native700': {'left_success': 0, 'right_success': 14, 'episodes': 100},
            'paired_supplementary1400': {'left_success': 1, 'right_success': 65, 'episodes': 100},
            'hypothesis': 'A teacher floor wider than the actor cap may keep useful candidate actions available after latent means concentrate.',
            'status': 'Hypothesis, not a demonstrated cause of route loss.',
            'risks': ['Wider candidates may emphasize critic extrapolation error.',
                      'Candidate action diversity need not produce successful trajectory diversity.']},
        screen='250k post-warmup only;40episodes each50k and100final. Compare successful routes and all failures, teacher ESS and Q range; no automatic extension or retry.')
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
