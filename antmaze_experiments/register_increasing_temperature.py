"""Existing linear teacher-temperature schedule with increasing endpoints."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_geodesic_retention import campaign_manifest as screen_manifest
from .register_geodesic_retention import CAMPAIGN as SCREEN_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT
from .register_normalized_discount_retention import CAMPAIGN as PRIORITY_CAMPAIGN
from .settings import NUM_ENVS, WARMUP, expected_updates

CAMPAIGN = 'antmaze-optiq-geodesic-T1to3-retention-1m-s0-20260925'
HOST = 'vast-heechan-180'
SCREEN_SOURCE = '8e9d7d3c2c79f797654ccfb21913d2c000b89f71'
POST_WARMUP_BUDGET = 1000000
TOTAL_STEPS = WARMUP + (POST_WARMUP_BUDGET // NUM_ENVS + 1) * NUM_ENVS


def campaign_manifest(source, sha):
    manifest = screen_manifest(source, sha)
    previous = {job['task']: job for job in manifest['jobs']}
    jobs = []
    for task in ('v4', 'v1'):
        job = copy.deepcopy(previous[task])
        job.update(id=f'{task}-optiq-geodesic-gamma999-T1to3-H0.7-i500-1m-s0',
                   hypothesis='increasing_teacher_temperature',
                   temperature_schedule=dict(enabled=True,final_temperature=3.,
                                             anneal_steps=1000000,decay='linear'))
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/INCREASING_TEMPERATURE_PROTOCOL.md',
        parent_source=SCREEN_SOURCE, comparison_source=SCREEN_SOURCE,
        comparison_campaign=SCREEN_CAMPAIGN,
        algorithm_invariant_scope='Seven pinned computational files and the existing interpolation formula are unchanged',
        interpolation_unchanged_from='f5b3fcbfff3e61ecac4517facaa446393045c5a9',
        validation_only_changes=['positive schedule endpoints may increase', 'runner positive final-temperature validation', 'controller verifies combined schedule and DACER target'],
        comparison_scope='Original geodesic gamma.999 fixed-T1 control; only existing temperature schedule enabled',
        priority_campaign='/home/heechan/optiq-experiments/' + PRIORITY_CAMPAIGN,
        discount_temperature_conditions={'increasing_teacher_temperature': (.999, 'linear1to3')},
        temperature_schedule=dict(initial=1.,final=3.,anneal_steps=1000000,
                                  decay='linear',exclude_warmup=True,hold_after_anneal=True),
        priority='Preserve live v3 discount pair and v4 T10; use independent free180 slots',
        evidence={
            'v1_fixed_T1_same_origin': {'successful_routes_500224': {'upper':49,'lower':40},
                                      'successful_routes_1008384': {'lower':100},'episodes':100},
            'v4_fixed_T1_screen': {'last_logged_step':626688,'policy550144':{'lower':40},
                                  'successful_routes550144':{'lower':26},'episodes':40},
            'v4_fixed_T3': {'step':258304,'entries':{'upper':38,'lower':45,'uncommitted':17},
                            'successes':0,'episodes':100},
            'hypothesis': 'Start with stronger Q preference for acquisition, then soften it as values and route competence develop.',
            'limitation': 'Higher temperature can still harm locomotion or fail to preserve route skills; final settings alone do not prove diversity.'},
        screen='Inspect250k/500k/750k; successful distinct routes and later retention are required. v1 random-start primary results require supplementary same-full-origin evaluation. Stop sustained minority-route loss; no automatic extension.')
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
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for service, module in services:
        (configs/(service+'.conf')).write_text(f'''[program:{service}]
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
    subprocess.run(ctl+['reread'], check=True)
    for service, _ in services:
        subprocess.run(ctl+['update', service], check=True)
        subprocess.run(ctl+['start', service], check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(), host=args.host,
        services=[s for s, _ in services], source_commit=sha,
        jobs=[j['id'] for j in manifest['jobs']]), indent=2)+'\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                         jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
