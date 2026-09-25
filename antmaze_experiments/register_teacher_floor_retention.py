"""Longer-budget confirmation of the completed v3 teacher-floor candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_teacher_floor_screen import campaign_manifest as short_manifest, CAMPAIGN as SHORT_CAMPAIGN
from .register_horizon_temperature import CORE_FILES, PARENT
from .settings import NUM_ENVS, WARMUP, expected_updates

CAMPAIGN = 'antmaze-optiq-v3-teacherfloor1-retention-1m-s0-20260925'
HOST = 'vast-heechan-180'
SHORT_SOURCE = 'd25930197ee9ba060a3aa84c3b6fea419dfe856d'
TOTAL_STEPS = WARMUP + (1000000 // NUM_ENVS + 1) * NUM_ENVS


def campaign_manifest(source, sha):
    manifest = short_manifest(source, sha)
    control = next(job for job in manifest['jobs'] if job['teacher_std_floor'] == 1.)
    job = copy.deepcopy(control)
    job.update(id='v3-optiq-startnorm-geodesic-T3-teacherfloor1-1m-s0',
               hypothesis='teacher_floor1_retention', steps=TOTAL_STEPS)
    for key in ('failed_preflight_campaign', 'failed_preflight_source', 'preflight_correction'):
        manifest.pop(key, None)
    manifest.update(campaign=CAMPAIGN, jobs=[job], host=HOST, shard=0,
        protocol='antmaze_experiments/TEACHER_FLOOR_RETENTION_PROTOCOL.md',
        parent_source=SHORT_SOURCE, comparison_source=SHORT_SOURCE,
        comparison_campaign=SHORT_CAMPAIGN,
        comparison_scope='Same-seed fresh longer-budget confirmation; only budget and job identity differ from the completed teacher-floor1 screen.',
        priority_campaign=None, teacher_floor_grid=[1.],
        discount_temperature_conditions={'teacher_floor1_retention': (.999, 3.)},
        post_warmup_budget=1000000, actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS,
        expected_learner_updates=expected_updates(TOTAL_STEPS),
        expected_regulator_updates=(expected_updates(TOTAL_STEPS)+499)//500,
        priority='Use one independently unlocked180 GPU; preserve all completed/cancelled campaigns.',
        evidence={
            'short_total_step': 258304, 'short_episodes': 100,
            'short_direct_entries': {'left': 11, 'right': 87, 'uncommitted': 2},
            'short_direct_successful_routes': {'right': 44},
            'short_mu_only_successful_routes': {'right': 80},
            'short_training_goal_visits': 0,
            'short_transitions_per_environment': 1009,
            'hypothesis': 'Test acquisition and minority retention after the changing-policy collector obtains more complete episodes; the short saved policy already succeeds on the right.',
            'limitations': ['No left success has been demonstrated.',
                            'Same seed is not independent replication.',
                            'The absence of collected goal terminals does not establish the cause of imbalance.']},
        screen='Inspect50k checkpoints; from500k stop if left entries are at most1/40 and left successes0 for three consecutive direct-policy evaluations. Stop on numerical/runtime failure without retry. Max1M post-warmup; no automatic extension. Preserve all failures and mu-only results.')
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=[HOST], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', PARENT, '--', *CORE_FILES], cwd=source, check=True)
    manifest = campaign_manifest(source, sha)
    manifest['algorithm_file_sha256'] = {p: hashlib.sha256((source/p).read_bytes()).hexdigest() for p in CORE_FILES}
    parent = Path('/home/heechan/optiq-experiments') / SHORT_CAMPAIGN
    original = json.loads((parent/'manifest.json').read_text())
    assert original['source_commit'] == SHORT_SOURCE
    control = next(job for job in original['jobs'] if job['teacher_std_floor'] == 1.)
    candidate = manifest['jobs'][0]
    changed = {k for k in set(control) | set(candidate) if control.get(k) != candidate.get(k)}
    assert changed == {'id', 'hypothesis', 'steps'}, changed
    result = json.loads((parent/'runs'/control['id']/'result.json').read_text())
    assert result['completed'] and result['source_commit'] == SHORT_SOURCE
    assert result['steps'] == 258304 and result['updates'] == 7816
    assert result['checkpoint']['readback_verified'] and result['checkpoint']['progress_replay_verified']
    manifest['comparison_validation'] = dict(changed_fields=sorted(changed), parent_completed=True,
                                            parent_checkpoint_sha256=result['checkpoint']['sha256'])
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    configs = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGN, 'controller'), (CAMPAIGN+'-wandb-sync', 'sync_wandb')]
    assert not root.exists(), root
    for service, _ in services:
        assert not (configs/(service+'.conf')).exists(), service
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
    ctl = ['/usr/local/bin/supervisorctl', '-c', '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'], check=True)
    for service, _ in services:
        subprocess.run(ctl+['update', service], check=True)
        subprocess.run(ctl+['start', service], check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(), host=args.host,
        services=[s for s, _ in services], source_commit=sha, jobs=[j['id'] for j in manifest['jobs']]), indent=2)+'\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha, jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
