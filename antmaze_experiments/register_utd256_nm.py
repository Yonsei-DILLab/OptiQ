"""Register the v3/v4 N=M128/256 comparison on three GPU hosts."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_utd256 import campaign_manifest as control_manifest


CAMPAIGN = 'antmaze-optiq-utd1-euclidean-NM128-256-v34-s0-20260925'
HOST_JOBS = {
    'vast-heechan-180': (('v3', 256),),
    'vast-heechan-199': (('v4', 256),),
    'vast1': (('v3', 128), ('v4', 128)),
}
TOTAL_STEPS = 1_000_192  # First 256-transition boundary at or above 1M.


def campaign_manifest(source, sha, host):
    if host not in HOST_JOBS:
        raise ValueError(host)
    jobs = []
    for task, n in HOST_JOBS[host]:
        shard = 0 if task == 'v3' else 1
        base = next(job for job in control_manifest(source, sha, shard, 'basic_euclidean')['jobs']
                    if job['task'] == task)
        job = dict(base, id=f'{task}-optiq-utd1-euclidean-NM{n}-s0',
                   nm=n, steps=TOTAL_STEPS)
        assert {key for key in base if base[key] != job[key]} == {'id', 'steps'}
        assert set(job) - set(base) == {'nm'}
        assert job['reward_specification']['formula'] == '100*(d(current)-d(next))'
        assert (job['temperature'], job['dacer'], job['noveld'], job['collection_profile']) == (1., 'off', 'off', 'env256-update256')
        assert job['optiq_config_profile'] == 'basic' and job['eval_starts'] == 'upstream'
        jobs.append(job)
    control = control_manifest(source, sha, 0, 'basic_euclidean')
    control.update(campaign=CAMPAIGN, host=host, shard=None, jobs=jobs,
        protocol='antmaze_experiments/UTD256_NM_PROTOCOL.md',
        condition='basic_euclidean_nm', nm_values=[n for _, n in HOST_JOBS[host]],
        total_transition_budget=TOTAL_STEPS, requested_budget=1_000_000,
        warmup_included=True, excluded_hosts=[],
        comparison='Only N=M and 1M aligned budget differ from the stopped T=1 basic Euclidean controls; concurrent T=3 runs are untouched.',
        changed_learning_setting={'num_policy_samples': '64 -> 128 or 256',
                                  'teacher_candidates': '64 -> 128 or 256',
                                  'total_steps': 'native 4M/5M -> 1000192'})
    return control


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=tuple(HOST_JOBS), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.host)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGN, 'controller'), (CAMPAIGN + '-wandb-sync', 'sync_wandb')]
    assert not root.exists(), root
    for name, _ in services:
        assert not (conf_root / (name + '.conf')).exists(), name
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, module in services:
        (conf_root / (name + '.conf')).write_text(f'''[program:{name}]
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
    for name, _ in services:
        subprocess.run(ctl + ['update', name], check=True)
        subprocess.run(ctl + ['start', name], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), host=args.host, services=[name for name, _ in services],
        source_commit=sha, jobs=[job['id'] for job in manifest['jobs']]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[job['id'] for job in manifest['jobs']])))


if __name__ == '__main__':
    main()
