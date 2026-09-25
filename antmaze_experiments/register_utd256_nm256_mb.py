"""Register N=M256 full-batch gradient accumulation after the OOM preflights."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_utd256_nm import campaign_manifest as nm_manifest


CAMPAIGN = 'antmaze-optiq-utd1-euclidean-NM256-microbatch-v34-s0-20260925'
FAILED_SOURCE = 'b0df33807a05debd3df47536c34058b0962d6c8f'
HOSTS = {'vast-heechan-180': 'v3', 'vast-heechan-199': 'v4'}


def campaign_manifest(source, sha, host):
    manifest = nm_manifest(source, sha, host)
    assert len(manifest['jobs']) == 1
    job = manifest['jobs'][0]
    assert job['task'] == HOSTS[host] and job['nm'] == 256
    job['id'] = f'{job["task"]}-optiq-utd1-euclidean-NM256-microbatch-s0'
    job['actor_microbatch_size'] = 256
    manifest.update(campaign=CAMPAIGN,
        protocol='antmaze_experiments/UTD256_NM256_MICROBATCH_PROTOCOL.md',
        comparison_source=FAILED_SOURCE,
        previous_preflight='N=M256 batch4096 full tensor OOM at first learner update on each RTX5090; no main training started',
        actor_microbatch_size=256,
        effective_actor_batch_size=4096,
        actor_optimizer_steps_per_256_transitions=256,
        other_jobs_preserved='N=M128 v3/v4 on vast1 and T=3 v3/v4 on RTX5090 hosts continue unchanged')
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=tuple(HOSTS), required=True)
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
