"""Register the temperature-only T=3 follow-up to the basic Euclidean UTD=1 run."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_utd256 import HOSTS, campaign_manifest as control_manifest


CAMPAIGN = 'antmaze-optiq-utd1-basic-euclidean-T3-v34-s0-20260925'
CONTROL_CAMPAIGN = 'antmaze-optiq-utd1-basic-euclidean-v1234-s0-20260925'
CONTROL_SOURCE = 'df70ccf94d268478314a5dc23a2fc5d724549ce0'
TASKS = {0: 'v3', 1: 'v4'}


def campaign_manifest(source, sha, shard):
    if shard not in TASKS:
        raise ValueError(shard)
    manifest = control_manifest(source, sha, shard, 'basic_euclidean')
    base = next(job for job in manifest['jobs'] if job['task'] == TASKS[shard])
    job = dict(base, id=f'{TASKS[shard]}-optiq-utd1-basic-euclidean-T3-s0', temperature=3.)
    assert {key for key in base if base[key] != job[key]} == {'id', 'temperature'}
    assert job['reward_specification']['formula'] == '100*(d(current)-d(next))'
    assert job['eval_starts'] == 'upstream'
    assert job['collection_profile'] == 'env256-update256'
    assert job['optiq_config_profile'] == 'basic'
    manifest.update(
        campaign=CAMPAIGN, jobs=[job],
        protocol='antmaze_experiments/UTD256_T3_PROTOCOL.md',
        condition='basic_euclidean_temperature3',
        temperature=3.,
        comparison_campaign=CONTROL_CAMPAIGN,
        comparison_source=CONTROL_SOURCE,
        comparison_task=TASKS[shard],
        changed_learning_setting={'temperature': {'control': 1., 'test': 3.}},
        comparison='T=1 v3/v4 had no successes in 40 direct-policy episodes at the last reviewed evaluations; the entire T=1 set was stopped at user request.',
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=tuple(TASKS), required=True)
    parser.add_argument('--host', choices=tuple(HOSTS.values()), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert HOSTS[args.shard] == args.host
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.shard)
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
