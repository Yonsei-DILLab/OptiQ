"""Register matched random-start v3/v4 basic OptiQ runs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_utd256 import campaign_manifest as control_manifest
from .settings import total_budget


CAMPAIGN = 'antmaze-optiq-utd1-basic-euclidean-random-starts-v34-s0-20260925'
HOST_TASK = {'vast-heechan-180': ('v3', 0), 'vast-heechan-199': ('v4', 1)}


def campaign_manifest(source, sha, host):
    task, shard = HOST_TASK[host]
    manifest = control_manifest(source, sha, shard, 'basic_euclidean')
    base = next(job for job in manifest['jobs'] if job['task'] == task)
    job = dict(base, id=f'{task}-optiq-utd1-basic-euclidean-random-starts-s0',
               train_starts='random', eval_starts='random')
    assert job['steps'] == total_budget(task)
    assert job['temperature'] == 1. and job['dacer'] == job['noveld'] == 'off'
    assert job['optiq_config_profile'] == 'basic' and job['collection_profile'] == 'env256-update256'
    assert job['reward_specification']['formula'] == '100*(d(current)-d(next))'
    assert job['interim_eval_episodes'] == 40 and job['eval_interval'] == 50000
    assert job['optiq_profile']['actor_hidden_dims'] == [256, 256]
    assert job['optiq_profile']['critic_hidden_dims'] == [256, 256]
    assert {key for key in base if base[key] != job[key]} == {'id', 'eval_starts'}
    assert set(job) - set(base) == {'train_starts'}
    manifest.update(campaign=CAMPAIGN, host=host, jobs=[job],
        protocol='antmaze_experiments/RANDOM_STARTS_V34_PROTOCOL.md',
        condition='basic_euclidean_matched_random_starts',
        evaluation_starts='v3/v4: fresh upstream random XY[-2,2] at each reset, matching training',
        training_starts='v3/v4: upstream random_init=True, as used for native v1',
        comparison='Fresh T=1/N=M64 basic Euclidean controls with v3/v4 training and evaluation resets randomized; T=3 runs were stopped and are not matched controls.',
        changed_learning_setting={'train_starts': 'upstream fixed -> native v1-style random XY[-2,2]'},
        previous_t3_stopped=True, existing_training_preserved=True,
        excluded_hosts=['vast1'])
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=tuple(HOST_TASK), required=True)
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
