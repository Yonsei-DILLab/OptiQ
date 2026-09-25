"""Register random-start OptiQ v3/v4 reward comparison, N=M=64."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .progress_reward import EUCLIDEAN_SCALE100_COST01_PROFILE, specification
from .register_utd256 import campaign_manifest as control_manifest
from .settings import total_budget


CAMPAIGN = 'antmaze-optiq-utd1-random-reward-pair-v34-s0-20260925'
HOST_TASKS = {
    'vast-heechan-180': ('v3',),
    'vast-heechan-199': ('v4',),
}
REWARDS = (
    ('progress100_cost01', EUCLIDEAN_SCALE100_COST01_PROFILE),
    ('negative_distance', 'dense'),
)


def campaign_manifest(source, sha, host):
    if host not in HOST_TASKS:
        raise ValueError(host)
    shard = 0 if host == 'vast-heechan-180' else 1
    manifest = control_manifest(source, sha, shard, 'basic_euclidean')
    originals = {job['task']: job for job in manifest['jobs']}
    jobs = []
    for task in HOST_TASKS[host]:
        for label, reward in REWARDS:
            job = copy.deepcopy(originals[task])
            job.update(id=f'{task}-optiq-utd1-random-{label}-nm64-s0',
                       reward_profile=reward, train_starts='random',
                       eval_starts='random')
            job.pop('reward_specification', None)
            if reward != 'dense':
                job['reward_specification'] = specification(task, reward)
            assert job['steps'] == total_budget(task)
            assert (job['temperature'], job['dacer'], job['noveld']) == (1., 'off', 'off')
            assert job['collection_profile'] == 'env256-update256'
            assert job['optiq_config_profile'] == 'basic'
            assert job['optiq_profile']['actor_hidden_dims'] == [256, 256]
            assert job['optiq_profile']['critic_hidden_dims'] == [256, 256]
            assert job['eval_interval'] == 50000 and job['interim_eval_episodes'] == 40
            jobs.append(job)
    assert len(jobs) == 2 and len({job['id'] for job in jobs}) == 2
    manifest.update(
        campaign=CAMPAIGN, host=host, shard=shard, jobs=jobs,
        condition='random_start_reward_pair',
        protocol='antmaze_experiments/RANDOM_REWARD_PAIR_PROTOCOL.md',
        reward_profiles={label: reward for label, reward in REWARDS},
        training_starts='all tasks: upstream random_init=True, XY[-2,2] sampled at each reset',
        evaluation_starts='all tasks: fresh random starts matching training; fixed-full-state supplementary',
        comparison='Fresh reward comparison, not directly matched to stopped fixed-start/N128/T3 runs',
        nm=64, excluded_hosts=['vast1'],
        cancelled_predecessors=[
            'antmaze-optiq-utd1-basic-euclidean-random-starts-v34-s0-20260925',
            'antmaze-optiq-utd1-euclidean-NM128-256-v34-s0-20260925',
            'antmaze-optiq-utd1-basic-euclidean-T3-v34-s0-20260925',
        ],
    )
    manifest.pop('reward_profile', None)
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=tuple(HOST_TASKS), required=True)
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
