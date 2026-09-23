"""Register the approved fresh 16-policy dense/NovelD-off campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .settings import total_budget, WANDB_ENTITY, WANDB_PROJECT
from .dependencies import prepare_dependencies, verify_dependencies

CAMPAIGN = 'antmaze-dense-off-16-current-s0-20260924'
HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}


def campaign_manifest(source, sha, shard):
    tasks = ('v1', 'v3') if shard == 0 else ('v2', 'v4')
    # Put slow DIPO jobs on GPUs immediately; completed slots independently
    # backfill from the remaining jobs without any environment/method barrier.
    jobs = []
    for method in ('dipo', 'optiq', 'sac', 'mfpo'):
        for task in tasks:
            entry = dict(id=f'{task}-{method}-s0', task=task, method=method,
                         reward_profile='dense', noveld='off', eval_starts='random',
                         steps=total_budget(task), interim_eval_episodes=40,
                         final_eval_episodes=100)
            if method == 'optiq':
                entry.update(temperature=.01, save_intermediate_policy=True,
                    optiq_profile=dict(actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256], actor_lr=3e-4,
                        critic_lr=5e-4))
            jobs.append(entry)
    return dict(campaign=CAMPAIGN, shard=shard, host=HOSTS[shard],
        source=str(source), source_commit=sha, jobs=jobs,
        wandb_mode='online', wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT,
        protocol='antmaze_experiments/DENSE_OFF_16_CURRENT_PROTOCOL.md',
        seed=0, num_envs=256, batch_size=4096, updates_per_vector_step=8,
        reward_profile='dense', noveld_enabled=False, noveld_coefficient=0.,
        reward='negative Euclidean distance from next xy to nearest goal; no sparse bonus',
        evaluation_starts='xy uniform[-2,2] per episode; native training resets unchanged',
        interim_eval_episodes=40, final_eval_episodes=100,
        optiq_temperature=.01, optiq_actor_hidden_dims=[256, 256, 256],
        optiq_critic_hidden_dims=[256, 256, 256], optiq_actor_lr=3e-4,
        optiq_critic_lr=5e-4, log_std_min=-5., log_std_max=-1., initial_log_std=-1.,
        optiq_tau=.005, dipo_dense_value_support=[-6000., 5.],
        parent_source='cac1365fd9878d46874bbc09a840816ba1461499',
        fresh_training=True, resume_cancelled_jobs=False,
        backfill_seconds=2, failure_holds_pending=True, automatic_restart=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--host', choices=list(HOSTS.values()), required=True)
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
        manifest['source_dependencies'] = verify_dependencies(source, ('mfpo',))
        print(json.dumps(manifest, indent=2))
        return
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    conf = Path('/home/heechan/OptiQ-ops/supervisor/jobs') / (CAMPAIGN + '.conf')
    assert not root.exists(), root
    assert not conf.exists(), conf
    manifest['source_dependencies'] = prepare_dependencies(source, ('mfpo',))
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    conf.write_text(f'''[program:{CAMPAIGN}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl = ['/usr/local/bin/supervisorctl', '-c',
           '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    for command in (['reread'], ['update', CAMPAIGN], ['start', CAMPAIGN]):
        subprocess.run(ctl + command, check=True)
    (root / 'registration.json').write_text(json.dumps(dict(time=time.time(),
        supervisor=CAMPAIGN, manifest=manifest, config=str(conf)), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
