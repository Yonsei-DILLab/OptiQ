"""Bounded gamma/temperature hypotheses; preserve all algorithm source files."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .register_dacer_positive import campaign_manifest as target_manifest, CAMPAIGN as PRIORITY_CAMPAIGN
from .settings import WARMUP, NUM_ENVS, expected_updates

CAMPAIGN='antmaze-optiq-horizon-temperature-250k-s0-20260925'
PARENT='2564b59faa0d319eece496b93eff0f19359efc37'
HOSTS={0:'vast-heechan-180',1:'vast-heechan-199'}
POST_WARMUP_BUDGET=250000
TOTAL_STEPS=WARMUP+(POST_WARMUP_BUDGET//NUM_ENVS+1)*NUM_ENVS
CONDITIONS={'gamma999':(.999,1.),'temp3':(.99,3.),'gamma999_temp3':(.999,3.)}
SHARDS={0:(('v3','gamma999'),('v4','temp3'),('v1','gamma999'),('v3','gamma999_temp3')),
        1:(('v4','gamma999'),('v3','temp3'),('v1','gamma999_temp3'),('v4','gamma999_temp3'))}
CORE_FILES=tuple('analysis_tools/experiments/20260920_truncated_mll/optiq_dime/'+p for p in
    ('algorithm.py','policy.py','distillation.py','semi_implicit.py','transport.py'))+(
    'analysis_tools/studies/20260918_nonstationary_nd/v5/diffusion/dime.py',
    'analysis_tools/experiments/20260921_gmm_trg_sweep/regulator.py')


def campaign_manifest(source,sha,shard):
    if shard not in SHARDS:raise ValueError(shard)
    manifest=target_manifest(source,sha,shard)
    prototypes={j['task']:j for part in (0,1) for j in target_manifest(source,sha,part)['jobs']
                if j['dacer_target_entropy_per_dim']==.7}
    jobs=[]
    for task,condition in SHARDS[shard]:
        discount,temperature=CONDITIONS[condition]
        job=copy.deepcopy(prototypes[task])
        job.update(id=f'{task}-optiq-{condition}-H0.7-i500-250k-s0',
                   hypothesis=condition,discount=discount,temperature=temperature,
                   steps=TOTAL_STEPS,eval_interval=50000)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN,protocol='antmaze_experiments/HORIZON_TEMPERATURE_PROTOCOL.md',
        parent_source=PARENT,comparison_source=PARENT,comparison_campaign=PRIORITY_CAMPAIGN,
        comparison_scope='Current H/d+.7, gamma.99, T1 at matched checkpoints; one seed, exploratory screen',
        priority_campaign='/home/heechan/optiq-experiments/'+PRIORITY_CAMPAIGN,
        jobs=jobs,target_entropy_per_dim_grid=[.7],target_entropy_grid=[5.6],
        optiq_temperature='per-job',discount_temperature_conditions=CONDITIONS,
        post_warmup_budget=POST_WARMUP_BUDGET,actual_post_warmup_transitions=TOTAL_STEPS-WARMUP,
        total_transitions_per_job=TOTAL_STEPS,expected_learner_updates=expected_updates(TOTAL_STEPS),
        expected_regulator_updates=(expected_updates(TOTAL_STEPS)+499)//500,
        eval_interval=50000,algorithm_source_unchanged_from=PARENT,
        priority='Older pending target-grid jobs receive GPU slots first; then independent per-slot backfill without waiting for all older running jobs',
        screen='250k fixed budget; compare successful routes, minority route retention and goal distances at 50k checkpoints; no success claim from mere route entry',
        evidence={'v3_control_400k':{'left_episodes':8,'right_episodes':27,'mean_route_gap_gamma099':106.0,'mean_route_gap_gamma0999':34.0},
                  'limitation':'Re-scoring saved trajectories is not a counterfactual training prediction; gamma changes the MDP objective.'})
    manifest['diagnostics']['policy_checkpoints']='50k intervals; evaluation only, final full state at 258304 total transitions'
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=list(HOSTS), required=True)
    parser.add_argument('--host', choices=list(HOSTS.values()), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if HOSTS[args.shard] != args.host:
        parser.error('host and shard must match the fixed 5090-only assignment')
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.shard)
    subprocess.run(['git','diff','--exit-code',PARENT,'--',*CORE_FILES],cwd=source,check=True)
    manifest['algorithm_file_sha256']={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in CORE_FILES}
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
        time=time.time(), services=[s for s, _ in services], manifest=manifest), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                         jobs=[entry['id'] for entry in manifest['jobs']])))


if __name__ == '__main__':
    main()
