"""Four bounded screens of existing conditional Gaussian scale parameters."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_teacher_floor_screen import campaign_manifest as v3_manifest
from .register_v4_teacher_floor import campaign_manifest as v4_manifest
from .register_env32_screen import PINNED
from .register_horizon_temperature import PARENT
from .actor_sigma_profile import settings

CAMPAIGN = 'antmaze-optiq-sigmacap-v34-250k-s0-20260925'
HOST = 'vast-heechan-199'
# Identical in the actual initial audits of all four O/P teacher-floor controls.
INITIAL = {
    'actor': {'sha256': '58fd041f7ff1c40ac56f7fa5ad311ba8bc0f39ff61c0fa1932c0db945c3713fc', 'parameters': 145424},
    'critic': {'sha256': '5ceb33a400875e63fb44f5d7123248179a94f15ecf0b9479fb21c05612219f52', 'parameters': 283138},
}


def campaign_manifest(source, sha):
    v3 = v3_manifest(source, sha)
    manifest = v4_manifest(source, sha)
    controls = [next(j for j in v3['jobs'] if j['teacher_std_floor'] == 1.),
                next(j for j in manifest['jobs'] if j['teacher_std_floor'] == .5)]
    jobs = []
    for profile in ('capm2', 'capm3'):
        for control in controls:
            job = copy.deepcopy(control)
            job.update(id=control['id'].replace('-250k-', '-'+profile+'-250k-'),
                       hypothesis=control['task']+'_'+profile, actor_sigma_profile=profile)
            assert {k for k in set(job)|set(control) if job.get(k)!=control.get(k)} == {'id','hypothesis','actor_sigma_profile'}
            jobs.append(job)
    for key in ('teacher_floor_grid','control_teacher_floor','parent_source','comparison_source','comparison_campaign'):
        manifest.pop(key, None)
    manifest.update(campaign=CAMPAIGN, jobs=jobs, host=HOST, shard=0,
        protocol='antmaze_experiments/ACTOR_SIGMA_SCREEN_PROTOCOL.md',
        priority_campaign=None, assigned_hosts={0:HOST}, excluded_hosts=['vast1'],
        comparison_scope='Only actor log-sigma cap and initial log-sigma change together; random latent and teacher proposal floor unchanged',
        initial_parameter_control=INITIAL,
        controls={
            'v3':dict(source='d25930197ee9ba060a3aa84c3b6fea419dfe856d',host='vast-heechan-180',
                      campaign='antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2',job=controls[0]['id']),
            'v4':dict(source='555bb7e3c0101ee939545cc35d4d0729291781ec',host=HOST,
                      campaign='antmaze-optiq-v4-teacherfloor-250k-s0-20260925',job=controls[1]['id'])},
        discount_temperature_conditions={j['hypothesis']:(j['discount'],j['temperature']) for j in jobs},
        actor_sigma_profiles={p:settings(p) for p in ('capm2','capm3')},
        control_actor_sigma=settings(None), num_envs=256, updates_per_vector_step=8,
        total_transitions_per_job=258304, expected_learner_updates=7816,
        expected_regulator_updates=16, fresh_start=True, checkpoint_resume=False, seed=0,
        priority='Independent free199GPU slots; preserve env32 final evaluations on180 and all completed/cancelled artifacts',
        evidence={
            'v3_control':'Direct44/100 versus mu-only80/100 goal successes; only right succeeds. Actor sigma is near the .3679 cap in inspected logs.',
            'v4_control':'Both entrances occur, but direct and mu-only final100 have zero goals. Contact diagnostic suggests slow net progress, not sustained wall contact.',
            'failed_screens':'Fixed64 gives v3 direct15right/0left successes and v4zero. env32 at250112 has v3right16/40 and v4lower40/40 with no goal.',
            'hypothesis':'Smaller state-conditional Gaussian spread may improve precise locomotion while random latent retains distinct action modes.',
            'risks':['Reduced conditional spread may impair exploration or accelerate route concentration.',
                     'DACER remains adaptive and can respond to changed policy entropy.',
                     'Cap and initial sigma change together; they are not isolated effects.',
                     'One seed and250k do not establish retention or robustness.']},
        screen='250k post-warmup only;40direct and40mu-only each50k;100final and full state. Both successful routes and later retention required; no automatic extension/retry.')
    return manifest


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--host', choices=[HOST], required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
    manifest = campaign_manifest(source,sha)
    manifest['algorithm_file_sha256'] = {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in PINNED}
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    root = Path('/home/heechan/optiq-experiments')/CAMPAIGN
    configs = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGN,'controller'), (CAMPAIGN+'-wandb-sync','sync_wandb')]
    assert not root.exists()
    for name,_ in services:
        assert not (configs/(name+'.conf')).exists()
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,module in services:
        (configs/(name+'.conf')).write_text(f'''[program:{name}]
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
    ctl = ['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'],check=True)
    for name,_ in services:
        subprocess.run(ctl+['update',name],check=True)
        subprocess.run(ctl+['start',name],check=True)
    registration = dict(time=time.time(),host=HOST,source_commit=sha,
                        services=[n for n,_ in services],jobs=[j['id'] for j in manifest['jobs']])
    (root/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    print(json.dumps(registration))


if __name__ == '__main__':
    main()
