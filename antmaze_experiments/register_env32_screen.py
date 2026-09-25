"""Two collection-cadence screens; global update ratio and learning code pinned."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_teacher_floor_screen import campaign_manifest as v3_manifest
from .register_v4_teacher_floor import campaign_manifest as v4_manifest
from .register_horizon_temperature import CORE_FILES, PARENT

CAMPAIGN = 'antmaze-optiq-env32-v34-250k-s0-20260925'
HOST = 'vast-heechan-180'
PINNED = CORE_FILES + (
    'analysis_tools/experiments/20260920_truncated_mll/optiq_dime/latent.py',
    'analysis_tools/experiments/20260920_truncated_mll/optiq_dime/box_gaussian.py')


def campaign_manifest(source, sha):
    v3 = v3_manifest(source, sha)
    manifest = v4_manifest(source, sha)
    controls = [next(j for j in v3['jobs'] if j['teacher_std_floor']==1.),
                next(j for j in manifest['jobs'] if j['teacher_std_floor']==.5)]
    jobs=[]
    for control in controls:
        job=copy.deepcopy(control)
        job.update(id=control['id'].replace('-250k-', '-env32-250k-'),
                   hypothesis=control['task']+'_env32',collection_profile='env32-update1')
        assert {k for k in set(job)|set(control) if job.get(k)!=control.get(k)}=={'id','hypothesis','collection_profile'}
        jobs.append(job)
    for key in ('teacher_floor_grid','control_teacher_floor','parent_source','comparison_source','comparison_campaign'):
        manifest.pop(key,None)
    manifest.update(campaign=CAMPAIGN,jobs=jobs,host=HOST,shard=0,
        protocol='antmaze_experiments/ENV32_SCREEN_PROTOCOL.md',
        priority_campaign=None,assigned_hosts={0:HOST},excluded_hosts=['vast1'],
        comparison_scope='Only collection cadence256/eight to32/one; same global1/32 update ratio, seed, budget and learning settings',
        controls={
            'v3':dict(source='d25930197ee9ba060a3aa84c3b6fea419dfe856d',host=HOST,
                      campaign='antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2',job=controls[0]['id']),
            'v4':dict(source='555bb7e3c0101ee939545cc35d4d0729291781ec',host='vast-heechan-199',
                      campaign='antmaze-optiq-v4-teacherfloor-250k-s0-20260925',job=controls[1]['id'])},
        discount_temperature_conditions={j['hypothesis']:(j['discount'],j['temperature']) for j in jobs},
        collection_profile='env32-update1',num_envs=32,updates_per_vector_step=1,
        updates_per_transition=1/32,eval_transition_quantum=256,
        actual_intermediate_eval_steps=[50176,100096,150016,200192,250112],
        total_transitions_per_job=258304,expected_learner_updates=7816,
        expected_regulator_updates=16,per_environment_transitions=8072,
        fresh_start=True,checkpoint_resume=False,seed=0,
        priority='Independent free180GPU slots; preserve fixed64 runs on199 and all completed/cancelled artifacts',
        evidence={
            'v3_control':'Zero training goal visits through258304 despite44right/0left final direct successes; longer control loses all left entries.',
            'v4_control':'Direct100 upper71/lower29 entries, no goals. Paired40 native700 rollouts move locally but progress slowly near the end.',
            'collection_budget':'256envs advance1009times each by258304global transitions. 32envs advance8072times; globalupdates remain7816.',
            'hypothesis':'Fewer updates during each individual trajectory and more completed trajectories may help goal acquisition before one side disappears.',
            'risks':['Replay order, cross-environment decorrelation and within-trajectory policy drift all change.',
                     'Less concurrency can be slower in wall time.',
                     'One training seed cannot establish robustness.']},
        screen='250k post-warmup only;40direct and40mu-only at matched global steps;100final. Both successful routes required; no automatic extension/retry.')
    return manifest


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--host',choices=[HOST],required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
    manifest=campaign_manifest(source,sha)
    manifest['algorithm_file_sha256']={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in PINNED}
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    configs=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services=[(CAMPAIGN,'controller'),(CAMPAIGN+'-wandb-sync','sync_wandb')]
    assert not root.exists()
    for name,_ in services:assert not (configs/(name+'.conf')).exists()
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
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'],check=True)
    for name,_ in services:
        subprocess.run(ctl+['update',name],check=True)
        subprocess.run(ctl+['start',name],check=True)
    registration=dict(time=time.time(),host=HOST,source_commit=sha,
                      services=[n for n,_ in services],jobs=[j['id'] for j in manifest['jobs']])
    (root/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    print(json.dumps(registration))


if __name__=='__main__':main()
