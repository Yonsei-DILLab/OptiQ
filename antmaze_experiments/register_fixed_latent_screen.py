"""Two existing fixed-codebook configuration screens; learning implementation pinned."""
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

CAMPAIGN = 'antmaze-optiq-fixed64-v34-250k-s0-20260925'
HOST = 'vast-heechan-199'
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
        job.update(id=control['id'].replace('-250k-', '-fixed64-250k-'),
                   hypothesis=control['task']+'_fixed64',latent_profile='fixed64')
        assert {k for k in set(job)|set(control) if job.get(k)!=control.get(k)}=={'id','hypothesis','latent_profile'}
        jobs.append(job)
    for key in ('teacher_floor_grid','control_teacher_floor','parent_source','comparison_source','comparison_campaign'):
        manifest.pop(key,None)
    manifest.update(campaign=CAMPAIGN,jobs=jobs,host=HOST,shard=0,
        protocol='antmaze_experiments/FIXED_LATENT_SCREEN_PROTOCOL.md',
        priority_campaign=None,assigned_hosts={0:HOST},excluded_hosts=['vast1'],
        comparison_scope='Only existing finite64 latent prior differs from corresponding completed random-prior control; same seed/budget/settings',
        controls={
            'v3':dict(source='d25930197ee9ba060a3aa84c3b6fea419dfe856d',host='vast-heechan-180',
                      campaign='antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2',job=controls[0]['id']),
            'v4':dict(source='555bb7e3c0101ee939545cc35d4d0729291781ec',host=HOST,
                      campaign='antmaze-optiq-v4-teacherfloor-250k-s0-20260925',job=controls[1]['id'])},
        discount_temperature_conditions={j['hypothesis']:(j['discount'],j['temperature']) for j in jobs},
        latent_profile='fixed64',latent_components=64,latent_codebook_seed=20260911,
        fresh_start=True,checkpoint_resume=False,seed=0,
        priority='Independent free199GPU slots; preserve live v3 retention on180 and all completed/cancelled artifacts',
        evidence={
            'v3_random_control':'258304total direct100: right44 successful, left0; entries right87/left11/uncommitted2. Longer same-seed direct400128: right39/left1, right35 successful.',
            'v4_random_control':'258304total direct100: upper71/lower29, no goals. Paired40 CPU diagnostic upper26/lower14, no goals; little net progress, not sustained wall contact.',
            'hypothesis':'A fixed finite64 support may reduce changing-latent marginal approximation variability and stabilize component specialization. This is unproven and need not preserve long-horizon routes.',
            'risks':['The prior changes from continuous Gaussian to the existing centered/scaled uniform finite codebook.',
                     'Action entropy or codebook diversity is not successful trajectory diversity.',
                     'A single training seed cannot establish robustness.']},
        screen='250k post-warmup only;40direct and40mu-only each50k;100final per mode. Require both successful routes, not entries alone. No automatic extension/retry.',
        deterministic_evaluation='component0_mu, not zero latent; direct/native select uniformly from the same fixed codebook each action')
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
