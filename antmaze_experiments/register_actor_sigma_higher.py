"""Raise only an existing actor scale cap; retain the original initialization."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .register_actor_sigma_screen import campaign_manifest as lower_manifest
from .register_env32_screen import PINNED
from .register_horizon_temperature import PARENT
from .actor_sigma_profile import settings

CAMPAIGN='antmaze-optiq-sigmacap-higher-v34-250k-s0-20260925'
HOSTS=('vast-heechan-180','vast-heechan-199')
PROFILES=('cap0-initm1','cap1-initm1','cap2-initm1','cap3-initm1','uncapped-initm1')


def campaign_manifest(source,sha,host):
    assert host in HOSTS
    manifest=lower_manifest(source,sha)
    task='v3' if host==HOSTS[0] else 'v4'
    control=next(j for j in manifest['jobs'] if j['task']==task and j['actor_sigma_profile']=='capm2')
    jobs=[]
    for profile in PROFILES:
        job=copy.deepcopy(control)
        job.update(id=job['id'].replace('capm2',profile),actor_sigma_profile=profile,
                   hypothesis=task+'_'+profile)
        jobs.append(job)
    manifest.update(campaign=CAMPAIGN,host=host,shard=HOSTS.index(host),
        assigned_hosts=dict(enumerate(HOSTS)),jobs=jobs,
        protocol='antmaze_experiments/ACTOR_SIGMA_HIGHER_PROTOCOL.md',
        comparison_scope='Only actor log-sigma upper bound changes; initial log sigma remains -1 in every condition',
        actor_sigma_profiles={p:settings(p) for p in PROFILES},
        discount_temperature_conditions={j['hypothesis']:(j['discount'],j['temperature']) for j in jobs},
        global_job_count=10,per_host_job_count=5,
        priority='Use independently free5090GPU slots, backfill every2seconds; no all-maze or all-method barriers. Preserve all existing jobs/results.',
        evidence={
            'lower_caps':'Changing cap AND initial to -2/-3 worsened v4 route balance; final direct successes were0/100 for both.',
            'observed_scale':'Always-logged actor_std_mean approaches the respective cap in -1/-2/-3 runs.',
            'v3_control':'At258304, default cap-1 gives44right/0left direct successes versus80right/0left mu-only.',
            'teacher_diagnostic':'Frozen8state x8cloud diagnostic gave M64 ESS averages32.6(v3)/26.2(v4); it does not establish a one-candidate collapse or justify changing N/M.',
            'hypothesis':'A higher permitted conditional scale may preserve useful exploration without reducing the initial scale.',
            'risks':['Higher conditional noise may impair accurate movement and goal acquisition.',
                     'The mixture may represent spread through sigma instead of distinct latent means.',
                     'DACER adaptation can respond to policy entropy; its settings remain unchanged.',
                     'A cap-saturated optimizer is not by itself evidence that raising the cap improves returns.',
                     'Unbounded raw sigma can exceed float32 inverse-CDF precision; existing finite checks stop a numerical failure without adding a hidden cap.']},
        initial_parameters_exactly_equal_control=True,
        screen='250k postwarmup;40direct+40native each50k;100final, original fixed full start. Both successful routes required. No automatic extension/retry.')
    return manifest


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--host',choices=HOSTS,required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
    manifest=campaign_manifest(source,sha,args.host)
    manifest['algorithm_file_sha256']={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in PINNED}
    if args.dry_run:print(json.dumps(manifest,indent=2));return
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    configs=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services=[(CAMPAIGN,'controller'),(CAMPAIGN+'-wandb-sync','sync_wandb')]
    assert not root.exists()
    for name,_ in services:assert not (configs/(name+'.conf')).exists()
    root.mkdir();(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
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
    registration=dict(time=time.time(),host=args.host,source_commit=sha,
                      services=[n for n,_ in services],jobs=[j['id'] for j in manifest['jobs']])
    (root/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    print(json.dumps(registration))


if __name__=='__main__':main()
