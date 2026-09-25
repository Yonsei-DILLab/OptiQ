"""Register bounded CPU inference on two existing teacher-floor controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from .register_env32_screen import PINNED
from .register_horizon_temperature import PARENT

CAMPAIGN='antmaze-teacher-mc-v34-20260925'
CONTROLS={
    'vast-heechan-180':dict(task='v3',source='d25930197ee9ba060a3aa84c3b6fea419dfe856d',
        campaign='antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2',
        job='v3-optiq-startnorm-geodesic-T3-teacherfloor1-250k-s0',temperature=3.,floor=1.),
    'vast-heechan-199':dict(task='v4',source='555bb7e3c0101ee939545cc35d4d0729291781ec',
        campaign='antmaze-optiq-v4-teacherfloor-250k-s0-20260925',
        job='v4-optiq-geodesic-T1-teacherfloor0.5-250k-s0',temperature=1.,floor=.5),
}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--host',choices=list(CONTROLS),required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/commit
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    subprocess.run(['git','diff','--exit-code',PARENT,'--',*PINNED],cwd=source,check=True)
    control=CONTROLS[args.host]
    run=Path('/home/heechan/optiq-experiments')/control['campaign']/'runs'/control['job']
    cfg=json.loads((run/'config.json').read_text())
    result=json.loads((run/'result.json').read_text())
    proof=json.loads((run/'checkpoint-verification.json').read_text())
    assert result['completed'] and result['steps']==proof['steps']==258304
    assert cfg['source_commit']==control['source'] and cfg['temperature']==control['temperature']
    assert cfg['teacher_std_floor_override']==control['floor'] and cfg['task']==control['task']
    assert proof['readback_verified'] and proof['environment_reward_verified']
    with (run/'checkpoint-final.pt').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==proof['sha256']
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    environment=dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore',
        MPLBACKEND='Agg',WANDB_MODE='disabled',USE_FLAX='0',USE_TORCH='1',
        D4RL_SUPPRESS_IMPORT_ERROR='1',MUJOCO_PY_FORCE_CPU='1',
        LD_LIBRARY_PATH='/home/heechan/.mujoco/mujoco210/bin',
        OPTIQ_SOURCE_DIR='/home/heechan/OptiQ-ops/sources/'+control['source'])
    command=['/usr/bin/nice','-n','19','/home/heechan/.venv-ddiffpg-native/bin/python','-u',
        str(source/'antmaze_experiments/diagnose_teacher_mc.py'),'--run',str(run),
        '--output',str(root/'diagnostic')]
    record=dict(time=time.time(),host=args.host,service=CAMPAIGN,command=command,
        environment=environment,reporting_source=commit,training_source=control['source'],
        control=control,checkpoint_sha256=proof['sha256'],inference_only=True,
        states=8,repeats=8,candidates=[64,256,1024,4096],reference_candidates=4096,
        algorithm_file_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in PINNED},
        protocol='antmaze_experiments/TEACHER_MC_DIAGNOSTIC_PROTOCOL.md',
        training_or_rollout_performed=False,automatic_retry=False)
    if args.dry_run:print(json.dumps(record,indent=2));return
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(CAMPAIGN+'.conf')
    assert not root.exists() and not conf.exists()
    root.mkdir()
    launch=root/'launch.sh'
    launch.write_text('#!/bin/bash\nset -euo pipefail\n'
        +'\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())
        +'\ncd '+shlex.quote(str(source))+'\nexec '+shlex.join(command)+'\n')
    conf.write_text(f'''[program:{CAMPAIGN}]
command=/bin/bash {launch}
directory={source}
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/diagnostic.log
stderr_logfile={root}/diagnostic.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    (root/'registration.json').write_text(json.dumps(record,indent=2)+'\n')
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'],check=True)
    subprocess.run(ctl+['update',CAMPAIGN],check=True)
    subprocess.run(ctl+['start',CAMPAIGN],check=True)
    print(json.dumps(dict(root=str(root),reporting_source=commit,training_source=control['source'])))


if __name__=='__main__':
    main()
