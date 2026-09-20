"""Four independent bounded runs on host180, one per GPU under Supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=Path('/home/heechan/optiq-experiments/trg-cap-20k-20260921')
PLAN=[('ant',0.),('ant',-2.),('humanoid',0.),('humanoid',-2.)]
PREFIX='trg-cap-20k-20260921'


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def source():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def setup():
    sha=source();ROOT.mkdir(exist_ok=False)
    save(ROOT/'manifest.json',dict(commit=sha,source=str(REPO),host='180',initial_sigma=.1,
        log_sigma_min=-5,log_sigma_caps=[0,-2],total_steps=20000,seed=0,jobs=PLAN))
    names=[]
    for gpu in range(4):
        name=f'{PREFIX}-gpu{gpu}';names.append(name)
        path=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
        assert not path.exists()
        path.write_text(f'''[program:{name}]
command=/home/heechan/OptiQ-ops/run-gpu.sh {gpu} --branch v5-direct-gmm /home/heechan/.venv-optiq-mujoco/bin/python -u {HERE}/launch.py worker {gpu}
directory={REPO}
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=45
redirect_stderr=true
stdout_logfile={ROOT}/worker-gpu{gpu}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
environment=OPTIQ_SOURCE_DIR="{REPO}",PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online"
''')
    ctl=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'],check=True)
    for name in names:subprocess.run(ctl+['update',name],check=True)
    print(json.dumps(dict(root=str(ROOT),commit=sha,services=names)))


def worker(gpu):
    sha=source();task,cap=PLAN[gpu]
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    name=f'{task}-trg-cap{cap:g}-sigma0.1-20k-s0'
    path=ROOT/(name+'.json');assert not path.exists()
    args=[sys.executable,'-u',str(HERE/'train_cap.py'),str(cap),f'benchmark={task}',
        f'run_name={name}',f'output_root={ROOT}/outputs']
    job=dict(task=task,cap=cap,sigma=.1,seed=0,host='180',gpu=gpu,commit=sha,
        command=args,status='running',started=time.time())
    save(path,job)
    with (ROOT/(name+'.log')).open('x') as log:
        rc=subprocess.run(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    paths=list((ROOT/'outputs').glob(name+'_*/completed.json'))
    job.update(exit_code=rc,finished=time.time(),status='failed')
    if len(paths)==1:
        result=json.loads(paths[0].read_text());job.update(result,run_dir=str(paths[0].parent))
        if result['timesteps']==20000 and rc==0:job['status']='completed'
    save(path,job);print(json.dumps(job),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    else:worker(int(sys.argv[2]))
