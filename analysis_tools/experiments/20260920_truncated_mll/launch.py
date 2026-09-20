"""Audited one-run-per-GPU Supervisor launcher; preserves all DIPO workers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=REPO.parent
PREFIX='optiq-truncated-mll64-20260920'
ALLOCATIONS={
    'vast5':[(0,'humanoid',0),(1,'humanoid',1),(2,'humanoid',2),(3,'ant',0)],
    'vast1':[(0,'ant',1),(2,'ant',2),(3,'halfcheetah',0)],
    'vast3':[(0,'halfcheetah',1),(1,'halfcheetah',2),(2,'walker2d',0),(3,'walker2d',1)],
    'vast4':[(0,'walker2d',2),(1,'hopper',0),(2,'hopper',1),(3,'hopper',2)],
}
CHILD=None
ACTIVE=None


def save(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    temp.replace(path)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def hashes():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    return {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths if p}


def verify():
    plan=json.loads((ROOT/'plan.json').read_text())
    assert git('rev-parse','HEAD')==plan['commit'] and not git('status','--porcelain')
    assert hashes()==json.loads((ROOT/'source-hashes.json').read_text()), 'Frozen source changed'
    return plan


def setup(host):
    from omegaconf import OmegaConf
    from train import compose_config
    assert ROOT.name==PREFIX and not (ROOT/'plan.json').exists()
    assert not git('status','--porcelain'), 'Commit experiment before launching'
    jobs=[]
    for gpu,task,seed in ALLOCATIONS[host]:
        ident=f'{task}-truncatedMLL-N64-M64-T025-s{seed}'
        overrides=[f'benchmark={task}',f'seed={seed}',f'run_name={ident}',
                   f'output_root={ROOT}/outputs','require_gpu=true']
        cfg=compose_config(overrides)
        assert cfg.total_steps==1000000
        jobs.append(dict(id=ident,task=task,seed=seed,gpu=gpu,status='queued',overrides=overrides,
                         config=OmegaConf.to_container(cfg,resolve=True)))
    for folder in ('state','logs','outputs'):(ROOT/folder).mkdir(exist_ok=True)
    save(ROOT/'source-hashes.json',hashes())
    save(ROOT/'plan.json',dict(host=host,commit=git('rev-parse','HEAD'),jobs=jobs))
    for job in jobs:save(ROOT/'state'/(job['id']+'.json'),job)
    names=[]
    for job in jobs:
        gpu=job['gpu'];name=f'{PREFIX}-gpu{gpu}';names.append(name)
        cfgpath=Path('/etc/supervisor/conf.d')/(name+'.conf')
        assert not cfgpath.exists()
        cfgpath.write_text(f'''[program:{name}]
command={sys.executable} -u {HERE}/launch.py worker {gpu}
directory={HERE}
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={ROOT}/logs/worker-gpu{gpu}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
redirect_stderr=true
environment=PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",JAX_PLATFORMS="cpu"
''')
    subprocess.run(['supervisorctl','reread'],check=True)
    for name in names:subprocess.run(['supervisorctl','update',name],check=True)
    print(json.dumps(dict(host=host,commit=git('rev-parse','HEAD'),services=names)),flush=True)


def environment(gpu):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('WANDB_','_WANDB_','JAX_','XLA_')) or key in {'PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME','CUDA_PATH','OPTIQ_CONFIG','OPTIQ_ENV_FILE'}:
            env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
               PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               WANDB_ENTITY='OptiQ',WANDB_PROJECT='heejoon-truncated-mll',WANDB_MODE='online',
               WANDB_INIT_TIMEOUT='300',WANDB_HTTP_TIMEOUT='300',WANDB__SERVICE_WAIT='300')
    choices=[Path('/workspace/optiq-ant-highbeta-20260916/private/wandb_api_key'),
             Path('/workspace/optiq-clean-seed3to7-20260909/wandb_api_key')]
    secret=next(p for p in choices if p.exists())
    env['WANDB_API_KEY']=secret.read_text().strip()
    return env


def worker(gpu):
    global CHILD,ACTIVE
    plan=verify()
    job=next(j for j in plan['jobs'] if j['gpu']==gpu)
    path=ROOT/'state'/(job['id']+'.json')
    assert json.loads(path.read_text())['status']=='queued', 'Do not repeat an unresolved attempt'
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    while subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if (ROOT/'CANCELLED').exists():return
        time.sleep(10)
    if (ROOT/'CANCELLED').exists():return
    verify()
    ACTIVE=job
    try:
        with (ROOT/'logs'/(job['id']+'.log')).open('x') as log:
            CHILD=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),*job['overrides']],
                cwd=HERE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=CHILD.pid,started=time.time());save(path,job)
            rc=CHILD.wait();CHILD=None
        configs=list((ROOT/'outputs').glob(job['id']+'_*/config.json'))
        assert len(configs)==1, 'Missing or duplicated scientific run'
        out=configs[0].parent;cfg=json.loads(configs[0].read_text())
        assert cfg['alg']==job['config']['alg'] and cfg['seed']==job['seed']
        files={p.name for p in out.rglob('*_1000000.msgpack')}
        complete={'actor_state_1000000.msgpack','critic_state_1000000.msgpack'}<=files
        if complete:
            import numpy as np
            for mode in ('zero_z','stochastic_z'):
                evaluations=list(out.rglob(f'evaluations_{mode}.npz'))
                assert len(evaluations)==1
                with np.load(evaluations[0]) as d:
                    complete=complete and bool(d['timesteps'][-1]==1000000 and np.isfinite(d['results'][-1]).all())
        job.update(status=('completed' if rc==0 else 'trained_logging_failed') if complete else 'failed',
                   exit_code=rc,training_complete=complete,finished=time.time(),run_dir=str(out))
    except Exception:
        if CHILD is not None:CHILD.terminate();CHILD.wait();CHILD=None
        job.update(status='failed',finished=time.time(),error=traceback.format_exc())
    save(path,job)
    print(json.dumps({k:job.get(k) for k in ('id','status','exit_code','training_complete')}),flush=True)
    ACTIVE=None


def stop(sig,frame):
    if CHILD is not None:CHILD.terminate()
    if ACTIVE is not None:
        ACTIVE.update(status='cancelled',finished=time.time())
        save(ROOT/'state'/(ACTIVE['id']+'.json'),ACTIVE)
    raise SystemExit(128+sig)


if __name__=='__main__':
    os.umask(0o077)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if sys.argv[1]=='setup':setup(sys.argv[2])
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:raise ValueError('Expected setup HOST or worker GPU')
