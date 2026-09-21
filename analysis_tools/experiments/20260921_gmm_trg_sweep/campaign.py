"""Durable per-host queues; GPU leases, immutable source, no automatic reruns.

Stage activation is global and performed by the campaign coordinator after
collecting all host snapshots. Services never advance themselves to later stages.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO.parent
PREFIX = 'optiq-gmm-trg-20260921'
OLD = Path('/workspace/optiq-truncated-mll64-r2-20260920')
SLOTS = {'vast1':[0,2,3], 'vast2':[1], 'vast3':[0,1,2,3],
         'vast4':[0,1,2,3], 'vast5':[0,1,2,3]}
TEMPS = {'humanoid':[.25], 'ant':[.25], 'halfcheetah':[.25,.1],
         'walker2d':[.25,.1], 'hopper':[.05,.1]}
CHILD = None
ACTIVE = None

def save(path, obj):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def git(*args): return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def hashes():
    return {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()
            for p in git('ls-files').splitlines()}
def verify():
    plan=json.loads((ROOT/'plan.json').read_text())
    assert git('rev-parse','HEAD')==plan['commit'] and not git('status','--porcelain')
    assert hashes()==json.loads((ROOT/'source-hashes.json').read_text())
    return plan

@contextlib.contextmanager
def locked():
    with (ROOT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield

def job(task,temp,seed,beta=1.,dacer=False,stage=1):
    def token(v): return str(float(v)).replace('.','p')
    ident=f'{task}-T{token(temp)}-b{token(beta)}-dacer{int(dacer)}-s{seed}'
    overrides=[f'benchmark={task}',f'seed={seed}',f'alg.actor.temperature={temp}',
               f'alg.actor.density_correction_beta={beta}',f'dacer.enabled={str(dacer).lower()}',
               f'dacer.noise_scale={.15 if task in ("humanoid","halfcheetah") else .1}',
               f'run_name={ident}',f'wandb.group={task}_T{token(temp)}_b{token(beta)}_dacer{int(dacer)}',
               f'output_root={ROOT}/outputs','require_gpu=true']
    return dict(id=ident,task=task,temperature=temp,seed=seed,beta=beta,dacer=dacer,
                stage=stage,status='queued',overrides=overrides,
                wandb_id=hashlib.sha256(('gmm-trg-20260921/'+ident).encode()).hexdigest()[:12])

def add_jobs(jobs):
    from train import compose_config
    from omegaconf import OmegaConf
    with locked():
        added=[]
        for j in jobs:
            order=j.get('order',0)
            # Rebuild overrides locally to avoid carrying another host's paths.
            j=job(j['task'],j['temperature'],j['seed'],j['beta'],j['dacer'],j['stage'])
            j['order']=order
            p=ROOT/'state'/(j['id']+'.json')
            if p.exists(): continue
            cfg=compose_config(j['overrides'])
            assert cfg.alg.utd==1 and cfg.alg.batch_size==256
            assert cfg.alg.optimizer.lr_actor==cfg.alg.optimizer.lr_critic==.0003
            assert cfg.total_steps==1000000 and cfg.alg.learning_starts==5000
            j['config']=OmegaConf.to_container(cfg,resolve=True)
            save(p,j);added.append(j['id'])
    return added

def setup(host,jobs):
    assert ROOT.name==PREFIX and not (ROOT/'plan.json').exists()
    assert not git('status','--porcelain')
    for d in ('state','logs','outputs','imports'):(ROOT/d).mkdir(exist_ok=True)
    save(ROOT/'source-hashes.json',hashes())
    save(ROOT/'plan.json',dict(host=host,commit=git('rev-parse','HEAD'),slots=SLOTS[host]))
    save(ROOT/'gate.json',dict(stage=1))
    added=add_jobs(jobs)
    names=[]
    for gpu in SLOTS[host]:
        name=f'{PREFIX}-gpu{gpu}'; names.append(name)
        path=Path('/etc/supervisor/conf.d')/(name+'.conf'); assert not path.exists()
        path.write_text(f'''[program:{name}]
command={sys.executable} -u {HERE}/campaign.py worker {gpu}
directory={HERE}
autostart=true
autorestart=unexpected
startsecs=3
startretries=2
stopasgroup=true
killasgroup=true
stopwaitsecs=60
stdout_logfile={ROOT}/logs/worker-gpu{gpu}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
redirect_stderr=true
environment=PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",JAX_PLATFORMS="cpu",PYTHONPATH="{ROOT}/deps"
''')
    subprocess.run(['supervisorctl','reread'],check=True)
    for name in names:subprocess.run(['supervisorctl','update',name],check=True)
    return dict(host=host,added=added,services=names)

def environment(gpu=None):
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('WANDB_','_WANDB_','JAX_','XLA_')) or k in {'PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME','CUDA_PATH','OPTIQ_CONFIG','OPTIQ_ENV_FILE'}:
            env.pop(k,None)
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONNOUSERSITE='1',
               PYTHONPATH=str(ROOT/'deps'),
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               WANDB_ENTITY='OptiQ',WANDB_PROJECT='gmm-trg',WANDB_MODE='online',
               WANDB_INIT_TIMEOUT='300',WANDB_HTTP_TIMEOUT='300',WANDB__SERVICE_WAIT='300')
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    else: env['JAX_PLATFORMS']='cpu'
    paths=[Path('/workspace/optiq-ant-highbeta-20260916/private/wandb_api_key'),
           Path('/workspace/optiq-clean-seed3to7-20260909/wandb_api_key')]
    secret=next(p for p in paths if p.exists())
    env['WANDB_API_KEY']=secret.read_text().strip()
    return env

def outcome(out):
    import numpy as np
    files={p.name for p in out.rglob('*_1000000.msgpack')}
    if not {'actor_state_1000000.msgpack','critic_state_1000000.msgpack'}<=files: return None
    scores={}
    for mode in ('zero_z','stochastic_z'):
        paths=list(out.rglob(f'evaluations_{mode}.npz'))
        if len(paths)!=1: return None
        with np.load(paths[0]) as d:
            steps=d['timesteps']; values=d['results']
            if steps[-1]!=1000000: return None
            mask=(steps>900000)&(steps<=1000000)
            if mask.sum()!=20 or not np.isfinite(values[mask]).all():return None
            scores[mode]=float(values[mask].mean())
    return scores

def worker(gpu):
    global CHILD,ACTIVE
    plan=verify(); assert gpu in plan['slots']
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    # Kernel lease shared by this campaign; also check all existing GPU processes.
    with (ROOT/f'gpu{gpu}.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'CANCELLED').exists():
            busy=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy: time.sleep(15);continue
            with locked():
                gate=json.loads((ROOT/'gate.json').read_text())['stage']
                pending=[json.loads(p.read_text()) for p in (ROOT/'state').glob('*.json')]
                pending=[j for j in pending if j['status']=='queued' and j['stage']<=gate]
                if pending:
                    pending.sort(key=lambda j:(j['stage'],j.get('order',0),j['id']))
                    ACTIVE=pending[0];ACTIVE.update(status='running',gpu=gpu,started=time.time())
                    save(ROOT/'state'/(ACTIVE['id']+'.json'),ACTIVE)
            if ACTIVE is None: time.sleep(15);continue
            j=ACTIVE;path=ROOT/'state'/(j['id']+'.json')
            try:
                verify()
                env=environment(gpu);env.update(WANDB_RUN_ID=j['wandb_id'],WANDB_RESUME='never')
                with (ROOT/'logs'/(j['id']+'.log')).open('x') as log:
                    CHILD=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),*j['overrides']],
                        cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
                    j['pid']=CHILD.pid;save(path,j)
                    rc=CHILD.wait();CHILD=None
                cfgpaths=list((ROOT/'outputs').glob(j['id']+'_*/config.json'))
                assert len(cfgpaths)==1,'Missing/duplicate run output'
                out=cfgpaths[0].parent;cfg=json.loads(cfgpaths[0].read_text())
                assert cfg['alg']==j['config']['alg'] and cfg['runtime']['git_commit']==plan['commit']
                score=outcome(out)
                j.update(status=('completed' if rc==0 else 'trained_logging_failed') if score else 'failed',
                         training_complete=bool(score),scores=score,exit_code=rc,run_dir=str(out),finished=time.time())
            except Exception:
                if CHILD is not None:CHILD.terminate();CHILD.wait();CHILD=None
                j.update(status='failed',error=traceback.format_exc(),finished=time.time())
            save(path,j);print(json.dumps({k:j.get(k) for k in ('id','status','scores','exit_code')}),flush=True)
            ACTIVE=None

def stop(sig,frame):
    if CHILD is not None: CHILD.terminate()
    if ACTIVE is not None:
        ACTIVE.update(status='interrupted',finished=time.time())
        save(ROOT/'state'/(ACTIVE['id']+'.json'),ACTIVE)
    raise SystemExit(128+sig)

def import_old():
    if not OLD.exists():return []
    reports=[]
    for path in sorted((OLD/'state').glob('*.json')):
        source=json.loads(path.read_text())
        if source['task']=='hopper':continue
        j=job(source['task'],.25,source['seed'])
        j.update(status='awaiting_original',imported=True,source_host=verify()['host'],source_id=source['id'])
        cfgpaths=list((OLD/'outputs').glob(source['id']+'_*/config.json'))
        if len(cfgpaths)==1:
            out=cfgpaths[0].parent;cfg=json.loads(cfgpaths[0].read_text())
            assert cfg['alg']['actor']['log_std_max']==-1 and cfg['alg']['actor']['initial_log_std']==-1
            assert cfg['runtime']['git_commit']=='fc6067b1f158384736c4e1e12ccc95aaac2e0f72'
            j.update(scores=outcome(out),run_dir=str(out),config=cfg)
            if j['scores']:
                j.update(status='completed',training_complete=True)
                blobs=list(out.glob('wandb/run-*/run-*.wandb'))
                assert len(blobs)==1
                source_id=blobs[0].stem.removeprefix('run-')
                j.update(wandb_id=source_id,source_wandb=f'OptiQ/heejoon-truncated-mll/{source_id}')
                marker=ROOT/'imports'/(source_id+'.json')
                if not marker.exists():
                    with (ROOT/'logs'/('import-'+source_id+'.log')).open('a') as log:
                        rc=subprocess.run([str(Path(sys.executable).with_name('wandb')),'sync','--legacy',
                            '--project','gmm-trg','--entity','OptiQ','--id',source_id,
                            '--include-online','--include-synced','--no-mark-synced','--no-sync-tensorboard',
                            str(blobs[0])],env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
                    if rc==0:save(marker,dict(source=j['source_wandb'],target=f'OptiQ/gmm-trg/{source_id}',time=time.time()))
                j['wandb_imported']=marker.exists()
        with locked():
            dest=ROOT/'state'/(j['id']+'.json')
            if dest.exists(): assert json.loads(dest.read_text()).get('imported'), 'Duplicate scientific run'
            save(dest,j)
        reports.append({k:j.get(k) for k in ('id','status','scores','wandb_imported')})
    return reports

def snapshot():
    result=[]
    for path in sorted((ROOT/'state').glob('*.json')):
        j=json.loads(path.read_text())
        item={k:j.get(k) for k in ('id','task','temperature','seed','beta','dacer','stage','status',
                                 'gpu','pid','scores','wandb_id','imported','wandb_imported','error')}
        log=ROOT/'logs'/(j['id']+'.log')
        if log.exists():
            text=log.read_text(errors='replace')
            steps=re.findall(r'total_timesteps\s*\|\s*(\d+)',text)
            item['steps']=int(steps[-1]) if steps else None
            item['traceback']='Traceback (most recent call last)' in text
        result.append(item)
    return dict(host=verify()['host'],gate=json.loads((ROOT/'gate.json').read_text()),jobs=result)

if __name__=='__main__':
    os.umask(0o077)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    mode=sys.argv[1]
    if mode=='setup': result=setup(sys.argv[2],json.load(sys.stdin))
    elif mode=='worker': worker(int(sys.argv[2]));sys.exit()
    elif mode=='add': verify();result=add_jobs(json.load(sys.stdin))
    elif mode=='gate':
        verify();stage=int(sys.argv[2]);assert 1<=stage<=4
        save(ROOT/'gate.json',dict(stage=stage));result={'stage':stage}
    elif mode=='import-old': result=import_old()
    elif mode=='snapshot': result=snapshot()
    else: raise ValueError(mode)
    print(json.dumps(result,allow_nan=False),flush=True)
