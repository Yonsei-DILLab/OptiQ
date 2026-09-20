"""Slurm-only execution; no credentials in source/config or command line."""
import hashlib,json,netrc,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=REPO.parent
CHILD=None
ACTIVE=None


def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def hashes():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    return {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths if p}


def check_source():
    plan=json.loads((ROOT/'plan.json').read_text())
    assert git('rev-parse','HEAD')==plan['commit'] and not git('status','--porcelain')
    assert hashes()==json.loads((ROOT/'source-hashes.json').read_text())
    return plan


def prepare():
    from omegaconf import OmegaConf
    from train_beta import compose_config
    assert ROOT.name=='optiq-mll-beta-20260920' and not (ROOT/'plan.json').exists()
    assert not git('status','--porcelain')
    for name in ('logs','state','outputs','cache','wandb-cache','validation'):(ROOT/name).mkdir(exist_ok=True)
    jobs=[]
    for env in ('humanoid','ant','halfcheetah','walker2d','hopper'):
        for seed in range(3):
            ident=f'{env}-directMLL-N64-M64-T025-beta0to1-100K-s{seed}'
            args=[f'benchmark={env}',f'seed={seed}',f'output_root={ROOT}/outputs',f'run_name={ident}']
            cfg=compose_config(args)
            jobs.append(dict(id=ident,task=env,seed=seed,overrides=args,config=OmegaConf.to_container(cfg,resolve=True)))
    save(ROOT/'plan.json',dict(commit=git('rev-parse','HEAD'),jobs=jobs))
    save(ROOT/'source-hashes.json',hashes())
    print(json.dumps(dict(prepared=len(jobs),commit=git('rev-parse','HEAD'))))


def smoke():
    check_source()
    assert os.environ.get('SLURM_JOB_ID')
    import jax,numpy as np
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    subprocess.run([sys.executable,'-m','pytest',str(HERE/'test_beta.py'),'-q'],check=True)
    from train_beta import compose_config,base,BetaAnnealedMLL
    from stable_baselines3.common.logger import configure
    base.runner.OptiQDIME=BetaAnnealedMLL
    cfg=compose_config(['benchmark=humanoid',f'output_root={ROOT}/validation',
        'total_steps=5020','checkpoint_interval=0','eval_interval=1000000'])
    model,callbacks=base.runner.create_algorithm(cfg)
    model.set_logger(configure(str(ROOT/'validation/logs'),['csv']))
    captured={};record=model.logger.record
    def capture(k,v,*args,**kwargs):
        if k.startswith('train/'):captured[k]=float(v)
        return record(k,v,*args,**kwargs)
    model.logger.record=capture
    try:
        model.learn(total_timesteps=5020)
        assert model._n_updates==20
        assert np.isclose(captured['train/density_beta'],.0502)
        assert all(np.isfinite(v) for v in captured.values())
        proof=dict(passed=True,validation_only=True,steps=5020,updates=20,
            beta=captured['train/density_beta'],node=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],
            device=str(jax.devices()),commit=git('rev-parse','HEAD'),time=time.time())
        save(ROOT/'smoke-result.json',proof);print(json.dumps(proof),flush=True)
    finally:
        for cb in callbacks.callbacks:cb.eval_env.close()
        model.get_env().close();model.logger.close()


def train(index):
    global CHILD,ACTIVE
    plan=check_source();assert json.loads((ROOT/'smoke-result.json').read_text())['passed']
    assert os.environ.get('SLURM_JOB_ID')
    job=plan['jobs'][index];path=ROOT/'state'/(job['id']+'.json')
    assert not path.exists() and not list((ROOT/'outputs').glob(job['id']+'_*/config.json'))
    job.update(status='running',started=time.time(),slurm_job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename)
    ACTIVE=job;save(path,job)
    env=os.environ.copy();auth=netrc.netrc().authenticators('api.wandb.ai');assert auth
    env.update(WANDB_API_KEY=auth[2],WANDB_ENTITY='OptiQ',WANDB_MODE='online',
               WANDB_PROJECT='heejoon-direct-mll-beta-anneal',WANDB_INIT_TIMEOUT='300',WANDB__SERVICE_WAIT='300')
    with (ROOT/'logs'/(job['id']+'.log')).open('x') as log:
        CHILD=subprocess.Popen([sys.executable,'-u',str(HERE/'train_beta.py'),*job['overrides']],
            cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
        job['pid']=CHILD.pid;save(path,job);rc=CHILD.wait();CHILD=None
    done=list((ROOT/'outputs').glob(job['id']+'_*/completed.json'))
    complete=len(done)==1 and json.loads(done[0].read_text())['timesteps']==1000000
    job.update(status='completed' if rc==0 and complete else 'failed',exit_code=rc,finished=time.time(),training_complete=complete)
    save(path,job);ACTIVE=None
    raise SystemExit(0 if complete and rc==0 else rc or 1)


def stop(sig,frame):
    if CHILD is not None:CHILD.terminate()
    if ACTIVE is not None:
        ACTIVE.update(status='cancelled',finished=time.time());save(ROOT/'state'/(ACTIVE['id']+'.json'),ACTIVE)
    raise SystemExit(128+sig)


if __name__=='__main__':
    os.umask(0o077);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='smoke':smoke()
    else:train(int(sys.argv[1]))
