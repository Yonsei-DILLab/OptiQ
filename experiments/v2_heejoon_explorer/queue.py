"""GPU-local resource queue after v1; no cross-environment completion dependency."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path

MAP={'ant':0,'humanoid':1,'halfcheetah':2}


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def predecessor_busy(predecessor,env):
    path=predecessor/'runtime'/f'{env}.lock'
    if not path.exists():return False
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        finally:
            # Closing the descriptor releases our probe lock, if acquired.
            pass
    return False


def gpu_busy(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return any(line.split(',')[0].strip()==uuid for line in text.splitlines())


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=MAP,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();repo=Path(__file__).resolve().parents[2];runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/f'{a.env}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text());plan=json.loads((repo/'experiments/v2_heejoon_explorer/plan.json').read_text())
    validated=json.loads((root/'validation/VALIDATION_PASSED.json').read_text());assert validated['passed'] and validated['commit']==manifest['commit']
    predecessor=Path(plan['predecessor_root']);gpu=MAP[a.env]
    state=dict(env=a.env,gpu=gpu,commit=manifest['commit'],queue_pid=os.getpid(),registered=time.time(),seeds=plan['seeds'])
    while True:
        if (runtime/'STOP_NEW_RUNS').exists():return
        old_busy=predecessor_busy(predecessor,a.env);busy=gpu_busy(gpu)
        if not old_busy and not busy:break
        write(runtime/f'{a.env}.json',dict(state,state='waiting_for_gpu',predecessor_worker_active=old_busy,gpu_has_compute=busy,checked=time.time()))
        time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),WANDB_MODE='online',WANDB_ENTITY='OptiQ',WANDB_PROJECT='legacy_explorer',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'),PYTHONPATH=str(repo))
    # Same default numerical precision as v1, not the separate GMM40 override.
    env.pop('JAX_DEFAULT_MATMUL_PRECISION',None);env.pop('LD_LIBRARY_PATH',None)
    env['WANDB_API_KEY']=(Path.home()/'.config/optiq-secrets/wandb_api_key').read_text().strip()
    cpus=f'{2*gpu},{2*gpu+1}'
    # Actual GPU smoke deferred until this environment's assigned GPU is free.
    out=root/'validation_gpu'/a.env;out.mkdir(parents=True,exist_ok=True)
    gate=out/'VALIDATION_PASSED.json'
    if not gate.exists():
        cmd=['taskset','-c',cpus,sys.executable,'-m','experiments.v2_heejoon_explorer.validate','--device','gpu','--env',a.env,'--output',str(out)]
        with (out/'console.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(runtime/f'{a.env}.json',dict(state,state='gpu_validation',pid=proc.pid,started=time.time()));rc=proc.wait()
        if rc or not gate.exists():
            write(runtime/f'{a.env}.json',dict(state,state='gpu_validation_failed',exit_code=rc,time=time.time()));return
    for seed in plan['seeds']:
        if (runtime/'STOP_NEW_RUNS').exists():return
        out=root/'runs'/f'{a.env}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():continue
        cmd=['taskset','-c',cpus,sys.executable,'-m','experiments.v2_heejoon_explorer.run','--env',a.env,'--seed',str(seed),'--out',str(out)]
        with (out/'console.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(state,state='running',seed=seed,pid=proc.pid,started=time.time());write(runtime/f'{a.env}.json',record);rc=proc.wait()
        complete=(out/'COMPLETE.json').exists();record.update(exit_code=rc,finished=time.time(),state='complete' if complete else 'paused' if rc==0 else 'failed')
        write(runtime/f'{a.env}.json',record)
        if not complete:return
    (runtime/f'{a.env}_QUEUE_DONE').write_text('All v2 seeds complete.\n')


if __name__=='__main__':main()
