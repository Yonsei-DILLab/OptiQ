"""Two independent MuJoCo runs/GPU; adopt existing processes without restarting them."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def arg(argv,name):
    return argv[argv.index(name)+1] if name in argv else None


def processes():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
            wrapped=any(Path(x).name=='online_runner.py' for x in argv)
            module=arg(argv,'--module') if wrapped else arg(argv,'-m')
            if module not in ['experiments.v1_heejoon_explorer.run','experiments.v2_heejoon_explorer.run','experiments.v3_heejoon_explorer.run']:continue
            version=module.split('.')[1].split('_')[0]
            # Read only the relevant key; never emit credentials from process environments.
            cuda=next((v.split('=',1)[1] for v in (p/'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')),None)
            found.append(dict(pid=int(p.name),version=version,env=arg(argv,'--env'),seed=int(arg(argv,'--seed')),out=arg(argv,'--out'),gpu=cuda,cpus=sorted(os.sched_getaffinity(int(p.name)))))
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
    return found


def select_next(plan,environment,active,attempted,failed,roots):
    busy={(r['version'],r['seed']) for r in active}
    for version,seed in plan['order']:
        key=f'{environment}_{version}_s{seed}';out=roots[version]/'runs'/f'{environment}_s{seed}'
        if (version,seed) in busy or key in attempted or key in failed:continue
        if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists():continue
        return key,version,seed,out
    return None


def free_cpus(plan,environment,active):
    used=set(c for r in active for c in r['cpus'])
    return next((pair for pair in plan['cpus'][environment] if not used.intersection(pair)),None)


def verify(plan):
    roots={v:Path(plan['base'])/p for v,p in plan['roots'].items()}
    for version,root in roots.items():
        m=json.loads((root/'repo/SOURCE_MANIFEST.json').read_text())
        assert m['commit']==plan['numerical_commits'][version]
        assert all(hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h for p,h in m['files'].items())
        v=json.loads((root/'validation/VALIDATION_PASSED.json').read_text());assert v['passed'] and v['commit']==m['commit']
    return roots


def activate(plan,root,roots):
    # Old v1 workers are waiting on the active run; only disable their future dispatch.
    # Old v2 workers have never launched: stop those idle queue processes only.
    for r in roots.values():
        (r/'runtime/STOP_NEW_RUNS').write_text('Superseded by two-per-GPU queue; current training remains alive.\n')
    stopped=[]
    for environment in plan['gpus']:
        p=roots['v2']/'runtime'/f'{environment}.json';old=json.loads(p.read_text())
        pid=old.get('queue_pid');proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists():
            cmd=proc.read_bytes().decode().split('\0')
            assert old['state']=='waiting_for_gpu',old
            assert 'experiments.v2_heejoon_explorer.queue' in cmd and arg(cmd,'--env')==environment
            os.kill(pid,signal.SIGTERM);stopped.append(pid)
            old.update(state='superseded_by_two_per_gpu',replacement=str(root),superseded=time.time());write(p,old)
    receipt=dict(time=time.time(),adopted=processes(),stopped_idle_v2_workers=stopped,source_commits=plan['numerical_commits'])
    write(root/'ACTIVATION.json',receipt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--activate',action='store_true');a=parser.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    package=Path(__file__).parent;plan=json.loads((package/'plan.json').read_text());roots=verify(plan)
    source=json.loads((package/'SCHEDULER_MANIFEST.json').read_text())
    assert all(hashlib.sha256((package/p).read_bytes()).hexdigest()==h for p,h in source['files'].items())
    if a.activate:activate(plan,root,roots)
    assert (root/'ACTIVATION.json').exists()
    children={}
    # Adopted jobs are already attempted: do not silently restart one if it pauses.
    attempted={f"{r['env']}_{r['version']}_s{r['seed']}" for r in processes()}
    failed=json.loads((root/'FAILED_ATTEMPTS.json').read_text()) if (root/'FAILED_ATTEMPTS.json').exists() else {}
    while True:
        for key,(proc,metadata,log) in list(children.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del children[key]
            complete=(Path(metadata['out'])/'COMPLETE.json').exists()
            record=dict(**metadata,exit_code=rc,finished=time.time(),complete=complete)
            with (root/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if not complete:failed[key]=record;write(root/'FAILED_ATTEMPTS.json',failed)
        active=processes()
        caps=json.loads((root/'CAPACITY.json').read_text()) if (root/'CAPACITY.json').exists() else {}
        states={}
        for environment,gpu in plan['gpus'].items():
            running=[r for r in active if r['env']==environment]
            assert all(r['gpu']==str(gpu) for r in running),running
            capacity=min(2,int(caps.get(environment,2)))
            states[environment]=dict(gpu=gpu,capacity=capacity,running=running)
            if (root/'STOP_NEW_RUNS').exists() or len(running)>=capacity:continue
            candidate=select_next(plan,environment,running,attempted,failed,roots)
            if candidate is None:continue
            key,version,seed,out=candidate;cpus=free_cpus(plan,environment,running)
            if cpus is None:continue
            repo=roots[version]/'repo';env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),WANDB_MODE='online',WANDB_ENTITY='OptiQ',WANDB_PROJECT='legacy_explorer',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',JAX_COMPILATION_CACHE_DIR=str(roots[version]/'jax_cache'),PYTHONPATH=str(repo))
            env.pop('LD_LIBRARY_PATH',None);env.pop('JAX_DEFAULT_MATMUL_PRECISION',None)
            env['WANDB_API_KEY']=(Path.home()/'.config/optiq-secrets/wandb_api_key').read_text().strip()
            # Keep each new algorithm's committed GPU preflight in the assigned slot.
            gate=roots[version]/'validation_gpu'/environment/'VALIDATION_PASSED.json'
            if version in ('v2','v3') and not gate.exists():
                outval=gate.parent;outval.mkdir(parents=True,exist_ok=True)
                cmd=['taskset','-c',','.join(map(str,cpus)),plan['python'],'-m',f'experiments.{version}_heejoon_explorer.validate','--device','gpu','--env',environment,'--output',str(outval)]
                write(root/'CURRENT_GATE.json',dict(env=environment,gpu=gpu,started=time.time(),scheduler_commit=source['commit']))
                with (outval/'console.log').open('a') as log:rc=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if rc or not gate.exists():
                    failed[key]=dict(reason='GPU validation failed',exit_code=rc);write(root/'FAILED_ATTEMPTS.json',failed);continue
                v=json.loads(gate.read_text());assert v['passed'] and v['commit']==plan['numerical_commits'][version]
            # Rescan after GPU validation, guarding against external duplicate dispatch.
            if any(r['env']==environment and r['version']==version and r['seed']==seed for r in processes()):continue
            out.mkdir(parents=True,exist_ok=True)
            cmd=['taskset','-c',','.join(map(str,cpus)),plan['python'],'-m',f'experiments.{version}_heejoon_explorer.run','--env',environment,'--seed',str(seed),'--out',str(out)]
            if version in plan.get('online_healthcheck_versions',[]):
                cmd=['taskset','-c',','.join(map(str,cpus)),plan['python'],str(package/'online_runner.py'),
                     '--module',f'experiments.{version}_heejoon_explorer.run','--source-root',str(repo),
                     '--env',environment,'--seed',str(seed),'--out',str(out)]
            log=(out/'console.log').open('a');proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            metadata=dict(key=key,version=version,env=environment,seed=seed,out=str(out),pid=proc.pid,gpu=gpu,cpus=cpus,started=time.time(),source_commit=plan['numerical_commits'][version],scheduler_commit=source['commit'],max_concurrent_runs_per_gpu=2,resumed=(out/'resume.zip').exists())
            children[key]=(proc,metadata,log);attempted.add(key)
            write(out/'SCHEDULING.json',metadata)
            with (root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(**metadata,event='launched'))+'\n')
        write(root/'STATUS.json',dict(time=time.time(),scheduler_commit=source['commit'],pid=os.getpid(),gpus=states,failed_attempts=failed,priority_order=plan['order']))
        if not active and not children and all(select_next(plan,e,[],attempted,failed,roots) is None for e in plan['gpus']):
            write(root/'QUEUE_DONE.json',dict(time=time.time(),failed_attempts=len(failed)));return
        time.sleep(plan['poll_seconds'])


if __name__=='__main__':main()
