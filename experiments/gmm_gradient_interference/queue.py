"""Independent one-run-per-GPU workers; GPU3 remains reserved for the parent study."""
import argparse,fcntl,hashlib,json,os,subprocess,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    root.mkdir(parents=True,exist_ok=True);runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    lock=(runtime/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[2];manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    assert all(hashlib.sha256((repo/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
    plan=json.loads((Path(__file__).parent/'plan.json').read_text());event_lock=threading.Lock()
    def setup(gpu):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(repo),JAX_COMPILATION_CACHE_DIR=str(root/f'jax_cache_gpu{gpu}'))
        env.pop('JAX_DEFAULT_MATMUL_PRECISION',None);env.pop('LD_LIBRARY_PATH',None)
        return ['taskset','-c',','.join(map(str,plan['cpu_sets'][str(gpu)])),plan['python']],env
    prefix,env=setup(plan['gpus'][0]);val=root/'validation';val.mkdir(exist_ok=True)
    if not (val/'VALIDATION_PASSED.json').exists():
        with (val/'console.log').open('a') as log:
            rc=subprocess.run(prefix+['-m','experiments.gmm_gradient_interference.validate','--out',str(val)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:write(runtime/'VALIDATION_FAILED.json',dict(exit_code=rc,time=time.time()));return
    v=json.loads((val/'VALIDATION_PASSED.json').read_text());assert v['passed'] and v['commit']==manifest['commit']
    tasks=[(n,m,seed) for seed in plan['seeds'] for n,m in plan['sizes']]
    write(runtime/'REGISTERED.json',dict(commit=manifest['commit'],pid=os.getpid(),tasks=[dict(n=n,m=m,seed=s,gpu=plan['gpus'][i%len(plan['gpus'])]) for i,(n,m,s) in enumerate(tasks)]))
    def worker(gpu,assigned):
        prefix,env=setup(gpu);failed=[]
        for end in range(plan['segment'],plan['updates']+1,plan['segment']):
            for n,m,seed in assigned:
                if (runtime/'STOP_NEW_RUNS').exists():return dict(gpu=gpu,stopped=True,failed=failed)
                name=f'N{n}_M{m}_s{seed}';out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
                if (out/'FAILED.json').exists() or (out/'COMPLETE.json').exists():continue
                if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['step']>=end:continue
                cmd=prefix+['-m','experiments.gmm_gradient_interference.run','--root',str(root),'--n',str(n),'--m',str(m),'--seed',str(seed),'--until',str(end)]
                with (out/'console.log').open('a') as log:
                    proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                    event=dict(run=name,pid=proc.pid,gpu=gpu,until=end,commit=manifest['commit'],started=time.time())
                    write(runtime/f'STATUS_gpu{gpu}.json',event)
                    with event_lock:
                        with (runtime/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    rc=proc.wait()
                    event.update(exit_code=rc,finished=time.time());write(runtime/f'STATUS_gpu{gpu}.json',event)
                if rc:
                    failed.append(name)
                    if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(exit_code=rc,time=time.time()))
        return dict(gpu=gpu,stopped=False,failed=failed)
    with ThreadPoolExecutor(max_workers=len(plan['gpus'])) as pool:
        futures=[pool.submit(worker,gpu,tasks[i::len(plan['gpus'])]) for i,gpu in enumerate(plan['gpus'])]
        results=[f.result() for f in futures]
    with (runtime/'report.log').open('a') as log:
        subprocess.run(prefix+['-m','experiments.gmm_gradient_interference.report','--root',str(root)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(runtime/'QUEUE_FINISHED.json',dict(finished=time.time(),workers=results,commit=manifest['commit']))

if __name__=='__main__':main()
