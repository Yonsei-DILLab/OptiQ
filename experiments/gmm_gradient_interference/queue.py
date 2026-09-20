"""GPU3 only, independent runs, round-robin 5K segments, checkpointed."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    root.mkdir(parents=True,exist_ok=True);runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    lock=(runtime/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[2];manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    assert all(hashlib.sha256((repo/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(repo),JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'))
    env.pop('JAX_DEFAULT_MATMUL_PRECISION',None);env.pop('LD_LIBRARY_PATH',None)
    prefix=['taskset','-c','6-9',plan['python']]
    val=root/'validation';val.mkdir(exist_ok=True)
    if not (val/'VALIDATION_PASSED.json').exists():
        with (val/'console.log').open('a') as log:
            rc=subprocess.run(prefix+['-m','experiments.gmm_gradient_interference.validate','--out',str(val)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:write(runtime/'VALIDATION_FAILED.json',dict(exit_code=rc,time=time.time()));return
    v=json.loads((val/'VALIDATION_PASSED.json').read_text());assert v['passed'] and v['commit']==manifest['commit']
    failed=[]
    for end in range(plan['segment'],plan['updates']+1,plan['segment']):
        for seed in plan['seeds']:
            for n in plan['n_values']:
                name=f'N{n}_M{plan["m"]}_s{seed}';out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
                if (out/'FAILED.json').exists() or (out/'COMPLETE.json').exists():continue
                if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['step']>=end:continue
                if (runtime/'STOP_NEW_RUNS').exists():return
                cmd=prefix+['-m','experiments.gmm_gradient_interference.run','--root',str(root),'--n',str(n),'--seed',str(seed),'--until',str(end)]
                with (out/'console.log').open('a') as log:
                    proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                    event=dict(run=name,pid=proc.pid,gpu=3,until=end,commit=manifest['commit'],started=time.time())
                    write(runtime/'STATUS.json',event)
                    with (runtime/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    rc=proc.wait()
                if rc:
                    failed.append(name);write(out/'QUEUE_FAILED.json',dict(exit_code=rc,time=time.time()))
                    if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(exit_code=rc,time=time.time()))
                with (runtime/'report.log').open('a') as log:
                    subprocess.run(prefix+['-m','experiments.gmm_gradient_interference.report','--root',str(root)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(runtime/'QUEUE_FINISHED.json',dict(finished=time.time(),failed=failed,commit=manifest['commit']))

if __name__=='__main__':main()
