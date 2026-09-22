"""One process per GPU. Claims are atomic; failed runs are retained, never hidden."""
import argparse,fcntl,json,os,subprocess,sys,time,shutil
from pathlib import Path
from .evaluate import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--phase',choices=['preflight','train'],required=True);p.add_argument('--wandb',action='store_true');a=p.parse_args()
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0));start=(int(a.gpu)*8)%len(cpus)
        os.sched_setaffinity(0,cpus[start:start+8] or cpus[:8])
    prov=json.loads((a.output/'provenance.json').read_text());commit=prov['source_commit']
    queue=a.output/('preflight_queue.json' if a.phase=='preflight' else 'queue.json')
    lock=a.output/'queue.lock';(a.output/'logs').mkdir(exist_ok=True)
    while True:
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            jobs=json.loads(queue.read_text());chosen=None
            passed={j['condition'] for j in json.loads((a.output/'preflight_queue.json').read_text()) if j['status']=='completed'}
            for j in jobs:
                if j['status']!='pending':continue
                if a.phase=='train' and j['condition'] not in passed:
                    j['status']='blocked_preflight';continue
                chosen=j;break
            if chosen is None:atomic_json(queue,jobs);return
            if shutil.disk_usage(a.output).free<2*1024**3:raise RuntimeError('Less than 2 GiB free; queue retained')
            chosen.update(status='running',gpu=a.gpu,worker_pid=os.getpid(),started=time.time())
            atomic_json(queue,jobs)
        dest=a.output/('preflight' if a.phase=='preflight' else 'runs')
        method=next(c['method'] for c in prov['campaign']['conditions'] if c['name']==chosen['condition'])
        python=os.environ.get('GMM40_TORCH_PYTHON',sys.executable) if method in ('sac','dipo','meow') else sys.executable
        cmd=[python,'-m','experiments.gmm40_bandit.run','--condition',chosen['condition'],'--seed',str(chosen['seed']),'--output',str(dest),'--commit',commit]
        # Target metadata in both output namespaces is prepared once, no race.
        (dest/'target').mkdir(parents=True,exist_ok=True)
        if not (dest/'target/definition.json').exists():shutil.copy2(a.output/'target/definition.json',dest/'target/definition.json')
        if a.phase=='preflight':cmd+=['--smoke','--steps','20']
        elif a.wandb:cmd+=['--wandb']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        log=a.output/'logs'/f"{a.phase}_{chosen['condition']}_s{chosen['seed']}.log"
        with log.open('a') as out:code=subprocess.call(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
        status=dest/f"{chosen['condition']}_s{chosen['seed']}"/'status.json'
        state=json.loads(status.read_text()) if status.exists() else {}
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(queue.read_text())
            row=next(j for j in jobs if j['condition']==chosen['condition'] and j['seed']==chosen['seed'])
            row.update(status='completed' if code==0 and state.get('status')=='completed' else 'failed',returncode=code,finished=time.time(),log=str(log))
            atomic_json(queue,jobs)
if __name__=='__main__':main()
