"""Resume immutable numerical revisions; fail closed on GPU infrastructure errors."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,sys,time,socket,shutil

STOP=False
CHILD=None
def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGUSR1)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def infra_failure(folder):
    log=folder/'console.log';text=log.read_text(errors='replace') if log.exists() else ''
    return any(s in text for s in ['No visible GPU devices','CUDA_ERROR_UNKNOWN','Unable to initialize backend','Failed to initialize backend'])

def reset_failures(root,event):
    # Run once before launching a worker wave. Numerical failures stay excluded.
    changed=[]
    for out in (root/'runs').iterdir():
        f=out/'FAILED.json'
        if not f.exists() or (out/'COMPLETE.json').exists():continue
        record=json.loads(f.read_text())
        if (out/'NONFINITE.npz').exists():continue
        if not infra_failure(out) and record.get('returncode') not in [-10,-15]:continue
        dst=out/'attempts'/event;dst.mkdir(parents=True,exist_ok=False)
        shutil.copy2(f,dst/'FAILED.json')
        for n in ['RUNNING.json','progress.json']:
            if (out/n).exists():shutil.copy2(out/n,dst/n)
        write(dst/'RETRY.json',{'reason':'infrastructure failure or interrupted launch','event':event,'time':time.time()})
        f.unlink();changed.append(out.name)
    write(root/'resume_events'/event/'reset.json',{'trials':changed,'count':len(changed),'time':time.time()})
    print('RESET_INFRASTRUCTURE_FAILURES',len(changed),flush=True)

def preflight(root):
    # A fresh process checks the same JAX environment used by the next trial.
    code="import jax, jax.numpy as j; assert jax.default_backend()=='gpu'; x=j.ones((64,64)); assert float((x@x).sum().block_until_ready())==262144.; print(jax.devices(),flush=True)"
    r=subprocess.run([sys.executable,'-c',code],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    return r.returncode==0,r.stdout[-6000:]

def scheduler_snapshot(job=None):
    """Never treat an unavailable scheduler as an empty set of active leases."""
    attempts=0
    while not STOP:
        try:
            r=subprocess.run(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],capture_output=True,text=True,check=True,timeout=30)
            active=set(r.stdout.split())
            if job is not None and job!='local' and job not in active:
                raise RuntimeError('The current allocated worker is absent from scheduler response')
            return active
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired,RuntimeError) as e:
            attempts+=1;delay=min(60,10*2**min(attempts-1,3))
            print('SCHEDULER_QUERY_RETRY',attempts,'delay',delay,'error',str(e),'stderr',getattr(e,'stderr',None),flush=True)
            time.sleep(delay)
    return None

