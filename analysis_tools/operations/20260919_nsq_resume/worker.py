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

def main(args):
    global CHILD
    root=Path(args.root).resolve();event=args.event
    if args.reset_infrastructure_failures:reset_failures(root,event);return
    code=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id']
    gate='LEGACY_TD_VALIDATION' if args.legacy_only else 'VALIDATION'
    for d in [1,2,4,8]:
        v=json.loads((root/f'{gate}_D{d}.json').read_text());assert v['passed'] and v['source_code_id']==code
    numerical=json.loads((root/'DEPLOYMENT.json').read_text())
    operations=json.loads((Path(__file__).parent/'OPS_DEPLOYMENT.json').read_text())
    plan=json.loads((root/'tasks.json').read_text())
    selected=[(i,t) for i,t in enumerate(plan) if ((t['stage']=='closed' and t['method']=='argmax_truncated')==args.legacy_only)]
    job=os.environ.get('SLURM_JOB_ID','local')
    if 'SLURM_ARRAY_JOB_ID' in os.environ:job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
    eventdir=root/'resume_events'/event;eventdir.mkdir(parents=True,exist_ok=True)
    write(eventdir/(job+'.json'),dict(time=time.time(),job=job,host=socket.gethostname(),numerical=numerical,operations=operations,legacy_only=args.legacy_only))
    (root/'queue').mkdir(exist_ok=True)
    while not STOP:
        try:healthy,description=preflight(root)
        except Exception as e:healthy,description=False,repr(e)
        print('GPU_PREFLIGHT',healthy,description,flush=True)
        if STOP:return 75
        if not healthy:
            write(eventdir/(job+'.GPU_FAILURE.json'),dict(time=time.time(),host=socket.gethostname(),description=description));return 86
        with (root/'queue/claim.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            r=subprocess.run(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],capture_output=True,text=True,check=True)
            active=set(r.stdout.split());chosen=None;leased=0;pending=0
            for i,t in selected:
                out=root/'runs'/t['name'];lease=root/'queue'/(t['name']+'.json')
                if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists():continue
                pending+=1
                if lease.exists():
                    old=json.loads(lease.read_text())
                    if old['job_id'] in active or time.time()-old['time']<60:leased+=1;continue
                    lease.unlink()
                deps=[x for x in [t.get('parent'),t.get('q_source')] if x]
                if all((root/'runs'/d/'COMPLETE.json').exists() for d in deps):
                    chosen=(i,t,out,lease);break
            if chosen:
                i,t,out,lease=chosen;write(lease,dict(job_id=job,time=time.time(),host=socket.gethostname(),task_index=i,event=event))
        if not chosen:
            if not pending:print('ALL_ELIGIBLE_WORK_PROCESSED',flush=True);return 0
            if not leased:print('REMAINING_WORK_BLOCKED_BY_FAILED_DEPENDENCIES',pending,flush=True);return 0
            time.sleep(30);continue
        out.mkdir(parents=True,exist_ok=True)
        progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        write(out/'attempts'/event/(job+'.json'),dict(time=time.time(),job=job,resume_step=progress.get('step',0),numerical_commit=numerical['commit'],operations_commit=operations['commit']))
        print('STARTING',i,t['name'],'resume_step',progress.get('step',0),flush=True)
        with (out/'console.log').open('a') as log:
            CHILD=subprocess.Popen([sys.executable,'-m','nsq.run','--root',str(root),'--index',str(i)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            result=CHILD.wait();CHILD=None
        with (root/'queue/claim.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if lease.exists() and json.loads(lease.read_text())['job_id']==job:lease.unlink()
        if result==75 or STOP:return 75
        if result and infra_failure(out):
            write(eventdir/(job+'.GPU_FAILURE.json'),dict(time=time.time(),task=t['name'],returncode=result,host=socket.gethostname()))
            return 86
        if result and not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(returncode=result,time=time.time(),job_id=job))
    return 75

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--event',required=True);p.add_argument('--legacy-only',action='store_true');p.add_argument('--reset-infrastructure-failures',action='store_true')
    sys.exit(main(p.parse_args()) or 0)
