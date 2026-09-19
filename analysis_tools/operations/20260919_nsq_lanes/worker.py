"""Two operational queues over immutable numerical sources and existing checkpoints."""
from pathlib import Path
import argparse, fcntl, json, os, socket, subprocess, sys, time
import support as rt

REVISIONS = {
    'cc11f537af33': 'cc11f537af330e23e1cc77cb94a9426660b55ebb',
    '91cb9c3_legacy_td': '91cb9c3d32f9d3c7f43ff886d34b6199080be811',
}

def revision_for(t):
    return '91cb9c3_legacy_td' if t['stage']=='closed' and t['method']=='argmax_truncated' else 'cc11f537af33'

def lane_for(t):
    return 'exact' if t['dim']>1 and t['n']*t['m']>16*64 and t['method'].startswith('exact_') else 'fast'

def rank(t, out):
    progress = out/'progress.json'
    # Interrupted work first; finished initial fits immediately enter their changes.
    resume = progress.exists()
    stage = {'source':0, 'mass':1, 'split':1, 'replay':2, 'prefix':3, 'closed':4}[t['stage']]
    return (0 if resume else 1, stage, t['priority'], t['dim'], t['seed'], t['name'])

def selectable(root, t, active):
    out=root/'runs'/t['name']
    if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists(): return False
    lease=root/'queue'/(t['name']+'.json')
    if lease.exists():
        old=json.loads(lease.read_text())
        if old['job_id'] in active or time.time()-old['time']<60: return False
    return all((root/'runs'/name/'COMPLETE.json').exists() for name in (t.get('parent'),t.get('q_source')) if name)

def environment(root):
    os.environ['PYTHONPATH']=f'{root}:{root}/v5:{root}/vendor'

def main(args):
    base=Path(args.base).resolve();ops=Path(args.ops).resolve()
    operation=json.loads((ops/'OPS_DEPLOYMENT.json').read_text())
    job=os.environ.get('SLURM_JOB_ID','local')
    if 'SLURM_ARRAY_JOB_ID' in os.environ: job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
    selected=[];versions={}
    for revision,commit in REVISIONS.items():
        root=base/revision;code=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id']
        deployment=json.loads((root/'DEPLOYMENT.json').read_text());assert deployment['commit']==commit
        gate='LEGACY_TD_VALIDATION' if revision=='91cb9c3_legacy_td' else 'VALIDATION'
        for dim in (1,2,4,8):
            v=json.loads((root/f'{gate}_D{dim}.json').read_text());assert v['passed'] and v['source_code_id']==code
        versions[str(root)]=deployment
        for i,t in enumerate(json.loads((root/'tasks.json').read_text())):
            if revision_for(t)==revision and lane_for(t)==args.lane: selected.append((root,i,t))
    eventdir=ops/'events'/args.event
    rt.write(eventdir/(job+'.json'),dict(time=time.time(),job=job,host=socket.gethostname(),lane=args.lane,operations=operation,numerical=versions))
    finished=set()
    while not rt.STOP:
        active=rt.scheduler_snapshot(job)
        if active is None: return 75
        chosen=None;pending=0
        with (ops/'claim.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            candidates=[]
            for root,i,t in selected:
                key=(str(root),i)
                if key in finished: continue
                out=root/'runs'/t['name']
                if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists(): finished.add(key);continue
                pending+=1
                if selectable(root,t,active): candidates.append((rank(t,out),root,i,t))
            if candidates:
                _,root,i,t=min(candidates,key=lambda x:x[0]);out=root/'runs'/t['name']
                lease=root/'queue'/(t['name']+'.json')
                rt.write(lease,dict(job_id=job,time=time.time(),host=socket.gethostname(),task_index=i,event=args.event,lane=args.lane))
                chosen=root,i,t,out,lease
        if not chosen:
            if not pending: print('LANE_COMPLETE',args.lane,flush=True);return 0
            print('WAITING_FOR_ELIGIBLE_WORK',args.lane,pending,flush=True);time.sleep(20);continue
        root,i,t,out,lease=chosen;environment(root)
        try: healthy,description=rt.preflight(root)
        except Exception as e: healthy,description=False,repr(e)
        print('GPU_PREFLIGHT',healthy,description,flush=True)
        if not healthy:
            rt.write(eventdir/(job+'.GPU_FAILURE.json'),dict(time=time.time(),description=description,task=t['name']));return 86
        if rt.STOP:return 75
        out.mkdir(parents=True,exist_ok=True)
        progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        rt.write(out/'attempts'/args.event/(job+'.json'),dict(time=time.time(),job=job,lane=args.lane,resume_step=progress.get('step',0),numerical_commit=versions[str(root)]['commit'],operations_commit=operation['commit']))
        print('STARTING',args.lane,i,t['name'],'resume_step',progress.get('step',0),flush=True)
        with (out/'console.log').open('a') as log:
            rt.CHILD=subprocess.Popen([sys.executable,'-m','nsq.run','--root',str(root),'--index',str(i)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            result=rt.CHILD.wait();rt.CHILD=None
        with (ops/'claim.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if lease.exists() and json.loads(lease.read_text())['job_id']==job:lease.unlink()
        if result==75 or rt.STOP:return 75
        if result and rt.infra_failure(out):
            rt.write(eventdir/(job+'.GPU_FAILURE.json'),dict(time=time.time(),task=t['name'],returncode=result));return 86
        if result and not (out/'FAILED.json').exists():rt.write(out/'FAILED.json',dict(returncode=result,time=time.time(),job_id=job))
    return 75

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--ops',required=True)
    p.add_argument('--event',required=True);p.add_argument('--lane',choices=['fast','exact'],required=True)
    sys.exit(main(p.parse_args()) or 0)
