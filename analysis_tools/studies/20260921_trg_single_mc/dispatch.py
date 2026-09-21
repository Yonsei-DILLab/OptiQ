"""Six GPU smoke checks then independent 24-run priority queue, two/GPU.

Never resumes or launches a previous campaign. A failed seed is recorded, not retried.
All runs are eligible together; environment order is priority, not dependency.
"""
import csv
import fcntl
import math
import shutil
import os
import subprocess
import sys
import time
from ops import ROOT,ORDER,read,write,verify,environment,command


def spawn(env,seed,gpu,slot,commit,m=64,smoke=False):
    name=f'{env}_m{m}_s{seed}'
    out=ROOT/('validation' if smoke else 'runs')/name
    out.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))
    offset=(gpu+4*slot)*4
    cpus=cpus[offset:offset+4]
    assert len(cpus)==4
    cmd=['taskset','-c',','.join(map(str,cpus)),*command(env,seed,commit,m,smoke)]
    log=(out/'stdout.log').open('w')
    proc=subprocess.Popen(cmd,cwd=ROOT/'repo',env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
    record=dict(name=name,env=env,seed=seed,M=m,gpu=gpu,slot=slot,pid=proc.pid,
        cpus=cpus,command=cmd,commit=commit,state='running',started=time.time(),validation=smoke)
    write(out/'LAUNCH.json',record)
    return proc,log,out,record


def result(item):
    proc,log,out,r=item
    code=proc.poll()
    if code is None:return None
    log.close()
    markers=list(out.glob('*/completed.json'))
    r.update(exit_code=code,finished=time.time(),state='failed')
    if code==0 and len(markers)==1:
        complete=read(markers[0])
        assert complete['timesteps']==(128 if r['validation'] else 1000000)
        r.update(state='complete',**complete)
    write(out/'STATUS.json',r)
    return r


def check_smoke(r):
    assert r['state']=='complete',r
    out=ROOT/'validation'/r['name']
    actual=next(out.glob('*/completed.json')).parent
    with (actual/'logs/progress.csv').open() as f:rows=list(csv.DictReader(f))
    for field in ['train/actor_loss','train/critic_loss','train/backup_q_mc_se']:
        values=[float(x[field]) for x in rows if x.get(field)]
        assert values and all(math.isfinite(x) for x in values),field
    for field,value in [('train/critic_count',1),('train/backup_samples',64),('train/backup_entropy_term',0)]:
        values=[float(x[field]) for x in rows if x.get(field)]
        assert values and all(x==value for x in values),(field,values)
    import wandb
    os.environ['WANDB_API_KEY']=environment(0)['WANDB_API_KEY']
    api=wandb.Api(timeout=20)
    rid=r['wandb_url'].rstrip('/').split('/')[-1]
    for _ in range(30):
        api.flush();remote=api.run(f'OptiQ/DirectGMM_heejoon/{rid}')
        if remote.summary.get('completed') and remote.summary.get('timesteps')==128:break
        time.sleep(5)
    assert remote.summary.get('completed') and remote.summary.get('timesteps')==128
    assert remote.config['alg']['critic']['n_critics']==1
    assert remote.config['alg']['critic']['backup_samples']==64
    assert remote.config['alg']['actor']['log_std_max']==-1
    assert remote.config['alg']['actor']['proposals_per_policy_sample']==r['M']//64
    stds=[float(x['train/actor_std_max']) for x in rows if x.get('train/actor_std_max')]
    assert stds and max(stds)<=math.exp(-1)+1e-5
    return dict(**r,online_verified=True)


def main():
    lock=(ROOT/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=verify();assert not (ROOT/'STATUS.json').exists(),'Existing queue; inspect instead of restarting'
    commit=source['commit']
    env=environment(0);env['JAX_PLATFORMS']='cpu'
    with (ROOT/'unit.log').open('w') as f:
        subprocess.run(['taskset','-c',','.join(map(str,sorted(os.sched_getaffinity(0))[:4])),sys.executable,str(ROOT/'validate_mc.py')],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'preflight.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'preflight.py')],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'truncated_tests.log').open('w') as f:
        subprocess.run(['taskset','-c',','.join(map(str,sorted(os.sched_getaffinity(0))[:4])),
            sys.executable,'-m','pytest','-q',str(ROOT/'test_truncated.py')],env=env,cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,check=True)
    active=[]; smoke_pending=[(name,m) for name in ORDER for m in (64,256)];passed=[]
    while active or smoke_pending:
        for item in active[:]:
            r=result(item)
            if r:
                active.remove(item);passed.append(check_smoke(r))
                write(ROOT/'SMOKE_PROGRESS.json',passed)
        used={it[3]['gpu'] for it in active}
        for gpu in range(4):
            if smoke_pending and gpu not in used:
                name,m=smoke_pending.pop(0)
                active.append(spawn(name,0,gpu,0,commit,m,True))
        time.sleep(2)
    write(ROOT/'VALIDATION.json',dict(passed=True,commit=commit,smoke=passed,time=time.time()))
    slots=[(gpu,slot) for slot in range(2) for gpu in range(4)]
    pending=[(name,seed,m) for name in ORDER for seed in range(4) for m in (64,256)]
    active=[];finished=[]
    while pending or active:
        for item in active[:]:
            r=result(item)
            if r:active.remove(item);finished.append(r)
        occupied={(it[3]['gpu'],it[3]['slot']) for it in active}
        for gpu,slot in slots:
            if pending and (gpu,slot) not in occupied and not (ROOT/'PAUSE_QUEUE').exists() and shutil.disk_usage(ROOT).free > 2*1024**3:
                name,seed,m=pending.pop(0)
                active.append(spawn(name,seed,gpu,slot,commit,m))
        write(ROOT/'STATUS.json',dict(commit=commit,time=time.time(),
            pending=[dict(env=e,seed=s,M=m) for e,s,m in pending],running=[it[3] for it in active],finished=finished))
        time.sleep(5)
    write(ROOT/'QUEUE_FINISHED.json',dict(commit=commit,finished=finished,time=time.time()))


if __name__=='__main__':main()
