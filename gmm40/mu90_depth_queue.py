"""One sequential depth-ablation queue per GPU, without learner retries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .campaign import read,write,occupied_gpus
from .mu90_launch import CTL,PYTHON,prepare,preflight,unchanged_algorithm

BASE_ROOT=Path('/home/heechan/optiq-experiments')
PARENT=BASE_ROOT/'gmm40-mu90-screen1-20260921'
QUEUE=BASE_ROOT/'gmm40-mu90-depth-queue-20260921'
SOURCE=Path(__file__).resolve().parents[1]
ROOTS=[BASE_ROOT/f'gmm40-mu90-{w}x3-20260921' for w in (256,512)]

def completed(folder,steps=100000):
    a=read(folder/'update_count_audit.json',{})
    return (a.get('status')=='passed' and a.get('full_budget_completed') is True
            and a.get('actor_updates')==steps and a.get('final_evaluation_step')==steps)

def env_for(root,m):
    return dict(os.environ,OPTIQ_SOURCE_DIR=m['source'],GMM40_REPO_ROOT=m['source'],
        GMM40_RESULTS_ROOT=str(root/'results'),GMM40_SOURCE_COMMIT=m['source_commit'],
        GMM40_CAMPAIGN=m['plan']['name'],GMM40_WANDB_DIR=str(root/'wandb'),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')

def execute(root,gpu):
    m=read(root/'manifest.json');j=next(j for j in m['jobs'] if j['gpu']==gpu)
    assert m['source']==str(SOURCE)
    unchanged_algorithm(SOURCE)
    marker=root/'attempts'/f'{j["name"]}.json'
    marker.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents restarting an already attempted learner.
    with marker.open('x') as f:f.write('{"status":"started"}\n')
    os.environ.update(env_for(root,m))
    pm=dict(m,plan=dict(m['plan'],profiles=[j['profile']]))
    preflight(root/'preflights'/j['name'],pm)
    result=subprocess.run([PYTHON,'-u','-m','gmm40.campaign_job',*j['args']],cwd=SOURCE)
    if result.returncode or not completed(root/'results'/j['name']):
        raise RuntimeError(f'Learner failed or incomplete: {j["name"]}, exit={result.returncode}')
    write(marker,dict(status='completed',time=time.time()))

def worker(gpu):
    status=QUEUE/f'status-gpu{gpu}.json'
    previous=read(PARENT/'manifest.json');pj=next(j for j in previous['jobs'] if j['gpu']==gpu)
    while True:
        out=subprocess.run(CTL+['status',pj['name']],capture_output=True,text=True).stdout.strip()
        state=out.split()[1] if len(out.split())>1 else 'UNKNOWN'
        write(status,dict(status='waiting_predecessor',predecessor=pj['name'],supervisor_state=state,pending=[str(r) for r in ROOTS],time=time.time()))
        if state=='EXITED' and completed(PARENT/'results'/pj['name']):break
        if state not in ('RUNNING','STARTING'):
            raise RuntimeError(f'Predecessor stopped without verified completion: {out}')
        time.sleep(5)
    for idx,root in enumerate(ROOTS):
        m=read(root/'manifest.json');j=next(j for j in m['jobs'] if j['gpu']==gpu)
        assert not (root/'results'/j['name']).exists()
        marker=root/'attempts'/f'{j["name"]}.json'
        assert not marker.exists(),'Never restart a prior attempt'
        while True:
            write(status,dict(status='waiting_gpu',job=j['name'],pending=[str(r) for r in ROOTS[idx:]],time=time.time()))
            if gpu in occupied_gpus():time.sleep(5);continue
            command=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-gmm40',PYTHON,'-u','-m','gmm40.mu90_depth_queue','execute','--root',str(root),'--gpu',str(gpu)]
            with (root/'logs'/f'{j["name"]}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=SOURCE,env=env_for(root,m),stdout=log,stderr=subprocess.STDOUT)
                write(status,dict(status='running',job=j['name'],pid=proc.pid,pending=[str(r) for r in ROOTS[idx+1:]],time=time.time()))
                rc=proc.wait()
            if rc==3 and not marker.exists():
                logpath=root/'logs'/f'{j["name"]}.log'
                logpath.rename(logpath.with_name(logpath.name+f'.lockbusy-{time.time_ns()}'))
                time.sleep(5);continue
            if rc:raise RuntimeError(f'Job {j["name"]} failed exit={rc}; no automatic restart')
            assert completed(root/'results'/j['name'])
            break
    write(status,dict(status='completed',pending=[],time=time.time()))

def register():
    unchanged_algorithm(SOURCE)
    for width,root in zip((256,512),ROOTS):
        prepare(root,SOURCE,SOURCE/f'gmm40/mu90_{width}x3_plan.json')
        for d in ['logs','wandb']:(root/d).mkdir(exist_ok=True)
    QUEUE.mkdir(exist_ok=True)
    d=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services=[f'gmm40-mu90-depth-queue-20260921-gpu{i}' for i in range(4)]
    assert all(not (d/(s+'.conf')).exists() for s in services)
    for gpu,name in enumerate(services):
        (d/(name+'.conf')).write_text('\n'.join([f'[program:{name}]',f'command={PYTHON} -u -m gmm40.mu90_depth_queue worker --gpu {gpu}',f'directory={SOURCE}','autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true',f'stdout_logfile={QUEUE}/{name}.log','stdout_logfile_maxbytes=10MB','environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"','']))
    write(QUEUE/'registration.json',dict(source=str(SOURCE),source_commit=read(ROOTS[0]/'manifest.json')['source_commit'],services=services,parent=str(PARENT),roots=[str(r) for r in ROOTS],automatic_restarts=False,time=time.time()))
    subprocess.run(CTL+['reread'],check=True)
    for name in services:subprocess.run(CTL+['update',name],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','worker','execute']);p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--root',type=Path);a=p.parse_args()
    try:
        if a.action=='register':register()
        elif a.action=='execute':execute(a.root,a.gpu)
        else:worker(a.gpu)
    except Exception as exc:
        if a.action=='worker':write(QUEUE/f'failure-gpu{a.gpu}.json',dict(error=repr(exc),time=time.time()))
        raise

if __name__=='__main__':main()
