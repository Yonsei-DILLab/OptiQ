"""Narrow-sigma matched follow-up; starts only after each depth queue completes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .campaign import read,write,occupied_gpus
from .mu90_launch import CTL,PYTHON,prepare,unchanged_algorithm
from .mu90_depth_queue import completed,env_for

SOURCE=Path(__file__).resolve().parents[1]
BASE=Path('/home/heechan/optiq-experiments')
QUEUE=BASE/'gmm40-mu90-narrow-queue-20260921'
PREVIOUS=BASE/'gmm40-mu90-512x3-20260921'

def service(gpu):return f'gmm40-mu90-narrow-queue-20260921-gpu{gpu}'
def plan_paths():return sorted((SOURCE/'gmm40').glob('mu90_narrow_gpu*.json'))
def register():
    unchanged_algorithm(SOURCE)
    plans=[read(p) for p in plan_paths()];assert len(plans)==4
    jobs=[];d=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    for path,p in zip(plan_paths(),plans):
        gpu=p['profiles'][0]['gpu'];root=BASE/p['name'];assert not (d/(service(gpu)+'.conf')).exists()
        prepare(root,SOURCE,path)
        for x in ['logs','wandb']:(root/x).mkdir(exist_ok=True)
        jobs.append(dict(gpu=gpu,root=str(root),source_commit=read(root/'manifest.json')['source_commit']))
    QUEUE.mkdir(exist_ok=True)
    write(QUEUE/'registration.json',dict(jobs=jobs,source=str(SOURCE),created=time.time(),automatic_restarts=False))
    for j in jobs:
        gpu=j['gpu'];name=service(gpu)
        (d/(name+'.conf')).write_text('\n'.join([f'[program:{name}]',f'command={PYTHON} -B -u -m gmm40.mu90_narrow_queue worker --gpu {gpu}',f'directory={SOURCE}','autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true',f'stdout_logfile={QUEUE}/{name}.log','stdout_logfile_maxbytes=10MB','environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"','']))
    subprocess.run(CTL+['reread'],check=True)
    for j in jobs:subprocess.run(CTL+['update',service(j['gpu'])],check=True)

def worker(gpu):
    reg=read(QUEUE/'registration.json');root=Path(next(j['root'] for j in reg['jobs'] if j['gpu']==gpu));m=read(root/'manifest.json');j=m['jobs'][0]
    pred=next(j for j in read(PREVIOUS/'manifest.json')['jobs'] if j['gpu']==gpu)
    statefile=QUEUE/f'status-gpu{gpu}.json';pred_service=f'gmm40-mu90-depth-queue-20260921-gpu{gpu}'
    while True:
        out=subprocess.run(CTL+['status',pred_service],capture_output=True,text=True).stdout.strip()
        state=out.split()[1] if len(out.split())>1 else 'UNKNOWN'
        write(statefile,dict(status='waiting_predecessor',predecessor=pred_service,supervisor_state=state,pending=[j['name']],time=time.time()))
        if state=='EXITED' and completed(PREVIOUS/'results'/pred['name']):break
        if state not in ('RUNNING','STARTING'):raise RuntimeError('Incomplete predecessor: '+out)
        time.sleep(5)
    marker=root/'attempts'/f'{j["name"]}.json'
    assert not marker.exists() and not (root/'results'/j['name']).exists()
    while True:
        if gpu in occupied_gpus():time.sleep(5);continue
        command=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-gmm40',PYTHON,'-B','-u','-m','gmm40.mu90_depth_queue','execute','--root',str(root),'--gpu',str(gpu)]
        logpath=root/'logs'/f'{j["name"]}.log'
        with logpath.open('x') as log:
            proc=subprocess.Popen(command,cwd=SOURCE,env=env_for(root,m),stdout=log,stderr=subprocess.STDOUT)
            write(statefile,dict(status='running',job=j['name'],pid=proc.pid,pending=[],time=time.time()));rc=proc.wait()
        if rc==3 and not marker.exists():
            logpath.rename(logpath.with_name(logpath.name+f'.lockbusy-{time.time_ns()}'));time.sleep(5);continue
        if rc:raise RuntimeError(f'Learner/preflight failed exit={rc}; no restart')
        assert completed(root/'results'/j['name']);break
    write(statefile,dict(status='completed',job=j['name'],pending=[],time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','worker']);p.add_argument('--gpu',type=int,choices=range(4));a=p.parse_args()
    try:
        if a.action=='register':register()
        else:worker(a.gpu)
    except Exception as exc:
        if a.action=='worker':write(QUEUE/f'failure-gpu{a.gpu}.json',dict(error=repr(exc),time=time.time()))
        raise
if __name__=='__main__':main()
