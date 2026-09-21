"""Matched smaller-sigma grid with a common interior initialization."""
import argparse,subprocess,time
from pathlib import Path
from .campaign import read,write,occupied_gpus
from .mu90_launch import CTL,PYTHON,prepare,unchanged_algorithm
from .mu90_depth_queue import completed,env_for
SOURCE=Path(__file__).resolve().parents[1]
BASE=Path('/home/heechan/optiq-experiments')
QUEUE=BASE/'gmm40-mu90-small-sigma-queue-20260921'

def service(gpu):return f'{QUEUE.name}-gpu{gpu}'
def register():
    unchanged_algorithm(SOURCE)
    specs=read(SOURCE/'gmm40/mu90_small_sigma_grid.json');jobs=[]
    d=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    assert all(not (d/(service(g)+'.conf')).exists() for g in range(4))
    for spec in specs:
        path=SOURCE/'gmm40'/spec['plan'];p=read(path);s=p['profiles'][0];assert s['n']==s['m']==64
        root=BASE/p['name'];prepare(root,SOURCE,path)
        for x in ('logs','wandb'):(root/x).mkdir(exist_ok=True)
        jobs.append(dict(gpu=s['gpu'],root=str(root),source_commit=read(root/'manifest.json')['source_commit']))
    QUEUE.mkdir(exist_ok=True);write(QUEUE/'registration.json',dict(source=str(SOURCE),jobs=jobs,created=time.time(),automatic_restarts=False))
    for gpu in range(4):
        name=service(gpu)
        (d/(name+'.conf')).write_text('\n'.join([f'[program:{name}]',f'command={PYTHON} -B -u -m gmm40.mu90_small_sigma_queue worker --gpu {gpu}',f'directory={SOURCE}','autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true',f'stdout_logfile={QUEUE}/{name}.log','stdout_logfile_maxbytes=10MB','environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"','']))
    subprocess.run(CTL+['reread'],check=True)
    for gpu in range(4):subprocess.run(CTL+['update',service(gpu)],check=True)

def worker(gpu):
    specs=[j for j in read(QUEUE/'registration.json')['jobs'] if j['gpu']==gpu]
    statefile=QUEUE/f'status-gpu{gpu}.json'
    for i,spec in enumerate(specs):
        root=Path(spec['root']);m=read(root/'manifest.json');j=m['jobs'][0]
        marker=root/'attempts'/f'{j["name"]}.json'
        assert not marker.exists() and not (root/'results'/j['name']).exists(),'Never repeat a prior attempt'
        while True:
            write(statefile,dict(status='waiting_gpu',job=j['name'],pending=[s['root'] for s in specs[i:]],time=time.time()))
            if gpu in occupied_gpus():time.sleep(5);continue
            command=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-gmm40',PYTHON,'-B','-u','-m','gmm40.mu90_depth_queue','execute','--root',str(root),'--gpu',str(gpu)]
            logpath=root/'logs'/f'{j["name"]}.log'
            with logpath.open('x') as log:
                proc=subprocess.Popen(command,cwd=SOURCE,env=env_for(root,m),stdout=log,stderr=subprocess.STDOUT)
                write(statefile,dict(status='running',job=j['name'],pid=proc.pid,pending=[s['root'] for s in specs[i+1:]],time=time.time()));rc=proc.wait()
            if rc==3 and not marker.exists():
                logpath.rename(logpath.with_name(logpath.name+f'.lockbusy-{time.time_ns()}'));time.sleep(5);continue
            if rc:raise RuntimeError(f'Learner/preflight failed exit={rc}; no automatic retry')
            assert completed(root/'results'/j['name']);break
    write(statefile,dict(status='completed',pending=[],time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','worker']);p.add_argument('--gpu',type=int,choices=range(4));a=p.parse_args()
    try:
        if a.action=='register':register()
        else:worker(a.gpu)
    except Exception as exc:
        if a.action=='worker':write(QUEUE/f'failure-gpu{a.gpu}.json',dict(error=repr(exc),time=time.time()))
        raise
if __name__=='__main__':main()
