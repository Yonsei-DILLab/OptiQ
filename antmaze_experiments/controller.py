"""Independent per-GPU preflight/train jobs; failure holds pending jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .run import write, CAMPAIGN


def job(root, identifier, gpu):
    manifest=json.loads((root/'manifest.json').read_text())
    entry=next(j for j in manifest['jobs'] if j['id']==identifier)
    source=Path(manifest['source'])
    for phase in ('preflight','runs'):
        target=root/phase/identifier
        cmd=['bash',str(source/'antmaze_experiments/launch.sh'),
             '--method',entry['method'],'--task',entry['task'],'--output',str(target)]
        if phase=='preflight':cmd.append('--preflight')
        write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,pid=os.getpid(),phase=phase,status='running'))
        with (root/'logs'/f'{identifier}-{phase}.log').open('x') as log:
            result=subprocess.run(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,phase=phase,status='failed',returncode=result.returncode))
            return result.returncode
        proof=json.loads((target/'result.json').read_text())
        assert proof['completed'] and proof['source_commit']==manifest['source_commit']
    write(root/'jobs'/f'{identifier}.json',dict(**entry,gpu=gpu,status='completed'))
    return 0


def controller(root):
    lock=(root/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text())
    source=Path(manifest['source'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==manifest['source_commit']
    pending=list(manifest['jobs']);live={};completed=[];failed=[]
    for directory in ('jobs','logs','preflight','runs'): (root/directory).mkdir(exist_ok=True)
    if (root/'status.json').exists():raise RuntimeError('Existing controller state: do not restart automatically')
    while pending or live:
        for gpu,(proc,entry) in list(live.items()):
            code=proc.poll()
            if code is None:continue
            del live[gpu]
            if code==0:completed.append(entry['id'])
            else:
                failed.append(dict(id=entry['id'],returncode=code,gpu=gpu))
                write(root/'failure.json',dict(failed=failed,pending_held=True,time=time.time()))
        if not failed:
            for gpu in range(4):
                if gpu in live or not pending:continue
                lock_path=Path(f'/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock')
                with lock_path.open('a') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                entry=pending.pop(0)
                env=dict(os.environ,OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu))
                cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',
                     '/home/heechan/.venv-ddiffpg-native/bin/python','-m','antmaze_experiments.controller',
                     '--root',str(root),'--job',entry['id'],'--gpu',str(gpu)]
                with (root/'logs'/f"{entry['id']}-job.log").open('x') as log:
                    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                live[gpu]=(proc,entry)
        state=dict(source_commit=manifest['source_commit'],controller_pid=os.getpid(),time=time.time(),
            pending=[j['id'] for j in pending],running=[dict(**entry,gpu=gpu,pid=proc.pid) for gpu,(proc,entry) in live.items()],
            completed=completed,failed=failed,pending_held=bool(failed))
        write(root/'status.json',state)
        if failed and not live: return 1
        time.sleep(2)
    results={key:json.loads((root/'runs'/key/'result.json').read_text()) for key in completed}
    write(root/'result.json',dict(completed=True,source_commit=manifest['source_commit'],results=results))
    return 0


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--job');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    sys.exit(job(a.root,a.job,a.gpu) if a.job else controller(a.root))


if __name__=='__main__':main()
