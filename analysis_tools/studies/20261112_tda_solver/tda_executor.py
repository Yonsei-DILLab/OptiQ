"""Own-child-only handoff between a reservation and committed TDA jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    for name in ('pending','active','completed','logs'):(a.root/name).mkdir(exist_ok=True)
    if list((a.root/'active').glob('*.json')):raise RuntimeError('Unresolved active job; inspect before restarting')
    alive=True;hold=None;hold_log=None
    def stop(signum,frame):
        nonlocal alive
        alive=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(**kw):
        data=dict(controller_pid=os.getpid(),gpu=a.gpu,source_commit=a.commit,updated_unix=time.time(),**kw)
        tmp=a.root/'status.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(a.root/'status.json')
    def stop_hold():
        nonlocal hold,hold_log
        if hold is not None:
            if hold.poll() is None:
                hold.terminate()
                try:hold.wait(timeout=10)
                except subprocess.TimeoutExpired:hold.kill();hold.wait()
            hold_log.close();hold=None;hold_log=None
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        while alive:
            jobs=sorted((a.root/'pending').glob('*.json'))
            if jobs:
                stop_hold();path=jobs[0];active=a.root/'active'/path.name;path.replace(active)
                job=json.loads(active.read_text());cmd=job['argv']
                if not isinstance(cmd,list) or not all(isinstance(x,str) for x in cmd):raise ValueError('argv must be a string list')
                with (a.root/'logs'/(active.stem+'.log')).open('a') as log:
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    status(state='experiment',job=active.name,child_pid=child.pid,argv=cmd)
                    job.update(started_unix=time.time(),pid=child.pid)
                    # Finish an already-running real job even if stop is requested.
                    rc=child.wait();job.update(finished_unix=time.time(),returncode=rc)
                active.write_text(json.dumps(job,indent=2)+'\n');active.replace(a.root/'completed'/active.name)
                continue
            if hold is None or hold.poll() is not None:
                stop_hold();hold_log=(a.root/'logs'/'runtime.log').open('a')
                hold=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('tda_runtime.py')),'--gpu',a.gpu],
                                      env=env,stdout=hold_log,stderr=subprocess.STDOUT)
                status(state='reservation',child_pid=hold.pid)
            time.sleep(.25)
    finally:
        stop_hold();status(state='stopped')


if __name__=='__main__':main()
