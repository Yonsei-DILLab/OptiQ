"""Preserve live Ant runs; keep canceled HalfCheetah off GPU 0/1 for AntMaze."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

SOURCE=Path('/home/heechan/OptiQ-ops/sources/8cba41497da673c5b643e2fd41536d7951f6c9da')
spec=importlib.util.spec_from_file_location('mean1_controller',SOURCE/'analysis_tools/experiments/20260921_dacer_mean1/controller.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ROOT=old.ROOT

class Adopted:
    def __init__(self,job):
        self.pid=job['pid'];self.expected=[old.PYTHON,'-u',str(old.HERE/'train_mean1.py'),job['task'],str(job['seed']),str(ROOT/'outputs')]
        assert self.command()==self.expected,(self.pid,self.command())
    def command(self):
        return [s.decode() for s in Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0') if s]
    def poll(self):
        p=Path(f'/proc/{self.pid}/stat')
        if not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z':return 'unavailable_after_scheduler_handoff'
        assert self.command()==self.expected,'Adopted PID was reused'
        return None

def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'manifest.json').read_text());assert manifest['gpus']==[2,3]
    jobs=[json.loads((ROOT/'jobs'/(n+'.json')).read_text()) for n in manifest['jobs']]
    assert all(j['status']=='canceled' for j in jobs if j['task']=='halfcheetah')
    ants=[j for j in jobs if j['task']=='ant'];assert sorted(j['seed'] for j in ants)==list(range(4))
    active={j['gpu']:(Adopted(j),j,None) for j in ants if j['status']=='running'}
    for _,j,_ in active.values():print(json.dumps(dict(event='adopted',name=j['name'],pid=j['pid'],gpu=j['gpu'])),flush=True)
    pending=[j for j in ants if j['status']=='queued']
    while active or pending:
        for gpu,(p,j,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            if log:log.close()
            del active[gpu];j.update(status='failed',exit_code=code,finished=time.time())
            try:
                j.update(old.verify(j))
                assert code in (0,'unavailable_after_scheduler_handoff') or j.get('logging_warning')
            except Exception as exc:j.update(status='failed',error=repr(exc))
            old.save(ROOT/'jobs'/(j['name']+'.json'),j)
            print(json.dumps(dict(event='finished',name=j['name'],status=j['status'])),flush=True)
        failed=[j for j in ants if j['status']=='failed']
        for gpu in [2,3]:
            if failed or not pending or gpu in active or not old.available(gpu):continue
            j=pending.pop(0)
            cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',old.PYTHON,'-u',str(old.HERE/'train_mean1.py'),j['task'],str(j['seed']),str(ROOT/'outputs')]
            env=os.environ.copy();env.pop('JAX_PLATFORMS',None)
            env.update(OPTIQ_SOURCE_DIR=str(SOURCE),CAMPAIGN_GPU=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(old.DEPS),WANDB_MODE='online')
            log=(ROOT/'logs'/(j['name']+'.log')).open('x')
            p=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            j.update(status='running',pid=p.pid,gpu=gpu,command=cmd,started=time.time())
            old.save(ROOT/'jobs'/(j['name']+'.json'),j);active[gpu]=(p,j,log)
            print(json.dumps(dict(event='started',name=j['name'],pid=p.pid,gpu=gpu)),flush=True)
        old.save(ROOT/'status.json',dict(phase='failed' if failed else 'running' if active or pending else 'completed',updated=time.time(),commit=manifest['commit'],gpus=[2,3],
            **{s:[j['name'] for j in jobs if j['status']==s] for s in ['running','queued','completed','failed','canceled']}))
        if failed:
            old.save(ROOT/'failure.json',dict(failed=failed,updated=time.time()))
            if not active:raise RuntimeError('Ant failure; pending jobs preserved; no retries')
        if active or pending:time.sleep(2)
    aggregate={}
    for mode in ['stochastic_z','zero_z']:
        v=[j['metrics'][mode+'_last_100k_mean'] for j in ants]
        aggregate[mode]=dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),seeds=[j['seed'] for j in ants])
    old.save(ROOT/'result.json',dict(status='completed',commit=manifest['commit'],jobs=jobs,aggregate={'ant':aggregate},expected_completed_runs=4,
        canceled_halfcheetah=[j['name'] for j in jobs if j['task']=='halfcheetah'],window='900000 < env_steps <= 1000000; 20 evaluations x 10 episodes',completed=time.time()))

if __name__=='__main__':main()
