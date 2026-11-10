"""Supervise three vector Fetch scale conditions with two seeds each."""
import argparse
from collections import Counter
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19]) if fields[0]!='Z' else None
    except OSError:return None


def gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    result={}
    for row in csv.reader(io.StringIO(output)):
        try:result[int(row[0])]=int(row[1])
        except (ValueError,IndexError):pass
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text());children={};stopping=False
    def handler(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    def env(gpu):
        return dict(os.environ,PYTHONPATH=plan['source'],CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',
            JAX_COMPILATION_CACHE_DIR=str(folder/'jax-cache'),MPLCONFIGDIR=str(folder/'mpl-cache'))
    def spawn(job,smoke=False):
        spec={k:job[k] for k in ('id','variant','seed','source','source_manifest','worker_cpus','output')}
        spec.update({k:job[k] for k in plan.get('extra_spec_fields',[])})
        specpath=folder/'specs'/(job['id']+'.json');atomic(specpath,spec)
        command=['taskset','-c',','.join(map(str,job['learner_cpus'])),plan['python'],'-u','-m',
            plan.get('worker_module','rebuttal_tools.fetch_sigma_vector_worker'),'--spec',str(specpath)]
        if smoke:command+=['--smoke']
        with (folder/'logs'/(job['id']+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=plan['source'],env=env(job['gpu']),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        job.update(status='initializing',pid=child.pid,start_ticks=identity(child.pid),started_unix=time.time())
        children[child.pid]=child
        print('Started '+job['id']+' GPU'+str(job['gpu']),flush=True)
    def update(jobs,preflight=False):
        apps=gpu_processes() if preflight else {}
        for job in jobs:
            if job['status'] not in ('initializing','running'):continue
            out=Path(job['output']);job['peak_gpu_mib']=max(job.get('peak_gpu_mib',0),apps.get(job['pid'],0))
            marker=out/plan.get('completion_marker','completed.json')
            if marker.exists():
                result=json.loads(marker.read_text())
                assert result['updates']==(64 if preflight else 123424) and result['observed_utd']==.125
                job.update(status='completed',result=result,finished_unix=time.time())
            elif (out/'failure.json').exists() or identity(job['pid'])!=job['start_ticks']:
                failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else dict(error='Exited without completion')
                job.update(status='failed',error=failure['error'],finished_unix=time.time())
            elif (out/'progress.json').exists():
                job['progress']=json.loads((out/'progress.json').read_text())
                if job['progress'].get('updates',0)>0:job['status']='running'
        for pid,child in list(children.items()):
            if child.poll() is not None:child.wait();del children[pid]
    try:
        plan.update(controller_pid=os.getpid(),status='validating_cpu');atomic(path,plan)
        if not plan.get('cpu_checks_passed'):
            cpuenv=env('');cpuenv['JAX_PLATFORMS']='cpu'
            with (folder/'logs'/'cpu-checks.log').open('w') as stream:
                subprocess.run(['taskset','-c',','.join(map(str,plan['preflights'][0]['learner_cpus'])),
                    plan['python'],'-u','-m','rebuttal_tools.check_fetch_vector','--collector'],
                    cwd=plan['source'],env=cpuenv,stdout=stream,stderr=subprocess.STDOUT,check=True)
            plan['cpu_checks_passed']=True;atomic(path,plan)
        plan['status']='validating_gpu';atomic(path,plan)
        for job in plan['preflights']:
            if job['status']=='queued':spawn(job,True)
        while any(j['status'] in ('initializing','running') for j in plan['preflights']):
            if stopping:raise InterruptedError('Stopped during preflight')
            update(plan['preflights'],True);atomic(path,plan);time.sleep(3)
        assert all(j['status']=='completed' for j in plan['preflights']), 'GPU preflight failed; see per-case logs'
        atomic(folder/'gpu-preflight-verification.json',dict(passed=True,cases=plan['preflights']))
        plan['status']='running'
        for job in plan['jobs']:
            if job['status']=='queued':spawn(job)
        while True:
            update(plan['jobs'])
            if stopping:raise InterruptedError('Campaign stopped')
            plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])),checked_unix=time.time())
            atomic(path,plan)
            if not any(j['status'] in ('initializing','running') for j in plan['jobs']):
                plan['status']='completed_with_failures' if plan['counts'].get('failed') else 'completed'
                atomic(path,plan);return
            time.sleep(10)
    except BaseException as error:
        for job in (*plan['preflights'],*plan['jobs']):
            if job['status'] in ('initializing','running') and identity(job.get('pid',0))==job.get('start_ticks'):
                try:os.kill(job['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        plan.update(status='stopped' if stopping else 'controller_error',error=repr(error));atomic(path,plan);raise


if __name__=='__main__':main()
