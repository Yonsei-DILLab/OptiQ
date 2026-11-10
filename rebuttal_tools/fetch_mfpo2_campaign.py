"""Resource-aware five-seed Fetch suite; each method advances after its five seeds finish."""
import argparse
from collections import Counter, deque
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def atomic(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(s[19]) if s[0]!='Z' else None
    except OSError:return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    path=a.plan.resolve();folder=path.parent
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text());children={};stopping=False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    topology={}
    for cpu in sorted(os.sched_getaffinity(0)):
        d=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=(d.joinpath('physical_package_id').read_text().strip(),d.joinpath('core_id').read_text().strip())
        topology.setdefault(key,[]).append(cpu)
    groups=list(topology.values());physical={c:k for k,cs in topology.items() for c in cs}
    history={i:deque(maxlen=3) for i in range(4)}
    def env(gpu):
        return dict(os.environ,PYTHONPATH=plan['source']+os.pathsep+plan.get('extra_pythonpath',''),CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',
            JAX_COMPILATION_CACHE_DIR=str(folder/'jax-cache'),MPLCONFIGDIR=str(folder/'mpl-cache'))
    def resources():
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
        gpus={int(i):dict(free_mib=int(m),util=int(u)) for i,m,u in csv.reader(io.StringIO(raw))}
        reserved={physical[c] for cs in groups[:plan.get('reserve_first_physical',0)] for c in cs}
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
        pids={int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()}
        for j in plan['jobs']+plan['preflights']:
            if j.get('pid') and identity(j['pid'])==j.get('start_ticks'):
                pids.add(j['pid'])
                cpus=j.get('learner_cpus',[])+[c for cs in j.get('worker_cpus',[]) for c in cs]
                reserved.update(physical[c] for c in cpus if c in physical)
        for pid in list(pids):
            try:pids.update(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
            except OSError:pass
        for pid in pids:
            try:cpus=os.sched_getaffinity(pid)
            except ProcessLookupError:continue
            if len(cpus)<=16:reserved.update(physical[c] for c in cpus if c in physical)
        # The pre-existing fixed 5M campaign can still have not-yet-GPU-visible workers.
        for filename in plan.get('fixed_reserve_plans',[]):
            previous=json.loads(Path(filename).read_text())
            for j in previous.get('jobs',[])+previous.get('preflights',[]):
                if j['status'] not in ('queued','initializing','running'):continue
                cpus=list(j.get('learner_cpus',[]))+[c for cs in j.get('worker_cpus',[]) for c in cs]
                reserved.update(physical[c] for c in cpus if c in physical)
        for i,g in gpus.items():history[i].append(g['util']);g['mean_util']=statistics.mean(history[i])
        return gpus,[cs for k,cs in topology.items() if k not in reserved]
    def update():
        for j in plan['preflights']+plan['jobs']:
            if j['status'] not in ('initializing','running'):continue
            out=Path(j['output']);marker=out/'verified-completion.json'
            if marker.exists():
                done=json.loads(marker.read_text())
                assert done['updates']==(64 if j['phase']=='preflight' else 623424) and done['observed_utd']==.125
                j.update(status='completed',result=done,finished_unix=time.time())
            elif (out/'failure.json').exists() or identity(j['pid'])!=j['start_ticks']:
                fail=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else dict(error='Exited without verified completion')
                j.update(status='failed',error=fail['error'],finished_unix=time.time())
            elif (out/'progress.json').exists():
                j['progress']=json.loads((out/'progress.json').read_text())
                if j['progress'].get('updates',0)>0:j['status']='running'
        for pid,child in list(children.items()):
            if child.poll() is not None:child.wait();del children[pid]
    def spawn(j,gpu,lanes):
        j.update(gpu=gpu,learner_cpus=lanes[0],worker_cpus=lanes[1:])
        spec={k:j[k] for k in ('id','method','task','seed','source','source_manifest','worker_cpus','output',
            'ibolt_reference_config','temperature','mfpo_target_entropy_coeff','mfpo_initial_temperature')}
        specpath=folder/'specs'/(j['id']+'.json');atomic(specpath,spec)
        args=['taskset','-c',','.join(map(str,j['learner_cpus'])),j['python'],'-u','-m','rebuttal_tools.fetch_mfpo2_worker','--spec',str(specpath)]
        if j['phase']=='preflight':args+=['--smoke']
        with (folder/'logs'/(j['id']+'.log')).open('x') as stream:
            child=subprocess.Popen(args,cwd=plan['source'],env=env(gpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        j.update(status='initializing',pid=child.pid,start_ticks=identity(child.pid),started_unix=time.time());children[child.pid]=child
        print('Started '+j['id']+' GPU'+str(gpu),flush=True)
    try:
        plan.update(controller_pid=os.getpid(),status='validating_cpu');atomic(path,plan)
        if not plan.get('cpu_checks_passed'):
            with (folder/'logs'/'cpu-checks.log').open('w') as stream:
                e=env('');e['JAX_PLATFORMS']='cpu'
                subprocess.run(['taskset','-c',','.join(map(str,groups[-1])),plan['ibolt_python'],'-u','-m',
                    'rebuttal_tools.check_fetch_suite_vector'],cwd=plan['source'],env=e,stdout=stream,stderr=subprocess.STDOUT,check=True)
            plan['cpu_checks_passed']=True
        while True:
            if stopping:raise InterruptedError('Study stopped')
            update()
            if any(j['status']=='failed' for j in plan['preflights']):raise RuntimeError('GPU preflight failed; full cohort has not been released')
            passed=all(j['status']=='completed' for j in plan['preflights'])
            if passed:
                plan['gpu_checks_passed']=True
            collection=plan['jobs'] if passed else plan['preflights']
            try:
                gpus,available=resources()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
                plan.update(observation_error=repr(error),checked_unix=time.time());atomic(path,plan)
                time.sleep(10);continue
            active=[j for j in plan['jobs']+plan['preflights'] if j['status'] in ('initializing','running')]
            for j in collection:
                if j['status']!='queued':continue
                if passed:
                    order=plan['task_order'][j['method']]
                    prior=order[:order.index(j['task'])]
                    if any(x['method']==j['method'] and x['task'] in prior and x['status']!='completed' for x in plan['jobs']):continue
                    if any(x['method']==j['method'] and x['task']==j['task'] and x['status']=='failed' for x in plan['jobs']):continue
                if len(active)>=plan['max_active'] or len(available)<3:break
                counts=Counter(x['gpu'] for x in active);initializing=Counter(x['gpu'] for x in active if x['status']=='initializing')
                eligible=[i for i,g in gpus.items() if len(history[i])==3 and g['free_mib']>=6144 and g['mean_util']<=plan['max_admission_util']
                    and counts[i]<plan['max_per_gpu'] and initializing[i]<2]
                if not eligible:break
                gpu=min(eligible,key=lambda i:(counts[i]*40+gpus[i]['mean_util'],-gpus[i]['free_mib'],i))
                spawn(j,gpu,available[:3]);available=available[3:];active.append(j);gpus[gpu]['free_mib']-=3072
            plan.update(status='running' if passed else 'validating_gpu_or_waiting_capacity',
                counts=dict(Counter(j['status'] for j in plan['jobs'])),checked_unix=time.time(),
                admission=dict(gpus=gpus,unused_physical_cores=len(available)))
            atomic(path,plan)
            if passed and all(j['status'] in ('completed','failed') for j in plan['jobs']):
                plan['status']='completed_with_failures' if any(j['status']=='failed' for j in plan['jobs']) else 'completed'
                atomic(path,plan);return
            time.sleep(10)
    except BaseException as error:
        for j in plan['jobs']+plan['preflights']:
            if j.get('pid') and identity(j['pid'])==j.get('start_ticks'):
                try:os.kill(j['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        plan.update(status='stopped' if stopping else 'controller_error',error=repr(error));atomic(path,plan);raise


if __name__=='__main__':main()
