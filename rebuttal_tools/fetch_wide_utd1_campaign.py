"""Concurrent four-task Fetch rerun with wide scales and native UTD1 collection."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from .fetch_wide_utd1_worker import atomic

TEMPERATURES=(.05,.1,.25,.5,1.,5.,10.)
TASKS=('pick-and-place','push','slide','reach')


def job_spec(plan,task,seed,temperature,phase):
    ident=f'{task}-ibolt-ls5to1-utd1-T{temperature:g}-s{seed}'
    return dict(id=ident,task=task,method='ibolt',seed=seed,temperature=temperature,phase=phase,
        source=plan['source'],source_manifest=plan['source_manifest'],
        output=str(Path(plan['folder'])/'runs'/ident),status='queued',gpu=None)


def choose_temperature(jobs):
    scores=[]
    for temperature in TEMPERATURES:
        seeds=[]
        for seed in (0,1):
            matched=[j for j in jobs if j['temperature']==temperature and j['seed']==seed]
            assert len(matched)==1 and matched[0]['status']=='completed'
            file=Path(matched[0]['output'])/'evaluations.jsonl'
            records=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
            records=[r for r in records if 0<r['steps']<1000000]
            assert len(records)==199 and all(r[r['primary_mode']]['episodes']==10 for r in records)
            values=[r[r['primary_mode']]['success_rate'] for r in records]
            seeds.append(dict(seed=seed,last5_success=sum(values[-5:])/5,auc=sum(values)/len(values),
                late_new_success=sum(r[r['primary_mode']]['new_final_success_count'] for r in records[-5:])))
        scores.append(dict(temperature=temperature,last5_success=sum(s['last5_success'] for s in seeds)/2,
            auc=sum(s['auc'] for s in seeds)/2,new_successes=sum(s['late_new_success'] for s in seeds),seeds=seeds))
    best=max(scores,key=lambda r:(r['last5_success'],r['auc'],-abs(math.log(r['temperature']))))
    return dict(temperature=best['temperature'],scores=scores,selected_unix=time.time(),
        reused_seeds=[0,1],added_seeds=[2,3,4],evidence_of_new_goal_success=best['new_successes']>0,
        rule='Mean of within-seed last five pre-final 10-episode evaluations, then mean over seeds 0/1; ties use AUC then temperature closest to 1.',
        limitation='Seeds 0/1 participated in temperature selection; a best rank does not establish successful learning.')


def process_info(pid):
    try:
        f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=f[0],start_ticks=int(f[19]),rss_mib=int(f[21])*os.sysconf('SC_PAGE_SIZE')//2**20)
    except OSError:return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent;plan=json.loads(path.read_text())
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from fetch_rebuttal.campaign import environment,gpu_memory,memory_available_mib,process_live
    children={};stopping=False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def env(gpu):
        return dict(environment(plan['source'],gpu),JAX_COMPILATION_CACHE_DIR=plan['jax_cache'],MPLCONFIGDIR=str(folder/'mpl-cache'))
    def start(job,smoke=False):
        spec=folder/'specs'/(job['id']+'.json');atomic(spec,job)
        command=['taskset','-c',','.join(map(str,job['cpus'])),plan['python'],'-u','-m',
            'rebuttal_tools.fetch_wide_utd1_worker','--spec',str(spec)]
        if smoke:command+=['--smoke']
        with (folder/'logs'/(job['id']+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=plan['source'],env=env(job['gpu']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='initializing',pid=child.pid,start_ticks=process_info(child.pid)['start_ticks'],started_unix=time.time())
        children[child.pid]=child;print('Started '+job['id']+' GPU'+str(job['gpu']),flush=True)
        return child
    def preflight(index):
        task=TASKS[index];job=dict(id='preflight-'+task,task=task,method='ibolt',seed=100,
            temperature={'pick-and-place':.25,'push':.05}.get(task,.25),source=plan['source'],
            source_manifest=plan['source_manifest'],gpu=index,cpus=plan['cpu_lanes'][index],output=str(folder/'preflights'/task))
        child=start(job,True)
        if child.wait()!=0:raise RuntimeError('GPU preflight failed: '+task)
        result=json.loads((Path(job['output'])/'verified-completion.json').read_text())
        assert result['updates']==4 and result['runtime']['backend']=='gpu'
        job.update(status='completed',result=result);return task,job
    def applications():
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
        result={}
        for row in csv.reader(io.StringIO(text)):
            try:result[int(row[0])]=int(row[1])
            except (ValueError,IndexError):pass
        return result
    try:
        plan.update(controller_pid=os.getpid(),status='validating_gpu');atomic(path,plan)
        if not plan.get('preflights_passed'):
            with ThreadPoolExecutor(max_workers=4) as pool:plan['preflights']=dict(pool.map(preflight,range(4)))
            plan['preflights_passed']=True
            atomic(folder/'gpu-preflight-verification.json',dict(passed=True,cases=plan['preflights']))
        plan['status']='running'
        while True:
            if stopping:raise InterruptedError('Stopped by request')
            for job in plan['jobs']:
                if job['status'] not in ('initializing','running'):continue
                out=Path(job['output'])
                if (out/'verified-completion.json').exists():
                    result=json.loads((out/'verified-completion.json').read_text());assert result['updates']==995000
                    job.update(status='completed',finished_unix=result['finished_unix'],result=result)
                elif (out/'scale-worker-failure.json').exists() or (out/'failure.json').exists() or not process_live(job):
                    fail=out/'scale-worker-failure.json'
                    if not fail.exists():fail=out/'failure.json'
                    reason=json.loads(fail.read_text()).get('error') if fail.exists() else 'Exited without verified completion'
                    job.update(status='failed',error=reason,finished_unix=time.time())
                elif (out/'progress.json').exists():
                    job['progress']=json.loads((out/'progress.json').read_text())
                    if job['progress'].get('updates',0)>0:job['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            for task in ('slide','reach'):
                if task in plan['selections']:continue
                group=[j for j in plan['jobs'] if j['task']==task and j['phase']=='temperature-sweep']
                assert len(group)==14
                if any(j['status']=='failed' for j in group):
                    plan.setdefault('blocked_selections',{})[task]='A search seed failed; no partial-grid selection.'
                elif all(j['status']=='completed' for j in group):
                    selection=choose_temperature(group);plan['selections'][task]=selection
                    atomic(folder/(task+'-temperature-selection.json'),selection)
                    plan['jobs'] += [job_spec(plan,task,seed,selection['temperature'],'selected-5seed') for seed in (2,3,4)]
            memory=gpu_memory();apps=applications();active=[j for j in plan['jobs'] if j['status'] in ('initializing','running')]
            for job in plan['jobs']:
                if job['status']!='queued':continue
                host_deficit=sum(max(0,plan['host_reserve_mib']-(process_info(j.get('pid',0)) or {}).get('rss_mib',0)) for j in active)
                if memory_available_mib()-host_deficit-plan['host_reserve_mib']<plan['host_headroom_mib']:
                    job['waiting_reason']='host memory reservation';continue
                choices=[]
                for gpu,(used,total) in memory.items():
                    own=[j for j in active if j['gpu']==gpu]
                    deficit=sum(max(0,plan['gpu_reserve_mib']-apps.get(j.get('pid'),0)) for j in own)
                    projected=used+deficit+plan['gpu_reserve_mib']
                    if projected<=total-plan['gpu_headroom_mib'] and sum(j['status']=='initializing' for j in own)<plan['max_initializing_per_gpu']:
                        choices.append((projected/total,gpu))
                if not choices:job['waiting_reason']='GPU reservation / initialization admission';continue
                job['gpu']=min(choices)[1]
                job['cpus']=min(plan['cpu_lanes'],key=lambda lane:sum(j.get('cpus')==lane for j in active))
                start(job);job.pop('waiting_reason',None);active.append(job)
            plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])),checked_unix=time.time(),gpu_memory=memory)
            atomic(path,plan)
            if not any(plan['counts'].get(s,0) for s in ('queued','initializing','running')):
                plan['status']='completed_with_failures' if plan['counts'].get('failed') else 'completed';atomic(path,plan);return
            time.sleep(10)
    except BaseException as error:
        for pid,child in children.items():
            if child.poll() is None:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        plan.update(status='stopped' if stopping else 'controller_error',error=repr(error));atomic(path,plan);raise


if __name__=='__main__':main()
