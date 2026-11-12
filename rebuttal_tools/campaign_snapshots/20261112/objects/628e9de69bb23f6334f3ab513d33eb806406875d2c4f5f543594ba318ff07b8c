"""Equal-budget Fetch entropy grids with independent pilot and final seeds."""
import argparse
from collections import Counter,deque
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

GRIDS={
    'ibolt':('temperature',[.1,.25,.5],.25),
    'sac':('target_entropy',[-4.,-2.,0.],-4.),
    'mfpo':('target_entropy_coeff',[-.5,-.25,0.],0.),
    'meow':('alpha',[.05,.2,.8],.2),
}
TASKS=('slide','reach')
PILOT_SEEDS=(5,6)
FINAL_SEEDS=(0,1,2,3,4)
STEPS=5000192
UPDATES=623424


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19]) if fields[0]!='Z' else None
    except OSError:return None


def job(plan,method,task,value,seed,phase):
    index=GRIDS[method][1].index(value)
    name=f'{task}-{method}-g{index}-s{seed}-{phase}'
    return dict(id=name,method=method,task=task,parameter=GRIDS[method][0],value=value,
        seed=seed,phase=phase,status='queued',source=plan['source'],
        source_manifest=plan['source_manifest'],python=plan['torch_python'] if method in ('sac','meow') else plan['jax_python'],
        temperature=value if method=='ibolt' else .25,
        mfpo_target_entropy_coeff=value if method=='mfpo' else 0.,mfpo_initial_temperature=1.,
        ibolt_reference_config=plan['ibolt_reference_config'],
        output=str(Path(plan['folder'])/('preflights' if phase=='preflight' else 'runs')/name))


def prepare_jobs(plan):
    plan['jobs']=[job(plan,m,t,value,s,'pilot') for s in PILOT_SEEDS
        for t in TASKS for value_index in range(3) for m in plan['methods']
        for value in [GRIDS[m][1][value_index]]]
    plan['preflights']=[job(plan,m,t,GRIDS[m][2],100,'preflight')
        for t in TASKS for m in plan['methods']]


def score_pair(rows):
    assert sorted(r['seed'] for r in rows)==list(PILOT_SEEDS)
    assert all(r['complete'] and r['steps']==STEPS and r['updates']==UPDATES for r in rows)
    late=[statistics.mean(r['regular'][-5:]) for r in rows]
    assert all(len(r['regular'])>=5 and r['final_episodes']==100 for r in rows)
    return dict(late5=statistics.mean(late),final100=statistics.mean(r['final100'] for r in rows),
        auc=statistics.mean(statistics.mean(r['regular']) for r in rows),
        worst_late5=min(late),late5_seeds=late)


def ranking(candidate,method):
    grid=GRIDS[method][1];default=GRIDS[method][2]
    return (candidate['late5'],candidate['final100'],candidate['auc'],candidate['worst_late5'],
        -abs(grid.index(candidate['value'])-grid.index(default)),-grid.index(candidate['value']))


def pilot_record(j):
    p=Path(j['output']);done=json.loads((p/'verified-completion.json').read_text())
    records=[json.loads(s) for s in (p/'evaluations.jsonl').read_text().splitlines() if s.strip()]
    regular=[r[r['primary_mode']]['success_rate'] for r in records
        if r['steps']>0 and r[r['primary_mode']]['episodes']==10]
    final=next(r[r['primary_mode']] for r in records if r['steps']==STEPS)
    return dict(seed=j['seed'],complete=done['completed'],steps=done['steps'],updates=done['updates'],
        regular=regular,final100=final['success_rate'],final_episodes=final['episodes'])


def promote(plan):
    for method in plan['methods']:
        for task in TASKS:
            key=f'{task}/{method}'
            if key in plan['selections']:continue
            pilots=[j for j in plan['jobs'] if j['phase']=='pilot' and j['method']==method and j['task']==task]
            assert len(pilots)==6
            if any(j['status'] not in ('completed','failed') for j in pilots):continue
            candidates=[];excluded=[]
            for value in GRIDS[method][1]:
                pair=[j for j in pilots if j['value']==value]
                if not all(j['status']=='completed' for j in pair):
                    excluded.append(dict(value=value,runs=[dict(id=j['id'],status=j['status'],error=j.get('error')) for j in pair]));continue
                candidates.append(dict(value=value,**score_pair([pilot_record(j) for j in pair])))
            if not candidates:
                plan['selections'][key]=dict(status='no_valid_pair',excluded=excluded);continue
            best=max(candidates,key=lambda r:ranking(r,method))
            plan['selections'][key]=dict(status='selected' if best['late5']>0 else 'selected_unvalidated_all_zero',
                best=best,candidates=candidates,excluded=excluded,pilot_seeds=list(PILOT_SEEDS),
                final_seeds=list(FINAL_SEEDS),selected_unix=time.time(),fresh_final_initialization=True)
            plan['jobs'].extend(job(plan,method,task,best['value'],seed,'final') for seed in FINAL_SEEDS)
            print('Selected '+key+' '+json.dumps(best)+'; fresh seeds0..4 queued',flush=True)
    atomic(Path(plan['folder'])/'selections.json',plan['selections'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent;plan=json.loads(path.read_text())
    assert plan['host'] in (2,3,4)
    assert subprocess.check_output(['git','-C',plan['source'],'branch','--show-current'],text=True).strip()=='heechan-rebuttal'
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=Path('/workspace/.ibolt-capacity-admission.lock').open('a')
    children={};stopping=False;history={i:deque(maxlen=3) for i in range(4)}
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    topology={}
    for cpu in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology.setdefault((p.joinpath('physical_package_id').read_text(),p.joinpath('core_id').read_text()),[]).append(cpu)
    physical={c:k for k,cpus in topology.items() for c in cpus}
    def update():
        for j in plan['preflights']+plan['jobs']:
            if j['status'] not in ('initializing','running'):continue
            out=Path(j['output']);marker=out/'verified-completion.json'
            if marker.exists():
                r=json.loads(marker.read_text());expected=64 if j['phase']=='preflight' else UPDATES
                assert r['completed'] and r['updates']==expected and r['observed_utd']==.125
                assert r['entropy_search']['value']==j['value']
                assert r['entropy_search']['parameter']==j['parameter']
                j.update(status='completed',result=r,finished_unix=time.time())
            elif (out/'failure.json').exists() or identity(j['pid'])!=j['start_ticks']:
                fail=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else dict(error='Exited without verified completion')
                j.update(status='failed',error=fail['error'],finished_unix=time.time())
            elif (out/'progress.json').exists():
                j['progress']=json.loads((out/'progress.json').read_text())
                if j['progress'].get('updates',0)>0:j['status']='running'
        for pid,child in list(children.items()):
            if child.poll() is not None:child.wait();del children[pid]
    def resources():
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
        gpus={int(i):dict(uuid=u.strip(),free=int(m),util=int(v),pids=set()) for i,u,m,v in csv.reader(io.StringIO(raw))}
        uuids={v['uuid']:i for i,v in gpus.items()}
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
        pids=set()
        for pid,uuid in csv.reader(io.StringIO(raw)):
            if pid.strip().isdigit() and uuid.strip() in uuids:
                pid=int(pid);gpus[uuids[uuid.strip()]]['pids'].add(pid);pids.add(pid)
        reserved={physical[c] for cpus in list(topology.values())[:4] for c in cpus}
        for file in [str(path),*plan.get('peer_plans',[])]:
            other=plan if file==str(path) else json.loads(Path(file).read_text())
            rows=[]
            for key in ('jobs','preflights'):
                value=other.get(key,[])
                if isinstance(value,list):rows.extend(value)
            for j in rows:
                if not isinstance(j,dict) or j.get('status') not in ('running','initializing') or not j.get('pid'):continue
                if identity(j['pid'])!=j.get('start_ticks'):continue
                pids.add(j['pid']);gpu=j.get('gpu')
                if gpu in gpus:gpus[gpu]['pids'].add(j['pid'])
                cpus=j.get('learner_cpus',j.get('cpus',[]))+[c for cs in j.get('worker_cpus',[]) for c in cs]
                reserved.update(physical[c] for c in cpus if c in physical)
        for file in plan.get('fixed_reserve_plans',[]):
            other=json.loads(Path(file).read_text())
            for j in other.get('jobs',[]):
                reserved.update(physical[c] for c in j.get('learner_cpus',[]) if c in physical)
        for pid in list(pids):
            try:pids.update(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
            except OSError:pass
        for pid in pids:
            try:cpus=os.sched_getaffinity(pid)
            except ProcessLookupError:continue
            if len(cpus)<=16:reserved.update(physical[c] for c in cpus if c in physical)
        for i,g in gpus.items():history[i].append(g['util']);g['mean_util']=statistics.mean(history[i]);g['active_processes']=len(g.pop('pids'))
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        available=(int(maximum)-int(Path('/sys/fs/cgroup/memory.current').read_text()))/1024**2 if maximum!='max' else 1e12
        return gpus,[cs for k,cs in topology.items() if k not in reserved],available
    def spawn(j,gpu,lanes):
        j.update(gpu=gpu,learner_cpus=lanes[0],worker_cpus=lanes[1:])
        spec=folder/'specs'/(j['id']+'.json');atomic(spec,j)
        env=dict(os.environ,PYTHONPATH=plan['source']+':'+plan.get('extra_pythonpath',''),CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            JAX_COMPILATION_CACHE_DIR=str(folder/'jax-cache'),MPLCONFIGDIR=str(folder/'mpl-cache'),MUJOCO_GL='egl',WANDB_MODE='disabled')
        cmd=['taskset','-c',','.join(map(str,j['learner_cpus'])),j['python'],'-u','-m','rebuttal_tools.fetch_entropy_worker','--spec',str(spec)]
        if j['phase']=='preflight':cmd.append('--smoke')
        with (folder/'logs'/(j['id']+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=plan['source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        j.update(status='initializing',pid=child.pid,start_ticks=identity(child.pid),started_unix=time.time());children[child.pid]=child
        print('Started '+j['id']+' GPU'+str(gpu),flush=True)
    plan.update(controller_pid=os.getpid(),status='validating_gpu');atomic(path,plan)
    try:
        while not stopping:
            update();promote(plan)
            if any(j['status']=='failed' for j in plan['preflights']):raise RuntimeError('GPU preflight failed; pilot cohort remains gated')
            ready=all(j['status']=='completed' for j in plan['preflights'])
            active=[j for j in plan['jobs']+plan['preflights'] if j['status'] in ('running','initializing')]
            with_admission=False
            try:
                fcntl.flock(admission,fcntl.LOCK_EX);with_admission=True
                gpus,cpus,available=resources()
                for j in plan['jobs'] if ready else plan['preflights']:
                    if j['status']!='queued':continue
                    if len(active)>=plan['max_active'] or len(cpus)<3 or available<12288:break
                    initializing=Counter(x['gpu'] for x in active if x['status']=='initializing')
                    eligible=[i for i,g in gpus.items() if len(history[i])==3 and g['free']>=6144 and g['mean_util']<=95 and g['active_processes']<10 and initializing[i]<2]
                    if not eligible:break
                    gpu=min(eligible,key=lambda i:(gpus[i]['mean_util'],gpus[i]['active_processes'],-gpus[i]['free']))
                    spawn(j,gpu,cpus[:3]);cpus=cpus[3:];active.append(j);gpus[gpu]['active_processes']+=1;gpus[gpu]['free']-=3072;available-=4096
                    atomic(path,plan)
                plan.update(status='running' if ready else 'validating_gpu_or_waiting_capacity',checked_unix=time.time(),
                    counts=dict(Counter(j['status'] for j in plan['jobs'])),admission=dict(gpus=gpus,unused_physical_cores=len(cpus),available_ram_mib=available))
                atomic(path,plan)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
                plan['resource_observation_error']=repr(error);atomic(path,plan)
            finally:
                if with_admission:fcntl.flock(admission,fcntl.LOCK_UN)
            if ready and all(j['status'] in ('completed','failed') for j in plan['jobs']):
                plan['status']='completed' if all(j['status']=='completed' for j in plan['jobs']) else 'completed_with_failures';atomic(path,plan);return
            time.sleep(10)
    except BaseException as error:
        plan.update(status='controller_error',error=repr(error));atomic(path,plan);raise
    finally:
        if stopping:
            for j in plan['jobs']+plan['preflights']:
                if j.get('pid') and identity(j['pid'])==j.get('start_ticks'):
                    try:os.kill(j['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
            plan['status']='stopped';atomic(path,plan)


if __name__=='__main__':main()
