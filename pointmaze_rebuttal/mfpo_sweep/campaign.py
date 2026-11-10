"""Host-local 3x3 MFPO sweep, independent selection seeds, automatic five-seed stage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from ..guards import assert_work_branch, assert_output_boundary
from ..baselines.run import save_json

COEFFICIENTS=(-1.,-.5,-.25)
TEMPERATURES=(.01,.1,1.)
SWEEP_SEEDS=(5,6)
FINAL_SEEDS=(0,1,2,3,4)
UPDATES=992000
STEPS=1000192


def make_plan(maze,folder,source,python):
    jobs=[]
    for coefficient in COEFFICIENTS:
        for temperature in TEMPERATURES:
            for seed in SWEEP_SEEDS:
                name=f'{maze}-mfpo-H{coefficient:g}-a{temperature:g}-s{seed}'
                jobs.append(dict(id=name,maze=maze,coefficient=coefficient,
                    temperature=temperature,seed=seed,phase='sweep',gpu=len(jobs)%4,
                    status='queued',output=str(folder/'runs'/name)))
    return dict(maze=maze,source=str(source),python=str(python),jobs=jobs,status='created',
        coefficients=list(COEFFICIENTS),initial_temperatures=list(TEMPERATURES),
        selection_seeds=list(SWEEP_SEEDS),final_seeds=list(FINAL_SEEDS),
        fixed=dict(steps=STEPS,warmup=8192,num_envs=256,utd=1,batch_size=256,
            actor_lr=3e-4,critic_lr=3e-4,logp_lr=3e-4,temp_lr=3e-4,
            actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],
            actor_layer_norm=False,critic_layer_norm=False,
            T=2,vel1_samples_num=16,vel2_samples_num=32,div_samples_num=2,
            eval_candidate_num=10,v_min=-1600,v_max=1600,atoms_num=101),
        evaluation=dict(primary='official 10-candidate target-Q selection',
            secondary='single direct policy sample',intermediate_episodes=200,
            final_episodes=500,both_normal_and_obstacle=True),
        selection_rule='Mean official normal success over last3 checkpoints and independent seeds5/6; ties: worst-seed late mean, goal coverage, full-curve success, proximity to default. Require some observed success. Repeat selected setting with seeds0..4.',
        gpu_reserve_mib=3072,gpu_headroom_mib=3072,host_reserve_mib=3072,
        max_initializing_per_gpu=3,created_unix=time.time())


def process(job):
    try:
        s=(Path('/proc')/str(job['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        if s[0]=='Z' or int(s[19])!=job['start_ticks']:return None
        return dict(rss_mib=int(s[21])*os.sysconf('SC_PAGE_SIZE')/1024**2)
    except (FileNotFoundError,KeyError):return None


def resources():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True)
    gpus={}
    for line in text.strip().splitlines():
        i,uuid,used,total=[x.strip() for x in line.split(',')]
        gpus[int(i)]=dict(uuid=uuid,used=int(used),total=int(total))
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    apps={}
    for line in text.strip().splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():apps[int(parts[0])]=int(parts[1])
    mem={line.split(':')[0]:line.split(':')[1] for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(mem['MemAvailable'].split()[0])/1024
    maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum!='max':
        available=min(available,(int(maximum)-int(Path('/sys/fs/cgroup/memory.current').read_text()))/1024**2)
    return gpus,apps,available


def environment(plan,gpu):
    return dict(os.environ,PYTHONPATH=os.pathsep.join(filter(None,(plan['source'],plan.get('extra_pythonpath')))),CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',WANDB_MODE='offline',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        MPLCONFIGDIR=str(Path(plan['source']).parent/'mpl-cache'),
        JAX_COMPILATION_CACHE_DIR=str(Path(plan['source']).parent/'jax-cache'))


def command(plan,job,smoke=False):
    result=[plan['python'],'-u','-m','pointmaze_rebuttal.mfpo_sweep.run',
        '--maze',job['maze'],'--seed',str(job['seed']),
        '--target-entropy-coeff',str(job['coefficient']),
        '--initial-temperature',str(job['temperature']),'--output',job['output']]
    if smoke:result.append('--smoke')
    return result


def source_hashes(source):
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((Path(source)/'pointmaze_rebuttal').rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def validate_source(plan):
    assert_work_branch()
    if source_hashes(Path(plan['source']))!=plan['source_hashes']:
        raise RuntimeError('Frozen MFPO sweep source changed')


def preflight(plan,folder):
    # Test the default and a simultaneous override with real B256 updates.
    specs=[dict(id='default',maze=plan['maze'],seed=5,coefficient=-.5,temperature=.01),
           dict(id='override',maze=plan['maze'],seed=6,coefficient=-1.,temperature=.1)]
    def execute(index):
        job=specs[index];job['gpu']=index;job['output']=str(folder/'preflights'/job['id'])
        result=Path(job['output'])/'completed.json'
        if not result.exists():
            log=folder/'logs'/('preflight-'+job['id']+'.log')
            cmd=['taskset','-c',','.join(map(str,plan['cpu_lanes'][index]))]+command(plan,job,True)
            with log.open('x') as stream:
                subprocess.run(cmd,cwd=plan['source'],env=environment(plan,index),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
        r=json.loads(result.read_text())
        if r['updates']!=4 or r['backend']!='gpu' or not r['completed']:
            raise RuntimeError('GPU MFPO preflight did not complete4 real updates')
        return job['id'],r
    with ThreadPoolExecutor(max_workers=2) as workers:results=dict(workers.map(execute,range(2)))
    save_json(folder/'ready.json',dict(all_gpu_preflights_passed=True,results=results,checked_unix=time.time()))


def score(records):
    if len(records)!=2 or sorted(r['seed'] for r in records)!=list(SWEEP_SEEDS):
        raise ValueError('Selection requires exactly independent seeds5 and6')
    late=[];coverage=[];aucs=[]
    for record in records:
        points=record['evaluations']
        if [p['steps'] for p in points]!=[200192,400128,600064,800000,1000192]:
            raise ValueError('Incomplete or unexpected evaluation checkpoints')
        late.append(statistics.mean(p['official']['success'] for p in points[-3:]))
        coverage.append(statistics.mean(p['official']['reachable_goals'] for p in points[-3:]))
        aucs.append(statistics.mean(p['official']['success'] for p in points))
    return dict(late_success=statistics.mean(late),worst_seed_late_success=min(late),
        late_goal_coverage=statistics.mean(coverage),evaluation_success_mean=statistics.mean(aucs),
        per_seed_late_success=late)


def ranking(candidate):
    return (candidate['late_success'],candidate['worst_seed_late_success'],
        candidate['late_goal_coverage'],candidate['evaluation_success_mean'],
        -abs(candidate['coefficient']+.5),-abs(math.log(candidate['temperature']/.01)))


def promote(plan,folder):
    if 'selection' in plan:return
    jobs=[j for j in plan['jobs'] if j['phase']=='sweep']
    if len(jobs)!=18 or not all(j['status'] in ('completed','failed') for j in jobs):return
    candidates=[];excluded=[]
    for coefficient in COEFFICIENTS:
        for temperature in TEMPERATURES:
            group=[j for j in jobs if j['coefficient']==coefficient and j['temperature']==temperature]
            if not all(j['status']=='completed' for j in group):
                excluded.append(dict(coefficient=coefficient,temperature=temperature,
                    reason='Two complete selection-seed results required',
                    runs=[dict(seed=j['seed'],status=j['status'],error=j.get('error'),output=j['output']) for j in group]))
                continue
            records=[dict(seed=j['seed'],evaluations=[json.loads(line) for line in
                (Path(j['output'])/'evaluations.jsonl').read_text().splitlines()]) for j in group]
            candidates.append(dict(coefficient=coefficient,temperature=temperature,**score(records)))
    if not candidates:
        plan['selection']=dict(status='needs_review_no_valid_pairs',candidates=[],excluded=excluded)
        save_json(folder/'selection.json',plan['selection']);return
    best=max(candidates,key=ranking)
    selection=dict(status='selected' if best['late_success']>0 else 'needs_review_no_success',
        best=best,candidates=candidates,excluded=excluded,selection_seeds=list(SWEEP_SEEDS),
        final_seeds=list(FINAL_SEEDS),selected_unix=time.time())
    plan['selection']=selection;save_json(folder/'selection.json',selection)
    if best['late_success']<=0:return
    for seed in FINAL_SEEDS:
        name=f"{plan['maze']}-mfpo-best-H{best['coefficient']:g}-a{best['temperature']:g}-s{seed}"
        plan['jobs'].append(dict(id=name,maze=plan['maze'],seed=seed,
            coefficient=best['coefficient'],temperature=best['temperature'],phase='selected-five-seed',
            gpu=seed%4,status='queued',output=str(folder/'runs'/name)))
    print('Selected '+json.dumps(best)+'; queued final seeds0..4',flush=True)


def aggregate(plan,folder):
    groups={}
    for j in plan['jobs']:
        if j['status']!='completed':continue
        points=[json.loads(line) for line in (Path(j['output'])/'evaluations.jsonl').read_text().splitlines()]
        final=points[-1]
        key=j['phase']+f"/H{j['coefficient']:g}/a{j['temperature']:g}"
        groups.setdefault(key,[]).append(dict(seed=j['seed'],output=j['output'],
            official=final['official'],single_sample=final['single_sample']))
    result={}
    for key,rows in groups.items():
        bymode={}
        for mode in ('official','single_sample'):
            bymode[mode]={}
            for field in ('success','reachable_goals','normalized_goal_entropy'):
                values=[r[mode][field] for r in rows if r[mode][field] is not None]
                bymode[mode][field]=dict(n=len(values),mean=statistics.mean(values) if values else None,
                    sample_std=statistics.stdev(values) if len(values)>1 else None)
        result[key]=dict(n_seeds=len(rows),statistics=bymode,per_seed=rows)
    save_json(folder/'metrics.json',dict(maze=plan['maze'],groups=result,
        selection=plan.get('selection'),updated_unix=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();path=args.plan.resolve();assert_output_boundary(path)
    folder=path.parent;plan=json.loads(path.read_text())
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate_source(plan)
    for name in ('logs','runs','preflights'):(folder/name).mkdir(exist_ok=True)
    plan.update(status='validating_gpu',controller_pid=os.getpid());save_json(path,plan)
    children={};last_summary=0
    try:
        if not (folder/'ready.json').exists():preflight(plan,folder)
        plan['status']='running'
        while True:
            for j in plan['jobs']:
                if j['status'] not in ('initializing','running'):continue
                out=Path(j['output'])
                if (out/'completed.json').exists():
                    r=json.loads((out/'completed.json').read_text())
                    if r['updates']!=UPDATES or r['backend']!='gpu' or not r['completed']:
                        raise RuntimeError('MFPO completion validation failed: '+j['id'])
                    j.update(status='completed',finished_unix=r['finished_unix'])
                elif (out/'failure.json').exists() or not process(j):
                    j.update(status='failed',error=(out/'failure.json').read_text()
                        if (out/'failure.json').exists() else 'Exited without completed result')
                elif (out/'progress.json').exists():
                    j['progress']=json.loads((out/'progress.json').read_text())
                    if j['progress'].get('updates',0)>0:j['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            promote(plan,folder)
            gpus,apps,available=resources()
            active=[j for j in plan['jobs'] if j['status'] in ('initializing','running')]
            for j in plan['jobs']:
                if j['status']!='queued':continue
                if len(active)>=plan.get('max_active',len(plan['cpu_lanes'])):
                    j['waiting_reason']='campaign concurrency limit';continue
                gpu=gpus[j['gpu']]
                same=[x for x in active if x['gpu']==j['gpu']]
                pending=sum(max(0,plan['gpu_reserve_mib']-apps.get(x['pid'],0)) for x in same)
                future_host=sum(max(0,plan['host_reserve_mib']-(process(x) or {}).get('rss_mib',0)) for x in active)
                if gpu['used']+pending+plan['gpu_reserve_mib']>gpu['total']-plan['gpu_headroom_mib']:
                    j['waiting_reason']='GPU memory';continue
                if available-future_host-plan['host_reserve_mib']<16384:
                    j['waiting_reason']='host memory';continue
                if sum(x['status']=='initializing' for x in same)>=plan['max_initializing_per_gpu']:
                    j['waiting_reason']='compiler startup admission';continue
                free=[lane for lane in plan['cpu_lanes'] if all(x.get('cpus')!=lane for x in active)]
                if not free:j['waiting_reason']='dedicated physical cores';continue
                validate_source(plan);cpu=free[0]
                cmd=['taskset','-c',','.join(map(str,cpu))]+command(plan,j)
                with (folder/'logs'/(j['id']+'.log')).open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=plan['source'],env=environment(plan,j['gpu']),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
                j.update(status='initializing',pid=child.pid,start_ticks=int(stat[19]),
                    cpus=cpu,command=cmd,started_unix=time.time())
                children[child.pid]=child;active.append(j);available-=plan['host_reserve_mib']
                print('Started '+j['id']+' GPU'+str(j['gpu']),flush=True)
                save_json(path,plan)
            counts={s:sum(j['status']==s for j in plan['jobs']) for s in
                    ('queued','initializing','running','completed','failed')}
            plan.update(counts=counts,checked_unix=time.time(),gpu_memory=gpus)
            if time.time()-last_summary>=60:aggregate(plan,folder);last_summary=time.time()
            save_json(path,plan)
            if not counts['queued'] and not counts['initializing'] and not counts['running']:
                final=[j for j in plan['jobs'] if j['phase']=='selected-five-seed']
                if final and len(final)==5 and all(j['status']=='completed' for j in final):
                    plan['status']='completed_with_failed_grid_runs' if counts['failed'] else 'completed'
                else:
                    plan['status']=plan.get('selection',{}).get('status','incomplete_failed_runs')
                    if plan['status']=='selected':plan['status']='incomplete_failed_final_runs'
                aggregate(plan,folder);save_json(path,plan);break
            time.sleep(10)
    except BaseException as error:
        plan.update(status='controller_error',error=repr(error));save_json(path,plan);raise


if __name__=='__main__':main()
