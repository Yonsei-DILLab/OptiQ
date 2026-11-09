"""Supervised, memory-admitted iBOLT CrossQ logstd [-5,-3.5], UTD 1/2 campaign."""
import argparse
from collections import Counter
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

from .mujoco_logstd_worker import TEMPERATURES, atomic_json

# Keep each UTD pair on the same GPU. The extra pair goes to the least-used VRAM.
SLOTS=((2,2),(1,2),(1,0),(2,1),(1,1),(2,0),(1,3),(2,3))
ACTIVE=('initializing','running')


def make_plan(folder,source,python,host,revision,cpus):
    jobs=[]
    for task_index,(task,temperature) in enumerate(TEMPERATURES.items()):
        for seed in range(5):
            owner,gpu=SLOTS[(task_index*5+seed)%len(SLOTS)]
            if owner!=host:continue
            for utd in (1,2):
                ident=f'{task}-ibolt-crossqAC-ls5to3.5-utd{utd}-T{temperature:g}-s{seed}'
                jobs.append(dict(id=ident,task=task,method='ibolt',seed=seed,utd=utd,
                    temperature=temperature,gpu=gpu,output=str(folder/'runs'/ident),
                    source_commit=revision,status='queued',spec=str(folder/'specs'/(ident+'.json'))))
    return dict(host=host,source=str(source),python=str(python),source_commit=revision,
        jobs=jobs,cpu_lanes=cpus,status='prepared',created_unix=time.time(),
        gpu_reserve_mib=4096,gpu_headroom_mib=4096,max_initializing_per_gpu=8,
        host_headroom_mib=16384,host_reserve_mib=dict(humanoid=8192,walker2d=4096,
            hopper=4096,ant=4096,halfcheetah=4096),
        settings=dict(steps=1000000,warmup=5000,batch_size=256,log_std_min=-5.,
            log_std_max=-3.5,initial_log_std=-3.5,critic='CrossQ 1024x2',
            actor='CrossQ actor, 256x2, policy_delay=3',dacer=True,
            primary_evaluation='stochastic_z (fresh z, conditional center only)',
            evaluation_modes=['zero_z','stochastic_z','stochastic_z_sigma'],
            eval_interval=5000,eval_episodes=10,checkpoint_interval=100000),
        failure_policy='Preserve errors and stop only failed jobs; no hidden parameter changes, seed replacement, or automatic retries.')


def process_info(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],start_ticks=int(fields[19]),
            rss_mib=int(fields[21])*os.sysconf('SC_PAGE_SIZE')//2**20)
    except (OSError,ValueError):return None


def live(job):
    p=process_info(job.get('pid',0))
    return bool(p) and p['state']!='Z' and p['start_ticks']==job.get('start_ticks')


def inventory():
    def query(fields):
        text=subprocess.check_output(['nvidia-smi',fields,'--format=csv,noheader,nounits'],text=True,timeout=15)
        return list(csv.reader(io.StringIO(text),skipinitialspace=True))
    gpus={int(i):dict(uuid=uuid,used=int(used),total=int(total),util=int(util))
          for i,uuid,used,total,util in query('--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu')}
    apps={}
    for row in query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory'):
        if len(row)<3:continue
        try:apps[int(row[1])]=int(row[2])
        except ValueError:continue
    return gpus,apps


def available_host_mib():
    info={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    available=info['MemAvailable']//1024
    maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum!='max':
        current=int(Path('/sys/fs/cgroup/memory.current').read_text())
        available=min(available,(int(maximum)-current)//2**20)
    return available


def environment(plan,gpu):
    return dict(os.environ,PYTHONPATH=plan['source'],CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',WANDB_MODE='disabled',
        JAX_COMPILATION_CACHE_DIR=str(Path(plan['source']).parent/'jax-cache'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')


def command(plan,spec,cpus,smoke=False):
    result=['taskset','-c',','.join(map(str,cpus)),plan['python'],'-u','-m',
            'rebuttal_tools.mujoco_logstd_worker','--spec',str(spec)]
    return result+(['--smoke'] if smoke else [])


def inspect_environments(plan,folder):
    destination=folder/'environment-validation.json'
    if destination.exists():return
    code='''import json,pathlib,gymnasium as gym,numpy as np
from rebuttal_tools.mujoco_logstd_worker import TEMPERATURES,overrides
from train import compose_config,validate_config
rows=[]
for task,temp in TEMPERATURES.items():
 for utd in (1,2):
  spec=dict(task=task,temperature=temp,seed=0,utd=utd,id='inspect',output='/tmp/inspect')
  cfg=compose_config(overrides(spec));validate_config(cfg)
  e=gym.make(cfg.env_name);obs,_=e.reset(seed=0);n,r,term,trunc,info=e.step(np.zeros(e.action_space.shape,np.float32))
  assert np.isfinite(obs).all() and np.isfinite(n).all() and np.isfinite(r)
  rows.append(dict(task=task,utd=utd,observation_shape=e.observation_space.shape,action_shape=e.action_space.shape,env_id=cfg.env_name,temperature=temp))
  e.close()
pathlib.Path(DEST).write_text(json.dumps(dict(passed=True,cases=rows),indent=2))
'''
    env=environment(plan,'');env['JAX_PLATFORMS']='cpu'
    with (folder/'logs'/'environment-validation.log').open('w') as stream:
        subprocess.run(['taskset','-c',','.join(map(str,plan['cpu_lanes'][0])),plan['python'],'-c',
            'DEST='+repr(str(destination))+'\n'+code],cwd=plan['source'],env=env,
            stdout=stream,stderr=subprocess.STDOUT,check=True)


def preflight(plan,folder):
    if plan.get('preflight',{}).get('passed'):return
    inspect_environments(plan,folder)
    utd=plan['host'];ident=f'humanoid-preflight-utd{utd}'
    spec=dict(id=ident,task='humanoid',method='ibolt',seed=100,temperature=.5,utd=utd,
        output=str(folder/'preflights'/ident),source_commit=plan['source_commit'])
    path=folder/'specs'/(ident+'.json');atomic_json(path,spec)
    result_path=Path(spec['output'])/'completed.json';peak=0
    if not result_path.exists():
        gpu=2;gpus,_=inventory();baseline=gpus[gpu]['used']
        with (folder/'logs'/(ident+'.log')).open('x') as stream:
            child=subprocess.Popen(command(plan,path,plan['cpu_lanes'][0],True),cwd=plan['source'],
                env=environment(plan,gpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                gpus,apps=inventory();peak=max(peak,apps.get(child.pid,0),gpus[gpu]['used']-baseline)
                time.sleep(2)
            if child.wait()!=0:raise RuntimeError('GPU preflight failed; see '+str(folder/'logs'/(ident+'.log')))
    result=json.loads(result_path.read_text())
    assert result['backend']=='gpu' and result['updates']==6*utd and result['actor_updates']==2*utd
    plan['preflight']=dict(passed=True,result=result,peak_gpu_mib=peak,
        proposed_gpu_reserve_mib=max(2048,math.ceil((peak*1.25+512)/512)*512))
    atomic_json(folder/'preflight-verification.json',plan['preflight'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['source'],text=True).strip()
    assert revision==plan['source_commit']
    plan.update(controller_pid=os.getpid(),status='validating_gpu');atomic_json(path,plan)
    for name in ('logs','runs','preflights','specs'):(folder/name).mkdir(exist_ok=True)
    try:
        preflight(plan,folder)
    except Exception as error:
        plan.update(status='preflight_failed',error=repr(error));atomic_json(path,plan);raise
    if args.preflight_only:
        plan['status']='prepared';atomic_json(path,plan);return
    for job in plan['jobs']:
        spec={k:job[k] for k in ('id','task','method','seed','utd','temperature','output','source_commit')}
        if not Path(job['spec']).exists():atomic_json(job['spec'],spec)
    children={}
    def shutdown(signum,_frame):
        for j in plan['jobs']:
            if j['status'] in ACTIVE and live(j):
                try:os.kill(j['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        plan['status']='stopped';atomic_json(path,plan);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    try:
        plan['status']='running'
        while True:
            for j in plan['jobs']:
                if j['status'] not in ACTIVE:continue
                o=Path(j['output'])
                if (o/'completed.json').exists():
                    r=json.loads((o/'completed.json').read_text());assert r['updates']==995000*j['utd']
                    j.update(status='completed',finished_unix=r['finished_unix'],progress=r)
                elif (o/'failure.json').exists() or not live(j):
                    error=json.loads((o/'failure.json').read_text()) if (o/'failure.json').exists() else {'error':'Exited without completion'}
                    j.update(status='failed',error=error['error'],finished_unix=time.time())
                elif (o/'progress.json').exists():
                    j['progress']=json.loads((o/'progress.json').read_text())
                    if j['progress'].get('updates',0)>0:j['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            gpus,apps=inventory();host_available=available_host_mib()
            active=[j for j in plan['jobs'] if j['status'] in ACTIVE]
            for j in plan['jobs']:
                if j['status']!='queued':continue
                host_available=available_host_mib()
                gpu=j['gpu'];own=[x for x in active if x['gpu']==gpu]
                deficit=sum(max(0,plan['gpu_reserve_mib']-apps.get(x.get('pid'),0)) for x in own)
                host_deficit=sum(max(0,plan['host_reserve_mib'][x['task']]-(process_info(x.get('pid',0)) or {}).get('rss_mib',0)) for x in active)
                if gpus[gpu]['used']+deficit+plan['gpu_reserve_mib']>gpus[gpu]['total']-plan['gpu_headroom_mib']:
                    j['waiting_reason']='GPU memory reservation';continue
                if host_available-host_deficit-plan['host_reserve_mib'][j['task']]<plan['host_headroom_mib']:
                    j['waiting_reason']='host memory reservation';continue
                if sum(x['status']=='initializing' for x in own)>=plan['max_initializing_per_gpu']:
                    j['waiting_reason']='initialization admission';continue
                cpus=min(plan['cpu_lanes'],key=lambda c:sum(x.get('cpus')==c for x in active))
                with (folder/'logs'/(j['id']+'.log')).open('x') as stream:
                    child=subprocess.Popen(command(plan,j['spec'],cpus),cwd=plan['source'],env=environment(plan,gpu),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                info=process_info(child.pid);j.update(status='initializing',pid=child.pid,start_ticks=info['start_ticks'],
                    started_unix=time.time(),cpus=cpus);j.pop('waiting_reason',None)
                children[child.pid]=child;active.append(j)
                print('Started '+j['id']+' GPU'+str(gpu),flush=True)
            plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])),checked_unix=time.time(),gpu_memory=gpus)
            atomic_json(path,plan)
            if not any(plan['counts'].get(s,0) for s in (*ACTIVE,'queued')):
                plan['status']='completed_with_failures' if plan['counts'].get('failed') else 'completed'
                atomic_json(path,plan);break
            time.sleep(10)
    except Exception as error:
        plan.update(status='controller_error',error=repr(error));atomic_json(path,plan);raise


if __name__=='__main__':main()
