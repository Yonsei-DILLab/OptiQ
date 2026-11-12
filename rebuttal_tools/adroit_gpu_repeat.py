"""Exploratory GPU-crossed reruns; retain all trials and instrument unchanged training."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time


def atomic(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def worker(spec_path):
    import importlib.metadata
    import numpy as np
    import jax
    from exploration import ExplorationIBOLT
    from adroit_rebuttal.run import run, guard
    spec=json.loads(spec_path.read_text());guard()
    original_init,original_train=ExplorationIBOLT.__init__,ExplorationIBOLT.train
    output=Path(spec['output'])

    def tree_hash(tree):
        leaves,structure=jax.tree_util.tree_flatten(tree);h=hashlib.sha256(str(structure).encode())
        for leaf in leaves:
            a=np.ascontiguousarray(leaf);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.hexdigest()

    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        # Pure reads: no RNG draws, optimizer calls, or parameter changes.
        atomic(output/'initial-state-identifiers.json',dict(seed=spec['seed'],gpu=spec['gpu'],
            gpu_uuid=spec['gpu_uuid'],hostname=socket.gethostname(),actor_params_sha256=tree_hash(self.policy.actor_state.params),
            critic_params_sha256=tree_hash(self.policy.qf_state.params),actor_optimizer_sha256=tree_hash(self.policy.actor_state.opt_state),
            critic_optimizer_sha256=tree_hash(self.policy.qf_state.opt_state),updates=int(self._n_updates),
            cpu_affinity=sorted(os.sched_getaffinity(0)),jax_devices=[str(d) for d in jax.devices()],
            versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','torch','numpy','mujoco')},
            instrumentation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime_flags={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','XLA_FLAGS','JAX_DEFAULT_MATMUL_PRECISION','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE')}))

    def train(self,*args,**kwargs):
        result=original_train(self,*args,**kwargs)
        if self._n_updates==32 or self.num_timesteps%10240==0 or self.num_timesteps>=5000192:
            metrics={k:float(v) for k,v in self.logger.name_to_value.items()
                     if k.startswith(('train/','exploration/')) and isinstance(v,(int,float,np.number)) and np.isfinite(v)}
            row=dict(steps=int(self.num_timesteps),updates=int(self._n_updates),observed_unix=time.time(),metrics=metrics,
                     discovery=self.replay_buffer.discovery())
            with (output/'training-metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        return result

    ExplorationIBOLT.__init__,ExplorationIBOLT.train=init,train
    run(spec,smoke=False)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False


def summaries(plan,root):
    all_rows=[];candidates={};hash_consistency={}
    for seed in (0,2,3,4):
        jobs=[j for j in plan['jobs'] if j['seed']==seed];states=[];complete=[]
        for j in jobs:
            row={k:j.get(k) for k in ('id','seed','gpu','gpu_uuid','status')}
            initial=Path(j['output'])/'initial-state-identifiers.json'
            if initial.exists():states.append(json.loads(initial.read_text()))
            if j['status']=='completed':
                lines=(Path(j['output'])/'evaluations.jsonl').read_text().splitlines()
                final=json.loads(lines[-1]);assert final['steps']==5000192
                row.update(final_zero_z=final['zero_z']['success_rate'],final_stochastic_z=final['stochastic_z']['success_rate'],final_full_policy=final['stochastic_z_sigma']['success_rate'])
                complete.append(row)
            all_rows.append(row)
        hash_consistency[str(seed)]=dict(initializations_recorded=len(states),all_four_recorded=len(states)==4,
            actor_identical=len({s['actor_params_sha256'] for s in states})<=1,
            critic_identical=len({s['critic_params_sha256'] for s in states})<=1)
        if len(complete)==4:
            candidates[str(seed)]=max(complete,key=lambda x:x['final_zero_z'])
    atomic(root/'initialization-consistency.json',hash_consistency)
    atomic(root/'all-trial-results.json',all_rows)
    atomic(root/'selected-candidates.json',dict(selection='Exploratory best final100 zero_z result among four GPU reruns per fixed seed; NOT an unbiased five-seed estimate',
        original_results_preserved=True,all_trials_preserved=True,candidates=candidates))


def campaign(path):
    root=path.parent;lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text());children={};stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    assert len(plan['jobs'])==16 and not any(j['status']=='running' for j in plan['jobs']), 'No automatic resume/retry'
    try:
        while not stopping:
            for pid,child in list(children.items()):
                if child.poll() is None:continue
                j=next(j for j in plan['jobs'] if j.get('pid')==pid);marker=Path(j['output'])/'completed.json'
                if child.returncode==0 and marker.exists():
                    result=json.loads(marker.read_text());assert result['completed'] and result['steps']==5000192 and result['updates']==623424
                    proof=json.loads((Path(j['output'])/'demonstration-validation.json').read_text())
                    assert proof['passed'] and proof['manifest_sha256']==j['demonstration_manifest_sha256']
                    j.update(status='completed',result=result)
                else:j.update(status='failed',error=f'exit={child.returncode}; inspect preserved log and failure.json')
                child.wait();del children[pid]
            active=[j for j in plan['jobs'] if j['status']=='running']
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            gpus={}
            for line in raw.splitlines():
                i,uuid,used,total,util=[x.strip() for x in line.split(',')];gpus[int(i)]=dict(uuid=uuid,used=int(used),total=int(total),util=int(util))
            topology={}
            for cpu in sorted(os.sched_getaffinity(0)):
                p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');key=(p.joinpath('physical_package_id').read_text(),p.joinpath('core_id').read_text())
                topology.setdefault(key,[]).append(cpu)
            reserved=set(c for group in list(topology.values())[:4] for c in group)
            for p in Path('/proc').iterdir():
                if not p.name.isdigit() or not alive(int(p.name)):continue
                try:cpus=os.sched_getaffinity(int(p.name))
                except (ProcessLookupError,PermissionError):continue
                if len(cpus)<=16:reserved.update(cpus)
            free=[group for group in topology.values() if not set(group)&reserved]
            initialized=Counter()
            for j in active:
                p=Path(j['output'])/'progress.json';v=json.loads(p.read_text()) if p.exists() else {}
                if v.get('updates',0)==0:initialized[j['gpu']]+=1
            for j in plan['jobs']:
                if j['status']!='queued':continue
                gpu=j['gpu'];info=gpus[gpu]
                if initialized[gpu]>=2 or info['total']-info['used']<6144 or len(free)<2:continue
                if shutil.disk_usage(root).free<15*1024**3:raise RuntimeError('Insufficient disk headroom')
                assert info['uuid']==j['gpu_uuid']
                cpus=free.pop(0)+free.pop(0)
                assert not Path(j['output']).exists(),'Refusing output overwrite'
                spec=root/'specs'/(j['id']+'.json');atomic(spec,j)
                env=dict(os.environ,PYTHONPATH=plan['source'],CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',WANDB_MODE='disabled',JAX_COMPILATION_CACHE_DIR=str(root/'jax-cache'),MPLCONFIGDIR=str(root/'mpl-cache'))
                command=['taskset','-c',','.join(map(str,cpus)),plan['python'],'-u',str(Path(__file__).resolve()),'--worker','--spec',str(spec)]
                with (root/'logs'/(j['id']+'.log')).open('x') as stream:child=subprocess.Popen(command,cwd=plan['source'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=child.pid,cpu_affinity=cpus,started_unix=time.time());children[child.pid]=child
                initialized[gpu]+=1;info['used']+=2048
                print('Started '+j['id']+' GPU'+str(gpu),flush=True)
            counts=dict(Counter(j['status'] for j in plan['jobs']))
            plan.update(status='running',counts=counts,observed_unix=time.time());atomic(path,plan);summaries(plan,root)
            with (root/'gpu-load.jsonl').open('a') as stream:stream.write(json.dumps(dict(observed_unix=time.time(),gpu=gpus,active_jobs=[dict(id=j['id'],gpu=j['gpu']) for j in active]))+'\n')
            if all(j['status'] in ('completed','failed') for j in plan['jobs']):
                plan['status']='completed_with_failures' if counts.get('failed') else 'completed';atomic(path,plan);return
            time.sleep(10)
    finally:
        for pid,child in children.items():
            if child.poll() is None:
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args()
    if a.worker:worker(a.spec)
    else:campaign(a.plan)
