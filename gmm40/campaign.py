"""Durable, restart-aware four-slot queue for the explicitly approved campaign."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except (FileNotFoundError,ProcessLookupError):return None


def prepare(root,plan_path=None):
    source=Path(__file__).resolve().parents[1]
    plan=read(plan_path or source/'gmm40/campaign_plan.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip():
        raise RuntimeError('Commit the source before preparing the campaign')
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        saved=read(manifest_path)
        assert saved['source_commit']==commit and saved['source']==str(source) and saved['plan']==plan
        return saved
    jobs=[]
    for seed in plan['seeds']:
        for method in plan['methods']:
            name=f"{plan.get('job_prefix','')}{method}_s{seed}_100k"
            argv=['--method',method,'--name',name,'--seed',str(seed),'--steps',str(plan['steps']),
                  '--batch',str(plan['batch']),'--temperature',str(plan['temperature']),
                  '--eval-samples',str(plan['eval_samples']),'--width',str(plan['width']),
                  '--depth',str(plan['depth'])]
            if method in ('optiq_trg','optiq'):argv+=['--n',str(plan['optiq_n']),'--m',str(plan['optiq_m'])]
            if method=='optiq_trg' and 'trg_actor' in plan:
                argv+=['--trg-log-std-max',str(plan['trg_actor']['log_std_max']),
                       '--trg-initial-log-std',str(plan['trg_actor']['initial_log_std'])]
                if 'teacher_std_floor' in plan['trg_actor']:
                    argv+=['--trg-teacher-std-floor',str(plan['trg_actor']['teacher_std_floor'])]
            jobs.append(dict(name=name,method=method,seed=seed,args=argv,steps=plan['steps']))
            write(root/'jobs'/f'{name}.json',dict(status='pending',name=name))
    manifest=dict(plan=plan,source=str(source),source_commit=commit,jobs=jobs,created=time.time(),
                  submodules=subprocess.check_output(['git','submodule','status','--recursive'],cwd=source,text=True))
    write(manifest_path,manifest)
    write(root/'results/queue.json',dict(jobs=jobs))
    return manifest


def gate_ready(manifest):
    if manifest['plan'].get('gate') is None:return True
    gate=Path(manifest['plan']['gate'])
    for seed in range(4):
        job=read(gate/'jobs'/f'humanoid-trg-dacer-T0.25-b1-s{seed}.json',{})
        if job.get('status')!='completed':return False
        if job.get('pid') and identity(job['pid']) is not None:return False
    return True


def occupied_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    mapping={}
    for row in csv.reader(output.splitlines(),skipinitialspace=True):
        index,uuid,name=row
        if '5090' not in name:raise RuntimeError('This campaign is restricted to the approved RTX 5090 server')
        mapping[uuid]=int(index)
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {mapping[row[0]] for row in csv.reader(output.splitlines(),skipinitialspace=True) if row and row[0] in mapping}


def preflight(root,manifest):
    from .target import initialize_target
    import numpy as np
    import jax
    import torch
    if not gate_ready(manifest):raise RuntimeError('Humanoid predecessor is not complete')
    if not torch.cuda.is_available() or jax.default_backend()!='gpu':raise RuntimeError('GPU runtime unavailable')
    target=initialize_target();plan=manifest['plan'];checks=[]
    for method in plan['methods']:
        if method=='optiq_trg':
            from .optiq_trg import OptiQTRG
            agent=OptiQTRG(target,n=plan['optiq_n'],m=plan['optiq_m'],batch=plan['batch'],
                           hidden_dims=(plan['width'],)*plan['depth'],temperature=plan['temperature'],
                           **plan.get('trg_actor',{}))
            requested=plan.get('trg_actor',dict(log_std_max=-1.,initial_log_std=-1.))
            assert agent.actor.log_std_min==-5. and agent.actor.log_std_max==requested['log_std_max']
            assert agent.actor.initial_log_std==requested['initial_log_std']
            assert agent.actor.mean_output_init_scale==1.
            assert agent.teacher_std_floor==requested.get('teacher_std_floor',math.exp(-5))
        elif method=='optiq':
            from .optiq import OptiQ
            agent=OptiQ(target,n=plan['optiq_n'],m=plan['optiq_m'],batch=plan['batch'])
        elif method=='mfpo':
            from .mfpo import MFPO
            agent=MFPO(target,0,plan['batch'])
        elif method=='sql':
            from .sql import SQL
            agent=SQL(target,0,plan['batch'])
        else:
            from .torch_agents import make_agent
            agent=make_agent(method,target,0,plan['batch'])
        info=agent.advance(2)
        samples,_,_=agent.evaluate_samples(128,937)
        assert agent.updates==2 and samples.shape==(128,2)
        assert np.isfinite(samples).all() and all(np.isfinite(v) for v in info.values()),(method,info)
        assert np.max(np.abs(samples))<=40.0001
        from .model_sizes import fixed_sizes
        dest=root/'preflight'/method;dest.mkdir(parents=True,exist_ok=True)
        agent.save(dest/'checkpoint.bin')
        sizes=fixed_sizes(dest,method,agent)
        checks.append(dict(method=method,updates=2,metrics=info,model_sizes=sizes,
                           trg_actor=plan.get('trg_actor') if method=='optiq_trg' else None))
        print(json.dumps(checks[-1]),flush=True)
        del agent;gc.collect();jax.clear_caches();torch.cuda.empty_cache()
    write(root/'preflight.json',dict(status='passed',source_commit=manifest['source_commit'],checks=checks,
          torch=torch.__version__,jax=jax.__version__,devices=[str(d) for d in jax.devices()],time=time.time()))


def verified(root,job):
    folder=root/'results'/job['name']
    status=read(folder/'status.json',{})
    audit=read(folder/'update_count_audit.json',{})
    latest=read(folder/'latest.json',{})
    return status.get('status')=='completed' and status.get('step')==job['steps'] and \
        audit.get('status')=='passed' and audit.get('actor_updates')==job['steps'] and latest.get('step')==job['steps']


def dispatch(root,manifest):
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=read(root/'preflight.json',{})
    if proof.get('status')!='passed' or proof.get('source_commit')!=manifest['source_commit']:
        raise RuntimeError('All planned GPU adapters must pass preflight at the frozen commit')
    processes={};logs={};jobs=manifest['jobs'];plan=manifest['plan']
    while True:
        states={j['name']:read(root/'jobs'/f"{j['name']}.json") for j in jobs}
        for job in jobs:
            state=states[job['name']]
            if state['status'] not in ('running','starting'):continue
            pid=state.get('pid')
            if pid and identity(pid)==state.get('proc_start') and identity(pid) is not None:continue
            if state['status']=='starting' and not pid:
                # A crash between spawn and PID publication must never duplicate work.
                import psutil
                matches=[]
                for process in psutil.process_iter(['pid','cmdline']):
                    cmd=process.info['cmdline'] or []
                    if 'gmm40.campaign_job' in cmd and job['name'] in cmd:matches.append(process.pid)
                if len(matches)==1:
                    state.update(status='running',pid=matches[0],proc_start=identity(matches[0]))
                    write(root/'jobs'/f"{job['name']}.json",state);continue
                if not matches and not (root/'results'/job['name']).exists() and not (root/'logs'/f"{job['name']}.log").exists():
                    state['status']='pending';write(root/'jobs'/f"{job['name']}.json",state);continue
            process=processes.pop(job['name'],None)
            code=process.wait() if process else None
            if job['name'] in logs:logs.pop(job['name']).close()
            good=verified(root,job) and code in (0,None)
            state.update(status='completed' if good else 'failed',exit_code=code,finished=time.time())
            write(root/'jobs'/f"{job['name']}.json",state)
            if not good:write(root/'failure.json',dict(job=job,state=state,log=str(root/'logs'/f"{job['name']}.log")))
        running=[s for s in states.values() if s['status']=='running']
        pending=[j for j in jobs if states[j['name']]['status']=='pending']
        failed=[s['name'] for s in states.values() if s['status']=='failed']
        completed=[s['name'] for s in states.values() if s['status']=='completed']
        phase='failed' if failed else 'running' if running else 'waiting'
        if len(completed)==len(jobs):
            write(root/'status.json',dict(phase='aggregating',completed=completed,running=[],queued=[],updated=time.time()))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
            subprocess.run([sys.executable,'-m','gmm40.campaign_report','--root',str(root)],env=env,check=True)
            write(root/'status.json',dict(phase='completed',completed=completed,running=[],queued=[],updated=time.time()))
            return
        if not failed and gate_ready(manifest):
            occupied=occupied_gpus()|{s['gpu'] for s in running}
            for gpu in plan['gpus']:
                if gpu in occupied or not pending:continue
                job=pending.pop(0);name=job['name']
                if (root/'results'/name).exists():raise RuntimeError(f'Unowned existing result: {name}')
                cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-gmm40',
                     sys.executable,'-u','-m','gmm40.campaign_job',*job['args']]
                env=os.environ.copy();env.update(OPTIQ_SOURCE_DIR=manifest['source'],GMM40_REPO_ROOT=manifest['source'],
                    GMM40_RESULTS_ROOT=str(root/'results'),GMM40_SOURCE_COMMIT=manifest['source_commit'],
                    GMM40_CAMPAIGN=plan['name'],GMM40_WANDB_DIR=str(root/'wandb'),CAMPAIGN_GPU=str(gpu),
                    PYTHONDONTWRITEBYTECODE='1')
                for d in ('logs','wandb'):(root/d).mkdir(exist_ok=True)
                state=dict(status='starting',name=name,gpu=gpu,command=cmd,started=time.time())
                write(root/'jobs'/f'{name}.json',state)
                log=(root/'logs'/f'{name}.log').open('x')
                process=subprocess.Popen(cmd,cwd=manifest['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
                processes[name]=process;logs[name]=log
                state.update(status='running',pid=process.pid,proc_start=identity(process.pid))
                write(root/'jobs'/f'{name}.json',state)
                running.append(state)
                print(f'Started {name} on GPU {gpu}, PID {process.pid}',flush=True)
            phase='running' if running else 'waiting'
        write(root/'status.json',dict(phase=phase,completed=completed,failed=failed,running=running,
              queued=[j['name'] for j in pending],updated=time.time(),source_commit=manifest['source_commit']))
        if failed and not running:return
        time.sleep(5)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','preflight','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,help='Committed plan for prepare; otherwise uses campaign_plan.json')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=prepare(root,args.plan) if args.action=='prepare' else read(root/'manifest.json')
    if not manifest:raise RuntimeError('Prepare the committed campaign first')
    os.environ.update(GMM40_REPO_ROOT=manifest['source'],GMM40_RESULTS_ROOT=str(root/'results'))
    if args.action=='preflight':preflight(root,manifest)
    elif args.action=='run':
        try:dispatch(root,manifest)
        except Exception as exc:
            write(root/'failure.json',dict(stage='controller',error=repr(exc),time=time.time()))
            raise
    else:print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
