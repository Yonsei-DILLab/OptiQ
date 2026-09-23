"""Separate Vast queue: preserve existing jobs and wait for idle GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import write
from .temperature_sweep import campaign_manifest, CAMPAIGN

ROOT=Path('/workspace/antmaze-temperature-20260924')
SOURCE=Path(__file__).resolve().parents[1]


def register(host):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=SOURCE).strip()
    manifest=campaign_manifest(SOURCE,sha)
    # Alternate settings across GPU generations; longer mazes on 5090s.
    hosts=['vast2','vast3','vast5','vast4']
    shards={h:[] for h in hosts}
    for i,j in enumerate(manifest['jobs']):
        target=(['vast2','vast5'][i//3%2] if j['task']=='v4'
                else ['vast3','vast4'][i//3%2] if j['task']=='v1'
                else hosts[i//3%4])
        shards[target].append(j)
    manifest['jobs']=shards[host];manifest['host']=host
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    for d in ('jobs','logs','preflight','runs'):(ROOT/d).mkdir(exist_ok=True)
    write(ROOT/'status.json',dict(phase='waiting_runtime',pending=[j['id'] for j in manifest['jobs']]))
    print(json.dumps({h:len(j) for h,j in shards.items()}))


def existing_pending():
    for name in ('optiq-nm-20260922','optiq-gmm-trg-20260921'):
        root=Path('/workspace')/name
        if (root/'CANCELLED').exists():continue
        gate=json.loads((root/'gate.json').read_text()).get('stage',0) if (root/'gate.json').exists() else 0
        for p in (root/'state').glob('*.json'):
            j=json.loads(p.read_text())
            if j.get('status')=='queued' and j.get('stage',0)<=gate:return True
    return False


def run():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    live={};pending=list(manifest['jobs']);completed=[];failed=[]
    lease=(ROOT/'queue.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STARTED').exists(),'Do not restart training automatically'
    (ROOT/'STARTED').touch()
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('WANDB_','JAX_','XLA_')):env.pop(k,None)
    env.update(ANTMAZE_PYTHON=str(ROOT/'venv/bin/python'),
               MUJOCO_PY_MUJOCO_PATH=str(ROOT/'mujoco210'),
               HF_HOME=str(ROOT/'cache'),PYTHONPATH='/workspace/optiq-nm-20260922/deps',
               WANDB_MODE='online',OPTIQ_CAMPAIGN=CAMPAIGN,
               PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for p in (Path('/workspace/optiq-ant-highbeta-20260916/private/wandb_api_key'),
              Path('/workspace/optiq-clean-seed3to7-20260909/wandb_api_key')):
        if p.is_file():env['WANDB_API_KEY']=p.read_text().strip();break
    assert env.get('WANDB_API_KEY'),'Missing credential; no jobs started'
    while pending or live:
        for gpu,(p,j,lock) in list(live.items()):
            code=p.poll()
            if code is None:continue
            lock.close();del live[gpu]
            (completed if code==0 else failed).append(j['id'])
        if not failed and not existing_pending():
            for gpu in range(4):
                if gpu in live or not pending:continue
                busy=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                    '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if busy:continue
                lock=(ROOT/f'gpu{gpu}.lock').open('a')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                j=pending.pop(0)
                with (ROOT/'logs'/f"{j['id']}-job.log").open('x') as log:
                    p=subprocess.Popen([str(ROOT/'venv/bin/python'),'-m',
                        'antmaze_experiments.controller','--root',str(ROOT),
                        '--job',j['id'],'--gpu',str(gpu)],cwd=SOURCE,
                        env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),CAMPAIGN_GPU=str(gpu)),
                        stdout=log,stderr=subprocess.STDOUT)
                live[gpu]=(p,j,lock)
        write(ROOT/'status.json',dict(phase='held_failure' if failed else 'active',
             pending=[j['id'] for j in pending],running=[dict(id=j['id'],gpu=g,pid=p.pid)
             for g,(p,j,l) in live.items()],completed=completed,failed=failed,
             source_commit=manifest['source_commit'],time=time.time()))
        if failed and not live:return 1
        time.sleep(10)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','run']);p.add_argument('--host')
    a=p.parse_args()
    if a.mode=='register':register(a.host)
    else:sys.exit(run())
