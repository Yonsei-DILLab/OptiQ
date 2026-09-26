"""Prepare a bounded local-host queue and supervisor configs; does not start it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .run_nway_job import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',required=True,choices=['vast1','vast2','vast3','vast4'])
    p.add_argument('--commit',required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    jobs=[]
    for task in ['pm_hard','pm_medium','pm_simple']:
        for n,m in [(n,64) for n in [1,4,16,128,256]]+[(64,m) for m in [1,4,16,128,256]]:
            host=('vast2' if max(n,m)>64 else 'vast1' if 16 in (n,m)
                  else 'vast3' if n<64 else 'vast4')
            jobs.append(dict(name=f'{task}-N{n}-M{m}-s0',task=task,method='optiq',
                             temperature=5.,N=n,M=m,host=host,state='pending'))
    a.root.mkdir(parents=True,exist_ok=False)
    for d in ['runs','preflights','proofs','logs']:(a.root/d).mkdir()
    atomic_json(a.root/'plan.json',dict(jobs=jobs,source_commit=a.commit,source=str(source)))
    atomic_json(a.root/'queue.json',dict(state='running',jobs=[j for j in jobs if j['host']==a.host]))
    gpus=[0,1,3] if a.host=='vast3' else [0,1,2,3]
    mujoco = '/home/heechan/.mujoco/mujoco210' if a.host=='vast1' else '/workspace/antmaze-temperature-20260924/mujoco210'
    for gpu in gpus:
        name=f'pointmaze-nm-20260926-gpu{gpu}'
        config=f'''[program:{name}]
directory={source}
command={sys.executable} -u -m maze_benchmarks.nm_worker --root {a.root} --commit {a.commit} --gpu {gpu}
environment=CUDA_VISIBLE_DEVICES="{gpu}",XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MUJOCO_GL="egl",MPLBACKEND="Agg",PYTHONPATH="{source}",MUJOCO_PY_MUJOCO_PATH="{mujoco}",LD_LIBRARY_PATH="{mujoco}/bin:/usr/lib/nvidia"
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
startsecs=0
stdout_logfile={a.root}/logs/worker-gpu{gpu}.log
redirect_stderr=true
'''
        dest=Path('/etc/supervisor/conf.d')/(name+'.conf')
        if dest.exists():raise FileExistsError(dest)
        dest.write_text(config)
    print(json.dumps(dict(host=a.host,jobs=len([j for j in jobs if j['host']==a.host]),gpus=gpus)))


if __name__=='__main__':main()
