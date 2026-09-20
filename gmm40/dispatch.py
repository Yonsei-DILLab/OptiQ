"""One process per worker; optional four-GPU queues with exclusive ownership."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"gmm40-results"


def publish_status(state, gpu):
    from .evaluation import atomic_json
    if gpu is None:
        atomic_json(RESULTS/'queue_status.json', state)
        return
    state.update(gpu=gpu,controller_pid=os.getpid(),timestamp=time.time())
    with (RESULTS/'queue_status.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        atomic_json(RESULTS/f'queue_status_gpu{gpu}.json',state)
        workers={}
        for index in range(4):
            path=RESULTS/f'queue_status_gpu{index}.json'
            workers[str(index)]=json.loads(path.read_text()) if path.exists() else {'status':'not_started'}
        statuses=[item['status'] for item in workers.values()]
        aggregate='failed' if 'failed' in statuses else 'completed' if all(s=='completed' for s in statuses) else 'running'
        atomic_json(RESULTS/'queue_status.json',dict(status=aggregate,mode='parallel',concurrency=4,workers=workers,timestamp=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,choices=range(4))
    args=parser.parse_args();gpu=args.gpu
    # Shared campaign lock excludes the original exclusive single-worker dispatcher.
    lock=(RESULTS/"queue.lock").open("a")
    fcntl.flock(lock,(fcntl.LOCK_EX if gpu is None else fcntl.LOCK_SH)|fcntl.LOCK_NB)
    if gpu is not None:
        worker_lock=(RESULTS/f'queue_gpu{gpu}.lock').open('a')
        fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        dispatch(gpu)
    except Exception as exc:
        publish_status(dict(status='failed',pid=os.getpid(),error=repr(exc)),gpu)
        raise


def dispatch(gpu):
    while True:
        jobs=json.loads((RESULTS/"queue.json").read_text())["jobs"]
        if gpu is not None:jobs=[job for job in jobs if job.get('gpu',0)==gpu]
        unfinished=[]
        for job in jobs:
            path=RESULTS/job["name"]
            state=json.loads((path/"status.json").read_text()) if (path/"status.json").exists() else {}
            if state.get("status")=="completed": continue
            if state.get("status")=="stopped_early" and job.get("early_stop_authorized",False): continue
            if path.exists():
                raise RuntimeError(f"Existing non-complete run needs review: {path}: {state}")
            unfinished.append(job)
        if not unfinished:
            publish_status(dict(status="completed",time=time.time()),gpu)
            return
        job=unfinished[0]
        if not job.get("ready",False):
            publish_status(dict(status="waiting_for_validated_adapter",next=job["name"],pid=os.getpid()),gpu)
            time.sleep(10)
            continue
        devices=job.get('gpu_ids',[0 if gpu is None else gpu])
        if gpu is None and len(devices)>1:raise ValueError('Multi-GPU jobs require an explicit owner worker')
        if gpu is not None and devices[0]!=gpu:raise ValueError('First device must be the owner worker')
        reservations=[]
        for extra_gpu in devices[1:]:
            guard=(RESULTS/f'queue_gpu{extra_gpu}.lock').open('a')
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            reservations.append(guard)
        snapshot=RESULTS/"sources"/job["name"]
        snapshot.mkdir(parents=True,exist_ok=False)
        shutil.copytree(ROOT/"gmm40",snapshot/"gmm40",ignore=shutil.ignore_patterns("__pycache__"))
        env=os.environ.copy()
        device=0 if gpu is None else gpu
        env.update(GMM40_REPO_ROOT=str(ROOT),PYTHONPATH=str(snapshot)+":"+str(ROOT),CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
        env.pop("JAX_PLATFORMS",None)
        python="/root/.venv-optiq-mujoco/bin/python" if job["method"] in ("optiq","mfpo") else "/root/.venv-gmm40/bin/python"
        first_cpu=192+16*device
        command=["taskset","-c",f"{first_cpu}-{first_cpu+15}",python,"-u","-m","gmm40.run","--method",job["method"],"--name",job["name"],"--steps",str(job["steps"])]
        command+=job.get("args",[])
        (RESULTS/"logs").mkdir(exist_ok=True)
        with (RESULTS/"logs"/(job["name"]+".log")).open("x") as log:
            process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
            publish_status(dict(status="running",name=job["name"],pid=process.pid,controller_pid=os.getpid(),command=command,gpu_ids=devices),gpu)
            print(f"Started {job['name']} pid={process.pid}",flush=True)
            code=process.wait()
        for guard in reservations:guard.close()
        print(f"Exited {job['name']} code={code}",flush=True)
        if code: raise RuntimeError(f"{job['name']} failed; see log")


if __name__=="__main__": main()
