"""Four exclusive GPU slots, immediate backfill, no restart on failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .evaluation import atomic_json

PYTHON = "/home/heechan/.venv-optiq-antmaze/bin/python"
WRAPPER = "/home/heechan/OptiQ-ops/run-gpu.sh"
METHODS = ("optiq", "sac", "meow", "sql", "mfpo", "dipo")


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--methods",nargs="+",choices=METHODS)
    a=p.parse_args(); root=a.root
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/"manifest.json").exists():
        raise RuntimeError("Campaign already exists; never auto-resume or duplicate launch")
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    if a.methods and not a.smoke: raise ValueError("Subset selection is preflight-only")
    jobs=[dict(id=f"{method}-s{seed}",method=method,seed=seed,status="pending")
          for seed in ([0] if a.smoke else range(4)) for method in (a.methods or METHODS)]
    manifest=dict(source=str(source),source_commit=sha,smoke=a.smoke,gpus=[0,1,2,3],jobs=jobs,
        project="OptiQ/gmm-trg",group="antmaze-umaze-online-20260921")
    atomic_json(root/"manifest.json",manifest)
    (root/"logs").mkdir(exist_ok=True); (root/"runs").mkdir(exist_ok=True)
    (root/"jobs").mkdir(exist_ok=True)
    live={}; failure=False
    while True:
        for gpu,(proc,job,log) in list(live.items()):
            code=proc.poll()
            if code is None: continue
            log.close(); del live[gpu]
            result=root/"runs"/job["id"]/"result.json"
            if code==0 and result.exists() and json.loads(result.read_text())["completed"]:
                job.update(status="completed",result=json.loads(result.read_text()),finished=time.time())
            else:
                failure=True
                job.update(status="failed",exit_code=code,finished=time.time())
                atomic_json(root/"failure.json",dict(job=job,pending_held=True,live_preserved=True))
            atomic_json(root/"jobs"/(job["id"]+".json"),job)
        if not failure:
            for gpu in range(4):
                if gpu in live: continue
                pending=next((j for j in jobs if j["status"]=="pending"),None)
                if pending is None: break
                # Inspect reservation without taking it away from its owner.
                probe=open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock","a")
                try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    probe.close(); continue
                fcntl.flock(probe,fcntl.LOCK_UN); probe.close()
                job=pending
                cmd=[WRAPPER,str(gpu),"--branch","v5-direct-gmm",PYTHON,"-m","antmaze.run",
                    "--method",job["method"],"--seed",str(job["seed"]),"--output",str(root/"runs"/job["id"])]
                if a.smoke: cmd.append("--smoke")
                env=os.environ.copy();env.update(OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu),
                    PYTHONPATH=str(source),WANDB_PROJECT="gmm-trg",WANDB_ENTITY="OptiQ")
                log=(root/"logs"/(job["id"]+".log")).open("w")
                proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status="running",pid=proc.pid,gpu=gpu,started=time.time(),command=cmd)
                atomic_json(root/"jobs"/(job["id"]+".json"),job)
                live[gpu]=(proc,job,log)
        done=sum(j["status"]=="completed" for j in jobs)
        phase="failed" if failure else ("completed" if done==len(jobs) else "running")
        atomic_json(root/"status.json",dict(phase=phase,source_commit=sha,updated=time.time(),
            completed=done,total=len(jobs),jobs=jobs))
        if not live and (failure or done==len(jobs)):
            if not failure: atomic_json(root/"result.json",dict(completed=True,source_commit=sha,jobs=jobs))
            return
        time.sleep(2)


if __name__=="__main__": main()
