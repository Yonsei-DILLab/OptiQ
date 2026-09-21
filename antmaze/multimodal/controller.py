"""Four exclusive GPU slots, immediate backfill, no restart on failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from antmaze.evaluation import atomic_json

PYTHON = "/home/heechan/.venv-optiq-antmaze/bin/python"
WRAPPER = "/home/heechan/OptiQ-ops/run-gpu.sh"
METHODS = ("optiq", "sac", "meow", "sql", "mfpo", "dipo")


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--methods",nargs="+",choices=METHODS)
    p.add_argument("--tasks",nargs="+",choices=["v1","v3","v4"],default=["v1","v4"])
    p.add_argument("--profile",choices=["100k","1m","noveld","sac-dense"],default="100k")
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3])
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--shard-count",type=int,default=1)
    a=p.parse_args(); root=a.root
    if not 0<=a.shard_index<a.shard_count:raise ValueError("Invalid shard")
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/"manifest.json").exists():
        raise RuntimeError("Campaign already exists; never auto-resume or duplicate launch")
    source=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    if a.methods and not a.smoke and a.profile=="100k": raise ValueError("Subset selection is preflight-only for the legacy profile")
    methods=tuple(a.methods or (("optiq","sac","mfpo","meow") if a.profile in ("1m","noveld") else METHODS))
    if a.profile in ("1m","noveld") and set(methods)!={"optiq","sac","mfpo","meow"}:
        raise ValueError("This campaign contains only the four user-selected methods")
    seeds=[0] if a.smoke else a.seeds
    if len(set(seeds))!=len(seeds) or not set(seeds)<=set(range(4)):raise ValueError("Invalid seed shard")
    budgets={task:(1000000 if a.profile=="1m" else 100000) for task in a.tasks}
    campaign="antmaze-multimodal-1m-20260921" if a.profile=="1m" else "antmaze-multimodal-100k-20260921"
    if a.profile=="noveld":
        assert a.tasks==["v1"] and seeds==[0] and a.shard_count==1
        campaign="antmaze-v1-noveld-100k-s0-20260921"
    if a.profile=="sac-dense":
        assert methods==("sac",) and a.tasks==["v1"] and seeds==[0] and a.shard_count==1
        campaign=root.name
        budgets={"v1":500000}
    order=tuple(x for x in ("meow","sac","optiq","mfpo") if x in methods) if a.profile=="1m" else methods
    all_jobs=[dict(id=f"{task}-{method}-s{seed}",task=task,method=method,seed=seed,steps=budgets[task],status="pending")
          for seed in seeds for method in order for task in a.tasks]
    jobs=[j for i,j in enumerate(all_jobs) if i%a.shard_count==a.shard_index]
    manifest=dict(source=str(source),source_commit=sha,smoke=a.smoke,gpus=[0,1,2,3],jobs=jobs,
        project="OptiQ/gmm-trg",group=campaign,tasks=a.tasks,budget=budgets,methods=methods,seeds=seeds,
        profile=a.profile,host=os.uname().nodename,all_jobs=all_jobs,shard_index=a.shard_index,shard_count=a.shard_count)
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
            proofs={} if a.smoke else json.loads((root/"preflight.json").read_text())
            for gpu in range(4):
                if gpu in live: continue
                pending=next((j for j in jobs if j["status"]=="pending" and
                    (a.smoke or (proofs.get(j["task"]+"-"+j["method"],{}).get("passed",False) and
                    (a.profile=="100k" or proofs[j["task"]+"-"+j["method"]].get("source_commit")==sha)))),None)
                if pending is None: break
                # Inspect reservation without taking it away from its owner.
                probe=open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock","a")
                try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    probe.close(); continue
                fcntl.flock(probe,fcntl.LOCK_UN); probe.close()
                job=pending
                cmd=[WRAPPER,str(gpu),"--branch","v5-direct-gmm",PYTHON,"-m","antmaze.multimodal.run","--task",job["task"],
                    "--method",job["method"],"--seed",str(job["seed"]),"--output",str(root/"runs"/job["id"]),
                    "--steps",str(job["steps"]),"--campaign",campaign,
                    "--eval-interval",str(25000 if a.profile=="1m" else 5000),
                    "--checkpoint-interval",str(100000 if a.profile=="1m" else 50000)]
                if a.profile=="noveld":
                    cmd=[WRAPPER,str(gpu),"--branch","v5-direct-gmm",PYTHON,"-m","antmaze.multimodal.ddiffpg_run",
                        "--task",job["task"],"--method",job["method"],"--seed",str(job["seed"]),
                        "--output",str(root/"runs"/job["id"])]
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
