"""Preflight and queue the authorized 16 runs across two four-GPU hosts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from antmaze.evaluation import atomic_json

METHODS=("optiq","sac","meow","mfpo")
PYTHON="/home/heechan/.venv-optiq-antmaze/bin/python"
WRAPPER="/home/heechan/OptiQ-ops/run-gpu.sh"


def validate_run(folder,steps,smoke):
    import numpy as np
    from .analysis import summarize
    r=json.loads((folder/"result.json").read_text());c=json.loads((folder/"config.json").read_text())
    assert r["completed"] and r["steps"]==steps and r["updates"]==steps-c["warmup"]
    assert c["batch_size"]==256 and c["utd"]==1 and c["num_envs"]==1
    assert c["intrinsic"]["type"]=="noveld" and "-nearest goal distance" in c["reward"]
    assert r["training"]["training_steps"]==steps and r["replay_size"]==min(steps,1000000)
    proof=json.loads((folder/"checkpoint.json").read_text())
    assert proof["step"]==steps and proof["replay_size"]==r["replay_size"]
    assert json.loads((folder/"intrinsic-audit.json").read_text())["passed"]
    for mode,s in r["summaries"].items():
        with np.load(folder/"rollouts"/f"{steps}-{mode}.npz") as d:
            assert len(d["returns"])==(2 if smoke else 100)
            assert np.isfinite(d["returns"]).all()
            computed=summarize(c["task"],d["xy"],d["lengths"],d["goal_ids"],d["returns"])
            assert computed["success_rate"]==s["success_rate"]
            assert computed["routes"]==s["routes"]
    return dict(passed=True,source_commit=c["source_commit"],steps=steps)


def worker(root,task,method,preflight):
    job=f"{task}-{method}-s0"
    def call(output,steps,*args):
        subprocess.run([sys.executable,"-m","antmaze.multimodal.dense_noveld_run","--task",task,
            "--method",method,"--output",str(output),"--steps",str(steps),*args],check=True)
    if preflight:
        first=root/"preflight"/job/"initial";second=root/"preflight"/job/"continued"
        call(first,272,"--smoke");validate_run(first,272,True)
        call(second,280,"--smoke","--resume",str(first/"resume"/"step_0000000272"))
        proof=validate_run(second,280,True)
        restored=json.loads((second/"resume-verification.json").read_text())
        assert restored["passed"] and restored["loaded_step"]==272 and restored["loaded_updates"]==16
        proof["resume_verification"]=restored
        atomic_json(root/"proofs"/(job+".json"),proof)
    else:
        folder=root/"runs"/job;call(folder,1000000)
        atomic_json(folder/"verification.json",validate_run(folder,1000000,False))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--shard",type=int,choices=[0,1],default=0)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--worker",choices=METHODS)
    p.add_argument("--task",choices=["v1","v2","v3","v4"])
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args();root=a.root
    if a.worker:return worker(root,a.task,a.worker,a.preflight)
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/"manifest.json").exists(),"Refuse duplicate registration"
    source=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=source,text=True).strip()
    tasks=["v1","v3"] if a.shard==0 else ["v2","v4"]
    jobs=[dict(id=f"{task}-{m}-s0",task=task,method=m,seed=0,steps=1000000,status="pending") for task in tasks for m in METHODS]
    for sub in ("logs","jobs","proofs","runs"):(root/sub).mkdir(exist_ok=True)
    atomic_json(root/"manifest.json",dict(source=str(source),source_commit=sha,shard=a.shard,
        all_tasks=["v1","v2","v3","v4"],methods=METHODS,seed=0,total_across_hosts=16,
        project="OptiQ/gmm-trg",jobs=jobs,preflight_only=a.preflight_only))
    subprocess.run([sys.executable,"-m","antmaze.multimodal.check_env"],check=True)
    for stage in (["preflight"] if a.preflight_only else ["preflight","training"]):
        for j in jobs:j.update(status="pending",stage=stage)
        live={};failed=False
        while True:
            for gpu,(proc,j,log) in list(live.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del live[gpu]
                proof=root/"proofs"/(j["id"]+".json") if stage=="preflight" else root/"runs"/j["id"]/"verification.json"
                passed=code==0 and proof.exists() and json.loads(proof.read_text())["passed"]
                j.update(status="completed" if passed else "failed",exit_code=code,finished=time.time())
                failed|=not passed
                if not passed:atomic_json(root/"failure.json",dict(job=j,pending_held=True,live_preserved=True,automatic_restart=False))
                atomic_json(root/"jobs"/(j["id"]+".json"),j)
            if not failed:
                for gpu in range(4):
                    if gpu in live:continue
                    j=next((j for j in jobs if j["status"]=="pending"),None)
                    if j is None:break
                    probe=open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock","a")
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:probe.close();continue
                    fcntl.flock(probe,fcntl.LOCK_UN);probe.close()
                    cmd=[WRAPPER,str(gpu),"--branch","v5-direct-gmm",PYTHON,"-m",__spec__.name,
                         "--root",str(root),"--worker",j["method"],"--task",j["task"]]
                    if stage=="preflight":cmd.append("--preflight")
                    else:
                        proof=json.loads((root/"proofs"/(j["id"]+".json")).read_text())
                        assert proof["passed"] and proof["source_commit"]==sha
                    env=os.environ.copy();env.update(OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu),PYTHONPATH=str(source))
                    log=(root/"logs"/f"{stage}-{j['id']}.log").open("w")
                    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    j.update(status="running",pid=proc.pid,gpu=gpu,started=time.time(),command=cmd)
                    atomic_json(root/"jobs"/(j["id"]+".json"),j);live[gpu]=(proc,j,log)
            done=sum(j["status"]=="completed" for j in jobs)
            atomic_json(root/"status.json",dict(stage=stage,phase="failed" if failed else "running",
                source_commit=sha,updated=time.time(),completed=done,total=len(jobs),jobs=jobs))
            if not live and (failed or done==len(jobs)):
                if failed:raise RuntimeError(f"{stage} failed; no automatic restart")
                break
            time.sleep(2)
    atomic_json(root/"status.json",dict(phase="completed",stage=stage,source_commit=sha,completed=len(jobs),total=len(jobs),jobs=jobs))
    atomic_json(root/"result.json",dict(completed=True,preflight_only=a.preflight_only,source_commit=sha,jobs=jobs))


if __name__=="__main__":main()
