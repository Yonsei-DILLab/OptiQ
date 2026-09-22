"""NovelD10: queue16 fresh1M runs on two4-GPU hosts, preserving native settings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from antmaze.evaluation import atomic_json

NAME="antmaze-dense-noveld10-1m-s0-20260922"
PROFILE="dense-noveld10-1m"
COEFFICIENT=10.0
METHODS=("optiq","sac","meow","mfpo")
PYTHON="/home/heechan/.venv-optiq-antmaze/bin/python"
WRAPPER="/home/heechan/OptiQ-ops/run-gpu.sh"


def source_sha():
    source=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=source,text=True).strip()
    return source,sha


def validate_run(folder,steps,smoke):
    from antmaze.multimodal.dense_noveld_report import verify_run
    sha=source_sha()[1]
    proof=verify_run(folder,allow_smoke=smoke,expected_source=sha,expected_steps=1000000,
        expected_profile=PROFILE,expected_coefficient=COEFFICIENT)
    assert proof["steps"]==steps
    proof["source_commit"]=sha
    return proof


def compare_baseline(folder,task,method,smoke):
    old=Path('/home/heechan/optiq-experiments/antmaze-dense-noveld-1m-s0-20260922')
    old=old/'preflight'/f'{task}-{method}-s0'/'initial' if smoke else old/'runs'/f'{task}-{method}-s0'
    def record(path):
        c=json.loads((path/'config.json').read_text())
        coefficient=c['intrinsic'].pop('coefficient')
        for key in ('source_commit','source_root','profile'):c.pop(key)
        c['native'].pop('output_root',None)
        return dict(config=c,initial=json.loads((path/'parameter-audit.json').read_text())['initial'],
            rnd=json.loads((path/'intrinsic-audit.json').read_text())['initial']),coefficient
    current,new_coefficient=record(folder);baseline,old_coefficient=record(old)
    assert new_coefficient==10 and old_coefficient==.01
    assert current==baseline,'Non-ablation config or initialization differs from NovelD .01 baseline'
    proof=dict(passed=True,baseline_matched=True,reference=str(old),coefficient=COEFFICIENT)
    atomic_json(folder/'baseline-config-initial-proof.json',proof)
    return proof


def worker(root,task,method,preflight):
    job=f"{task}-{method}-s0"
    def call(output,steps,*args):
        subprocess.run([sys.executable,"-m","antmaze.multimodal.dense_noveld_run","--task",task,
            "--method",method,"--output",str(output),"--steps",str(steps),
            "--noveld-coefficient",str(COEFFICIENT),"--campaign-name",NAME,"--run-name",job+"-c10-1m",
            "--profile",PROFILE,*args],check=True)
    if preflight:
        if method=="optiq" and task in ("v1","v2"):
            from antmaze.noveld_strength.check_coefficient import check
            atomic_json(root/"coefficient-proof.json",check())
        first=root/"preflight"/job/"initial";second=root/"preflight"/job/"continued"
        call(first,272,"--smoke");validate_run(first,272,True)
        baseline_proof=compare_baseline(first,task,method,True)
        call(second,280,"--smoke","--resume",str(first/"resume"/"step_0000000272"))
        proof=validate_run(second,280,True)
        restored=json.loads((second/"resume-verification.json").read_text())
        assert restored["passed"] and restored["loaded_step"]==272 and restored["loaded_updates"]==16
        proof["resume_verification"]=restored;proof["baseline_match"]=baseline_proof
        atomic_json(root/"proofs"/(job+".json"),proof)
    else:
        folder=root/"runs"/job;call(folder,1000000)
        compare_baseline(folder,task,method,False)
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
        project="OptiQ/gmm-trg",group=NAME,profile=PROFILE,coefficient=COEFFICIENT,
        only_changed_hyperparameter="NovelD coefficient .01 to10",jobs=jobs,preflight_only=a.preflight_only))
    for stage in (["preflight"] if a.preflight_only else ["preflight","training"]):
        if stage=="training":
            assert json.loads((root/"coefficient-proof.json").read_text())["passed"]
            for j in jobs:
                proof=json.loads((root/"proofs"/(j["id"]+".json")).read_text())
                assert proof["passed"] and proof["baseline_match"]["passed"]
        for j in jobs:
            for key in ("exit_code","finished","pid","gpu","command","started"):j.pop(key,None)
            j.update(status="pending",stage=stage)
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
