"""One guarded GPU, sequential read-only checkpoint evaluations, no retries."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from .run_nway_job import atomic_json


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--main-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reporting-source",required=True)
    a=p.parse_args()
    jobs=[]
    for task in ("4way","8way","16way"):
        for temp in (1,3,5,10):
            name=f"{task}-optiq-t{temp}-s0"
            source=a.inputs/name if task!="4way" and temp==1 else a.main_root/"runs"/name
            jobs.append(dict(name=name,source=str(source)))
    for maze in ("simple","medium","hard"):
        name=f"pm_{maze}-optiq-s0"; jobs.append(dict(name=name,source=str(a.inputs/name)))
    a.output.mkdir(exist_ok=False,parents=True)
    state=dict(status="running",source=a.reporting_source,jobs=jobs,completed=[],started=time.time())
    atomic_json(a.output/"status.json",state)
    for job in jobs:
        cmd=[sys.executable,"-m","maze_benchmarks.posthoc_mu","--run",job["source"],
             "--output",str(a.output/job["name"]),"--reporting-source",a.reporting_source]
        state["current"]=job["name"]; atomic_json(a.output/"status.json",state)
        with (a.output/(job["name"]+".log")).open("w") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status="failed",exit_code=result.returncode)
            atomic_json(a.output/"status.json",state)
            raise SystemExit(result.returncode)
        state["completed"].append(job["name"])
        atomic_json(a.output/"status.json",state)
    state.update(status="complete",finished=time.time())
    atomic_json(a.output/"status.json",state)


if __name__=="__main__":main()
