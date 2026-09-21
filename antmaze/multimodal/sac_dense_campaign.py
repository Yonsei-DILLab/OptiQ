"""One authorized SAC dense-only pilot, with preflight and final verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from antmaze.evaluation import atomic_json


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() or (root/"preflight"/"manifest.json").exists():
        raise RuntimeError("Existing campaign; refuse duplicate launch or automatic resume")
    source=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    def call(module,*argv):
        subprocess.run([sys.executable,"-m",module,*map(str,argv)],cwd=source,check=True)
    common=["--profile","sac-dense","--methods","sac","--tasks","v1","--seeds","0"]
    started=time.time()
    try:
        atomic_json(root/"pipeline.json",dict(phase="preflight",source_commit=commit,started=started))
        call("antmaze.multimodal.check_env")
        call("antmaze.multimodal.controller","--root",root/"preflight",*common,"--smoke")
        from .verify import verify_campaign
        proof=verify_campaign(root/"preflight",preflight=True)
        assert proof["complete"] and proof["training_source_commit"]==commit
        atomic_json(root/"preflight.json",{"v1-sac":dict(passed=True,source_commit=commit,verification=proof)})
        atomic_json(root/"pipeline.json",dict(phase="training",source_commit=commit,started=started))
        call("antmaze.multimodal.controller","--root",root,*common)
        call("antmaze.multimodal.report","--root",root)
        result=json.loads((root/"runs/v1-sac-s0/result.json").read_text())
        episodes=json.loads((root/"runs/v1-sac-s0/training_episodes.json").read_text())
        assert result["completed"] and result["steps"]==500000 and result["updates"]==495000
        assert episodes["training_steps"]==500000
        assert sum(x["length"] for x in episodes["episodes"])<=500000
        assert sum(x["success"] for x in episodes["episodes"])==episodes["training_successes"]
        atomic_json(root/"pipeline.json",dict(phase="completed",source_commit=commit,started=started,
            finished=time.time(),training=result["training"],
            evaluation={k:dict(episodes=v["episodes"],success_rate=v["success_rate"])
                        for k,v in result["summaries"].items()}))
    except BaseException as error:
        atomic_json(root/"pipeline-failure.json",dict(type=type(error).__name__,message=str(error),
            source_commit=commit,automatic_restart=False))
        raise


if __name__=="__main__":main()
