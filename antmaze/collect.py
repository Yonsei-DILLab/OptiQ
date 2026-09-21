"""Collect recoverable experiment outputs to the local antmaze/results folder."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",default="antmaze-multimodal-100k-20260921")
    p.add_argument("--host",default="vast-heechan-199")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    remote=f"{a.host}:/home/heechan/optiq-experiments/{a.campaign}/"
    subprocess.run(["rsync","-az","--prune-empty-dirs","--exclude=wandb/","--exclude=tensorboard/",
        "--exclude=checkpoints/","--exclude=train_log/","--include=*/","--include=*.json",
        "--include=*.npz","--include=*.png","--include=*.md","--include=*.txt","--exclude=*",
        remote,str(a.output)+"/"],check=True)
    s=json.loads((a.output/"status.json").read_text())
    print(json.dumps({k:s[k] for k in ("phase","completed","total","source_commit")},indent=2))


if __name__=="__main__":main()
