"""Collect recoverable experiment outputs to the local antmaze/results folder."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",default="antmaze-multimodal-100k-20260921")
    p.add_argument("--host",default="vast-heechan-199")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    p.add_argument("--final-checkpoints",action="store_true",help="Also collect audited final policy/critic checkpoints")
    p.add_argument("--verify",action="store_true",help="Require final remote validation and verify every archived SHA256 locally")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    remote=f"{a.host}:/home/heechan/optiq-experiments/{a.campaign}/"
    subprocess.run(["rsync","-az","--prune-empty-dirs","--exclude=wandb/","--exclude=tensorboard/",
        "--exclude=checkpoints/","--exclude=train_log/","--include=*/","--include=*.json",
        "--include=*.npz","--include=*.png","--include=*.md","--include=*.txt","--exclude=*",
        remote,str(a.output)+"/"],check=True)
    if a.final_checkpoints:
        files=[]
        status=json.loads((a.output/"status.json").read_text())
        for job in status["jobs"]:
            if job["status"]!="completed":continue
            folder=Path("runs")/job["id"]
            checkpoint=json.loads((a.output/folder/"checkpoint.json").read_text())
            for item in checkpoint["files"]:
                assert Path(item["name"]).name==item["name"]
                files.append(str(folder/"checkpoints"/item["name"]))
        if files:
            with tempfile.NamedTemporaryFile(mode="w") as file:
                file.write("\n".join(files)+"\n");file.flush()
                subprocess.run(["rsync","-az","--checksum","--files-from",file.name,
                                remote,str(a.output)+"/"],check=True)
    if a.verify:
        validation=json.loads((a.output/"report/validation.json").read_text())
        assert validation["complete"] and len(validation["verified"])==validation["total"]
        hashes={}
        for job,record in validation["verified"].items():
            for relative,expected in record["sha256"].items():
                assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
                path=a.output/"runs"/job/relative
                h=hashlib.sha256()
                with path.open("rb") as file:
                    for block in iter(lambda:file.read(1024*1024),b""):h.update(block)
                assert h.hexdigest()==expected,f"Archive integrity mismatch: {path}"
                hashes[str(path.relative_to(a.output))]=expected
        (a.output/"local-integrity.json").write_text(json.dumps(dict(passed=True,
            training_source_commit=validation["training_source_commit"],
            reporting_source_commit=validation["reporting_source_commit"],sha256=hashes),indent=2)+"\n")
    s=json.loads((a.output/"status.json").read_text())
    print(json.dumps({k:s[k] for k in ("phase","completed","total","source_commit")},indent=2))


if __name__=="__main__":main()
