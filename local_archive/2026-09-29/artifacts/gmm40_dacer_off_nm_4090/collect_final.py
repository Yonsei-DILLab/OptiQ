"""Collect and verify the final 100k GMM40 records without changing training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
HOST = "vast1"
CAMPAIGNS = {
    "nm64": "gmm40-ibolt-dacer-off-nm64-100k-4seed-4090-20260925",
    "nm128_256_512": "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(which: str) -> dict:
    campaign = CAMPAIGNS[which]
    remote_root = "/home/heechan/optiq-experiments/" + campaign
    remote_code = '''
from pathlib import Path
import hashlib,json,tarfile
r=Path(%r)
s=json.loads((r/'status.json').read_text())
if s['phase']!='completed':
 print(json.dumps({'phase':s['phase'],'completed':len(s.get('completed',[])),
                   'running':s.get('running',[]),'queued':len(s.get('queued',[]))}))
else:
 m=json.loads((r/'manifest.json').read_text())
 assert len(s['completed'])==len(m['jobs'])
 files=['manifest.json','registration.json','status.json','result.json',
        'archive.json','results/target/definition.json']
 files += [str(p.relative_to(r)) for p in (r/'jobs').glob('*.json')]
 files += [str(p.relative_to(r)) for p in (r/'proofs').glob('*.json')]
 for job in m['jobs']:
  name=job['name']; base='results/'+name+'/'
  for item in ('config.json','status.json','latest.json','update_count_audit.json',
               'wandb_status.json','metrics.jsonl','model_sizes.json'):
   files.append(base+item)
  for item in ('samples.npy','metrics.json','samples_mu_only.npy','metrics_mu_only.json'):
   files.append(base+'evaluations/step_0100000/'+item)
 assert all((r/p).is_file() for p in files), [p for p in files if not (r/p).is_file()]
 archive=r/'final-comparison-inputs.tar.gz'
 with tarfile.open(archive,'w:gz') as output:
  for rel in sorted(files):output.add(r/rel,arcname=rel)
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 print(json.dumps({'phase':'completed','jobs':len(m['jobs']),'archive':archive.name,
                   'sha256':digest,'bytes':archive.stat().st_size,'members':len(files)}))
''' % remote_root
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,
                           "python3 -c " + shlex.quote(remote_code)],
                          check=True, capture_output=True, text=True)
    message = json.loads(proc.stdout.splitlines()[-1])
    if message["phase"] != "completed":
        return message
    out = HERE/"inputs"/campaign
    out.mkdir(parents=True, exist_ok=True)
    path = HERE/"inputs"/(campaign+".tar.gz")
    if not path.exists() or path.stat().st_size != message["bytes"] or sha256(path) != message["sha256"]:
        partial = path.with_suffix(".partial.tar.gz")
        if partial.exists():
            partial.unlink()
        subprocess.run(["scp", "-q", HOST+":"+remote_root+"/"+message["archive"], str(partial)], check=True)
        assert partial.stat().st_size == message["bytes"] and sha256(partial) == message["sha256"]
        partial.replace(path)
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len(members) == message["members"]
        archive.extractall(out, filter="data")
    receipt = dict(campaign=campaign, archive_sha256=message["sha256"],
                   archive_bytes=message["bytes"], files={})
    for member in members:
        if member.isfile():
            local = out/member.name
            assert local.is_file() and local.stat().st_size == member.size
            receipt["files"][member.name] = sha256(local)
    (out/"LOCAL_COPY_VERIFIED.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return dict(phase="completed", campaign=campaign, jobs=message["jobs"],
                files=len(receipt["files"]), archive_bytes=message["bytes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=["nm64", "nm128_256_512", "all"], default="all")
    args = parser.parse_args()
    names = CAMPAIGNS if args.campaign == "all" else [args.campaign]
    print(json.dumps({name: collect(name) for name in names}, indent=2))


if __name__ == "__main__":
    main()
