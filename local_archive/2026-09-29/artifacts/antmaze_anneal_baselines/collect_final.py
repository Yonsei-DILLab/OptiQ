"""Read-only archive of final rollouts, configs and all evaluation summaries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib, io, json, pathlib, sys, tarfile
root=pathlib.Path('/home/heechan/optiq-experiments')
paths=[]
for campaign in ['antmaze-dense-anneal-baselines-s0-20260924', 'antmaze-optiq-dense-off-T1-s0-20260924', 'antmaze-dense-off-16-current-s0-20260924']:
 campaign_root=root/campaign
 paths.extend(p for p in campaign_root.glob('*.json') if p.is_file())
 for run in sorted((campaign_root/'runs').glob('*')):
  if not run.is_dir():continue
  paths.extend(p for p in run.glob('*.json') if p.is_file())
  paths.extend((run/'evaluations').glob('*/*/summary.json'))
  final=run/'result.json'
  if final.is_file():
   result=json.loads(final.read_text())
   for mode in ['policy-natural','policy-fixed','native-natural']:
    p=run/'evaluations'/f"{result['steps']:010d}"/mode/'rollouts.npz'
    if p.is_file():paths.append(p)
digests={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in paths:
  raw=path.read_bytes(); name=str(path.relative_to(root))
  if path.suffix=='.json':json.loads(raw)
  info=tarfile.TarInfo(name);info.size=len(raw)
  archive.addfile(info,io.BytesIO(raw));digests[name]=hashlib.sha256(raw).hexdigest()
 raw=json.dumps(digests,indent=2).encode()
 info=tarfile.TarInfo('collected-sha256.json');info.size=len(raw)
 archive.addfile(info,io.BytesIO(raw))
'''

def collect(item):
    host, destination = item
    folder = destination / host
    folder.mkdir()
    path = destination / (host + '.tar.gz')
    with path.open('wb') as output:
        process = subprocess.run(['ssh', host, "python3 - <<'PY'\n" + REMOTE + '\nPY'], stdout=output, stderr=subprocess.PIPE, timeout=180)
    if process.returncode:
        raise RuntimeError(process.stderr.decode())
    with tarfile.open(path) as archive:
        archive.extractall(folder, filter='data')
    digests = json.loads((folder / 'collected-sha256.json').read_text())
    for name, digest in digests.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    return dict(host=host, files=len(digests), archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'reports' / stamp
    destination.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(collect, [(host,destination) for host in ['vast-heechan-180','vast-heechan-199']]))
    (destination/'collection-verification.json').write_text(json.dumps(dict(timestamp_utc=stamp,verified=True,hosts=results,scope='Config, result, checkpoint verification sidecar, evaluation summaries and final raw rollouts; full model/replay checkpoints remain on servers.'),indent=2)+'\n')
    (ROOT/'latest-report-path.txt').write_text(str(destination)+'\n')
    print(json.dumps(dict(destination=str(destination),hosts=results),indent=2))
