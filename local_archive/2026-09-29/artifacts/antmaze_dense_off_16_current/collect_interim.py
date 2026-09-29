"""Collect immutable copies of completed natural-start evaluations, without GPU work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'antmaze-dense-off-16-current-s0-20260924'
REMOTE = r'''
import hashlib, io, json, pathlib, sys, tarfile
root=pathlib.Path('/home/heechan/optiq-experiments/antmaze-dense-off-16-current-s0-20260924')
paths=[p for p in root.glob('*.json') if p.is_file()]
for run in sorted((root/'runs').iterdir()):
 paths.extend(p for p in run.glob('*.json') if p.is_file())
 for summary in sorted((run/'evaluations').glob('*/*-natural/summary.json')):
  data=summary.with_name('rollouts.npz')
  if data.exists(): paths.extend([summary,data])
digests={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in paths:
  raw=path.read_bytes(); name=str(path.relative_to(root))
  if path.suffix=='.json': json.loads(raw)
  info=tarfile.TarInfo(name);info.size=len(raw)
  archive.addfile(info,io.BytesIO(raw))
  digests[name]=hashlib.sha256(raw).hexdigest()
 raw=json.dumps(digests,indent=2).encode()
 info=tarfile.TarInfo('collected-sha256.json');info.size=len(raw)
 archive.addfile(info,io.BytesIO(raw))
'''

def collect(item):
    host, destination = item
    folder=destination/host;folder.mkdir()
    path=destination/(host+'.tar.gz')
    with path.open('wb') as output:
        process=subprocess.run(['ssh',host,"python3 - <<'PY'\n"+REMOTE+'\nPY'],stdout=output,stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr.decode())
    with tarfile.open(path) as archive:
        archive.extractall(folder,filter='data')
    digests=json.loads((folder/'collected-sha256.json').read_text())
    for name,digest in digests.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    return dict(host=host,files=len(digests),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'interim'/stamp;destination.mkdir(parents=True)
    hosts=['vast-heechan-180','vast-heechan-199']
    results=list(ThreadPoolExecutor(max_workers=2).map(collect,[(h,destination) for h in hosts]))
    (destination/'collection-verification.json').write_text(json.dumps(dict(timestamp_utc=stamp,campaign=CAMPAIGN,verified=True,hosts=results),indent=2)+'\n')
    (ROOT/'latest-interim-path.txt').write_text(str(destination)+'\n')
    print(json.dumps(dict(destination=str(destination),hosts=results),indent=2))
