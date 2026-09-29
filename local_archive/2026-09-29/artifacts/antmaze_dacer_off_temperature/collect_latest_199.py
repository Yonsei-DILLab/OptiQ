"""Read-only snapshot of completed evaluations in the approved DACER OFF experiment."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib, io, json, pathlib, sys, tarfile
r=pathlib.Path('/home/heechan/optiq-experiments/antmaze-optiq-dense-dacer-off-T3-T5-T10-s0-20260924')
paths=list(r.glob('*.json'))+list((r/'jobs').glob('*.json'))
for run in sorted((r/'runs').glob('*')):
 if not run.is_dir():continue
 paths.extend(run.glob('*.json'))
 # summary is written only after its raw rollout NPZ has finished writing.
 summaries=sorted((run/'evaluations').glob('*/*/summary.json'))
 paths.extend(summaries)
 policy=[p for p in summaries if p.parent.name=='policy-natural']
 latest=max((int(p.parent.parent.name) for p in policy),default=-1)
 for p in summaries:
  if int(p.parent.parent.name)==latest:
   raw=p.parent/'rollouts.npz'
   if raw.is_file():paths.append(raw)
digests={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(set(paths)):
  raw=path.read_bytes();name=str(path.relative_to(r))
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
        proc = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', host,
                               "python3 - <<'PY'\n" + REMOTE + '\nPY'],
                              stdout=output, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    with tarfile.open(path) as archive:
        archive.extractall(folder, filter='data')
    digests = json.loads((folder/'collected-sha256.json').read_text())
    for name, digest in digests.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    return dict(host=host, files=len(digests),
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT/'reports'/stamp
    destination.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(collect, [(h, destination) for h in
                                         ['vast-heechan-199']]))
    (destination/'collection-verification.json').write_text(json.dumps(dict(
        timestamp_utc=stamp, verified=True, hosts=results,
        scope='Saved evaluation summaries/raw rollouts and status/config JSON; no new rollouts or training changes.'), indent=2)+'\n')
    (ROOT/'latest-report-path.txt').write_text(str(destination)+'\n')
    print(json.dumps(dict(destination=str(destination), hosts=results), indent=2))
