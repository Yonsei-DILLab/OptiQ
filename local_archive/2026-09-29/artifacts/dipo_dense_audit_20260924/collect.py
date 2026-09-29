"""Read-only archive of every actual dense-reward DIPO run on the two hosts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE=r'''
import io,json,pathlib,hashlib,sys,tarfile,datetime
base=pathlib.Path('/home/heechan/optiq-experiments')
runs=[];paths=set();inventory=[]
for cfgpath in sorted(base.rglob('config.json')):
 if 'dipo' not in str(cfgpath).lower() or 'preflight' in str(cfgpath).lower():continue
 if '/runs/' not in str(cfgpath):continue
 cfg=json.loads(cfgpath.read_text())
 desc=cfg.get('reward','')
 dense=cfg.get('reward_profile')=='dense' or any(s in desc for s in ['negative Euclidean','nearest goal distance','-min distance'])
 if not dense:continue
 run=cfgpath.parent;campaign=run.parent.parent;runs.append(str(run.relative_to(base)))
 paths.update(p for p in run.glob('*.json') if p.is_file())
 paths.update(p for p in campaign.glob('*.json') if p.is_file())
 for summary in (run/'evaluations').glob('*/*/summary.json'):
  npz=summary.with_name('rollouts.npz')
  if npz.exists():paths.update([summary,npz])
 for p in (run/'rollouts').glob('*'):
  if p.is_file() and p.suffix in ('.npz','.json'):paths.add(p)
 inventory.append(dict(run=str(run.relative_to(base)),source=cfg.get('source_commit'),task=cfg.get('task'),reward=desc,files=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and ('checkpoint' not in str(p))]))
digests={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(paths):
  raw=path.read_bytes();name=str(path.relative_to(base))
  if path.suffix=='.json':json.loads(raw)
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
  digests[name]=hashlib.sha256(raw).hexdigest()
 for name,data in [('collected-sha256.json',digests),('inventory.json',dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=inventory))]:
  raw=json.dumps(data,indent=2).encode();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
'''

def collect(item):
 host,destination=item;folder=destination/host;folder.mkdir()
 archive=destination/(host+'.tar.gz')
 with archive.open('wb') as f:
  r=subprocess.run(['ssh',host,"python3 - <<'PY'\n"+REMOTE+'\nPY'],stdout=f,stderr=subprocess.PIPE)
 if r.returncode:raise RuntimeError(r.stderr.decode())
 with tarfile.open(archive) as tar:tar.extractall(folder,filter='data')
 digests=json.loads((folder/'collected-sha256.json').read_text())
 for name,digest in digests.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
 return dict(host=host,files=len(digests),runs=len(json.loads((folder/'inventory.json').read_text())['runs']),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__':
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 destination=ROOT/stamp;destination.mkdir(parents=True)
 rows=list(ThreadPoolExecutor(2).map(collect,[(h,destination) for h in ['vast-heechan-180','vast-heechan-199']]))
 (destination/'collection-verification.json').write_text(json.dumps(dict(timestamp_utc=stamp,verified=True,hosts=rows),indent=2)+'\n')
 (ROOT/'latest.txt').write_text(str(destination)+'\n')
 print(json.dumps(dict(destination=str(destination),hosts=rows),indent=2))
