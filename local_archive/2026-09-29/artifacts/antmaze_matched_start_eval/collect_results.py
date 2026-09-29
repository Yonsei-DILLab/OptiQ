from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import subprocess,tarfile,hashlib,json
ROOT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,io,sys,tarfile,hashlib
r=Path('/home/heechan/optiq-experiments/antmaze-matched-start-eval-20260924-r2')
paths=[]
for p in r.glob('*.json'):paths.append((p,str(p.relative_to(r))))
for job in (r/'runs').glob('*'):
 completed=sorted([p for p in job.glob('step_*') if (p/'result.json').exists() and (p/'verification.json').exists()])
 if not completed:continue
 folder=completed[-1];proof=json.loads((folder/'verification.json').read_text());assert proof['passed']
 for p in folder.glob('*.json'):paths.append((p,str(p.relative_to(r))))
 for p in folder.glob('evaluations/*/*/*'):
  if p.is_file() and p.suffix in ('.json','.npz'):paths.append((p,str(p.relative_to(r))))
 prov=json.loads((folder/'provenance.json').read_text());step=prov['checkpoint_steps']
 parent=Path(prov['parent_run'])/'evaluations'/f'{step:010d}'/'policy-natural'
 if (parent/'summary.json').exists():
  for name in ['rollouts.npz','summary.json']:
   p=parent/name;paths.append((p,str(folder.relative_to(r))+'/paired_random/'+name))
checks={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p,name in paths:
  raw=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw));checks[name]=hashlib.sha256(raw).hexdigest()
 raw=json.dumps(checks).encode();info=tarfile.TarInfo('sha256.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
'''
def collect(item):
 host,dest=item;file=dest/(host+'.tar.gz')
 with file.open('wb') as f:
  p=subprocess.run(['ssh',host,'python3','-'],input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=180)
 assert p.returncode==0,p.stderr.decode()
 folder=dest/host;folder.mkdir()
 with tarfile.open(file) as t:t.extractall(folder,filter='data')
 for name,digest in json.loads((folder/'sha256.json').read_text()).items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
 return {'host':host,'verified':True,'archive_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
if __name__=='__main__':
 dest=ROOT/'reports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest.mkdir(parents=True)
 with ThreadPoolExecutor(3) as pool:rows=list(pool.map(collect,[(h,dest) for h in ['vast-heechan-180','vast-heechan-199','vast1']]))
 (dest/'collection.json').write_text(json.dumps(rows,indent=2));(ROOT/'latest-report-path.txt').write_text(str(dest)+'\n');print(dest)
