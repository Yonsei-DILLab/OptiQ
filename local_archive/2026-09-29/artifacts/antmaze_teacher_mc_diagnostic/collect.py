"""Collect completed frozen teacher diagnostics without launching jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

OUT=Path(__file__).resolve().parent
REMOTE='/home/heechan/optiq-experiments/antmaze-teacher-mc-v34-20260925'
CODE=r'''
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time
root=Path(REMOTE)
files={}
for name in ['registration.json','diagnostic/progress.json','diagnostic/result.json']:
 p=root/name
 if p.is_file():files[name]=p.read_bytes();json.loads(files[name])
if 'diagnostic/result.json' in files:
 result=json.loads(files['diagnostic/result.json'])
 assert result['completed'] and result['model_optimizer_rng_checkpoint_unchanged']
 raw=(root/'diagnostic/teacher-mc.npz').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==result['raw_sha256']
 files['diagnostic/teacher-mc.npz']=raw
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
p=subprocess.run(ctl+['status',root.name],capture_output=True,text=True)
metadata=dict(time=time.time(),read_only=True,status=p.stdout.strip(),files={},
 stderr_tail=(root/'diagnostic.err').read_text()[-2500:])
for k,v in files.items():metadata['files'][k]=dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
files['collection.json']=json.dumps(metadata,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''


def collect(host):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-'],
     input=('REMOTE='+repr(REMOTE)+'\n'+CODE).encode(),capture_output=True,timeout=90,check=True)
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
  files={}
  for member in tar.getmembers():
   path=PurePosixPath(member.name)
   assert member.isfile() and not path.is_absolute() and '..' not in path.parts
   files[member.name]=tar.extractfile(member).read()
 meta=json.loads(files['collection.json'])
 for name,record in meta['files'].items():
  assert len(files[name])==record['bytes']
  assert hashlib.sha256(files[name]).hexdigest()==record['sha256']
 destination=OUT/'results'/host
 for name,content in files.items():
  path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
  temporary=path.with_suffix(path.suffix+'.download');temporary.write_bytes(content);temporary.replace(path)
 record=dict(host=host,status=meta['status'],complete='diagnostic/result.json' in files,
     archive_sha256=hashlib.sha256(p.stdout).hexdigest(),files=len(files))
 if record['complete']:
  result=json.loads(files['diagnostic/result.json'])
  assert result['reporting_source']=='637fdb7080334fbb6f89a5c5c826191d317d458d'
  record['task']=result['task']
  record['summary']={m:{k:v[k]['mean'] for k in ['ess','max_weight','output_gradient_cosine','weighted_action_l2_error']} for m,v in result['summary'].items()}
 elif 'diagnostic/progress.json' in files:record['progress']=json.loads(files['diagnostic/progress.json'])
 return record


if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:
  rows=list(pool.map(collect,['vast-heechan-180','vast-heechan-199']))
 record=dict(time_utc=datetime.now(timezone.utc).isoformat(),records=rows,
     collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'collection.json').write_text(json.dumps(record,indent=2)+'\n')
 for row in rows:print(json.dumps(row),flush=True)
