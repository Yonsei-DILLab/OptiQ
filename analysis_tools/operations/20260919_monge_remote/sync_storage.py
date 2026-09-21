"""Runs on dildata; dedicated read-only SSH key never leaves this storage host."""
from pathlib import Path
import subprocess,shlex,time,json,hashlib,fcntl,traceback
OPS=Path('/data1/heejoonorm/OptiQ/ops/monge-3114850247')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247')
SSH=['ssh','-p','11717','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20',
 '-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def write(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def verify():
 index=OUT/'BACKUP_VERIFIED.json';known=json.loads(index.read_text()) if index.exists() else {}
 manifests=[OUT/'base/SOURCE_MANIFEST.json',OUT/'pilot/SOURCE_MANIFEST.json']+list((OUT/'pilot/runs').glob('*/ARTIFACTS_SHA256.json'))
 for m in manifests:
  if not m.exists():continue
  signature=sha(m);key=str(m.relative_to(OUT))
  if known.get(key,{}).get('manifest_sha256')==signature:continue
  content=json.loads(m.read_text());files=content['files'] if m.name=='SOURCE_MANIFEST.json' else content
  bad=[p for p,h in files.items() if not (m.parent/p).is_file() or sha(m.parent/p)!=h]
  if bad:raise RuntimeError(str(m)+' incomplete: '+repr(bad[:10]))
  known[key]=dict(manifest_sha256=signature,files=len(files),verified_at=time.time());write(index,known)
 return len(known)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OPS/'sync.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:
    r=subprocess.run(['rsync','-rtz','--timeout=90','--exclude=__pycache__','--exclude=*.tmp','--exclude=base_source.tar.gz',
     '-e',shlex.join(SSH),'heejoonorm@31.148.50.247:',str(OUT)+'/'],text=True,capture_output=True,timeout=600)
    assert r.returncode in (0,24),(r.returncode,r.stderr)
    count=verify();write(OUT/'STORAGE_SYNC_STATUS.json',dict(ok=True,time=time.time(),verified_manifests=count,code=r.returncode))
   except Exception:write(OUT/'STORAGE_SYNC_STATUS.json',dict(ok=False,time=time.time(),error=traceback.format_exc()))
   time.sleep(120)
if __name__=='__main__':main()
