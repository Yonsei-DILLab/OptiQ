"""Finite dildata collector. Never starts or stops GPU work."""
from pathlib import Path
import os,time,json,subprocess,hashlib,fcntl,shlex,traceback
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/nonstationary-q-20260917')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260917_nonstationary_q')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',
     '-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]

def write(p,obj):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def once():
    OUT.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=.pytest_cache',
        '--exclude=*.tmp','--exclude=/ops/*.lock','-e',shlex.join(SSH),
        'hobbit9882@165.132.142.208:./',str(OUT)+'/'],check=True,timeout=1800)
    verified=[]
    for marker in OUT.glob('runs/*/COMPLETE.json'):
        rec=marker.parent/'BACKUP_VERIFIED.json'
        if rec.exists():verified.append(marker.parent.name);continue
        manifest=marker.parent/'ARTIFACTS_SHA256.json'
        if not manifest.exists():continue
        files=json.loads(manifest.read_text())
        if all((marker.parent/f).exists() and sha(marker.parent/f)==h for f,h in files.items()):
            write(rec,dict(verified=True,files=len(files),time=time.time()));verified.append(marker.parent.name)
    report_status=OUT/'report/STATUS.json'
    report_ready=report_status.exists() and json.loads(report_status.read_text()).get('comparison_complete')==448
    status=dict(ok=True,time=time.time(),verified_nodes=len(verified),total_nodes=564,
                report_ready=report_ready,destination=str(OUT))
    write(OPS/'SYNC_STATUS.json',status);write(OUT/'STORAGE_SYNC_STATUS.json',status)
    return len(verified)==564 and report_ready

if __name__=='__main__':
    lock=(OPS/'sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            if once():break
        except Exception:write(OPS/'SYNC_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        time.sleep(180)
