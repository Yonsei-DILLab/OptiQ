"""Run on dildata; reuse existing folder-restricted read-only login4 key."""
import json
import shlex
import subprocess
import time
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260923_forward_far')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
OUT.mkdir(parents=True,exist_ok=True)
while True:
    started=time.time()
    try:
        r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp','-e',shlex.join(SSH),
            'hobbit9882@165.132.142.208:extensions/forward_far_20260923/',str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1200)
        result=dict(time=time.time(),started=started,exit_code=r.returncode,stderr=r.stderr[-2500:])
    except Exception as e:result=dict(time=time.time(),error=repr(e))
    (OUT/'SYNC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result.get('exit_code')==0:
        count=len(list((OUT/'campaign').glob('N*/runtime/runs/*/COMPLETE.json')))
        if count==16:break
    time.sleep(180)
