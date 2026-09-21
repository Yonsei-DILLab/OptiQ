"""Use the existing approved login4 read-only backup key, scoped to new child path."""
import json,subprocess,time,shlex
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_selection_frozen')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
OUT.mkdir(parents=True,exist_ok=True)
while True:
 t=time.time()
 try:
  r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp','-e',shlex.join(SSH),
    'hobbit9882@165.132.142.208:extensions/mode_selection_frozen_b128_20260921/',str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1200)
  d=dict(time=time.time(),started=t,exit_code=r.returncode,stderr=r.stderr[-2500:])
 except Exception as e:d=dict(time=time.time(),error=str(e))
 (OUT/'SYNC_STATUS.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
 if d.get('exit_code')==0:
  count=len(list((OUT/'campaign/runtime/runs').glob('*/COMPLETE.json')))
  if count==168:break
 time.sleep(180)
