"""Run on dildata; read-only backup via the already authorized login4 route."""
import argparse,json,shlex,subprocess,time
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260925_kl_nongmm_targets')
ext='kl_nongmm_targets_20260925'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
OUT.mkdir(parents=True,exist_ok=True)
while True:
    try:
        r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp','-e',shlex.join(SSH),
                          f'hobbit9882@165.132.142.208:extensions/{ext}/',str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1200)
        result=dict(time=time.time(),exit_code=r.returncode,stderr=r.stderr[-2500:])
    except Exception as e:result=dict(time=time.time(),error=repr(e))
    (OUT/'SYNC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    if result.get('exit_code')==0 and (OUT/'campaign/REVIEW_READY.json').exists():break
    time.sleep(180)
