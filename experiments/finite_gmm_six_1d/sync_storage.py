"""Execute on dildata; existing read-only SSH route, no new authorization keys."""
import json,shlex,subprocess,time
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260925_finite_gmm_six')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
OUT.mkdir(parents=True,exist_ok=True)
while True:
    try:
        r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp','-e',shlex.join(SSH),
             'hobbit9882@165.132.142.208:extensions/finite_gmm_six_20260925/',str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1200)
        result=dict(time=time.time(),exit_code=r.returncode,stderr=r.stderr[-2000:])
    except Exception as e:result=dict(time=time.time(),error=repr(e))
    (OUT/'SYNC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    if result.get('exit_code')==0 and len(list((OUT/'campaign/runs').glob('*/*/COMPLETE.json')))==75:break
    time.sleep(180)
