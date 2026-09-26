"""Reuse the authorized read-only route; wait for both figure families."""
import json,shlex,subprocess,time
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260926_kl_five_progress')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
while True:
    try:
        r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp','-e',shlex.join(SSH),
            'hobbit9882@165.132.142.208:extensions/kl_five_progress_20260926/',str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1800)
        state=dict(time=time.time(),exit_code=r.returncode,stderr=r.stderr[-2000:])
    except Exception as e:state=dict(time=time.time(),error=repr(e))
    state['complete']=len(list((OUT/'campaign/attempt4/runtime/replay').glob('*/*/COMPLETE.json')))
    state['early_complete']=len(list((OUT/'campaign/early_steps/runtime/early').glob('*/*/COMPLETE.json')))
    (OUT/'SYNC_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    reports=[OUT/'campaign/attempt4/reports/ten_k/SHA256.json',OUT/'campaign/early_steps/reports/log_steps/SHA256.json']
    if state.get('exit_code')==0 and all(p.exists() for p in reports):break
    time.sleep(180)
