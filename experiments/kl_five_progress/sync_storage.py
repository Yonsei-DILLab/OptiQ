"""Existing restricted read-only backup; render on dildata after all runs finish."""
import json,os,shlex,subprocess,time
from pathlib import Path
OPS=Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT=Path('/data1/heejoonorm/OptiQ/studies/20260926_kl_five_progress')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(OPS/'known_hosts'),'-i',str(OPS/'backup_key')]
OUT.mkdir(parents=True,exist_ok=True)
while True:
    try:
        r=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=__pycache__','--exclude=*.tmp',
            '-e',shlex.join(SSH),'hobbit9882@165.132.142.208:extensions/kl_five_progress_20260926/',
            str(OUT/'campaign')+'/'],capture_output=True,text=True,timeout=1500)
        state=dict(time=time.time(),exit_code=r.returncode,stderr=r.stderr[-2000:])
    except Exception as e:state=dict(time=time.time(),error=repr(e))
    complete=list((OUT/'campaign/runtime/replay').glob('*/*/COMPLETE.json'))
    state['complete']=len(complete)
    (OUT/'SYNC_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    # Figures are rendered by a CPU-only job on login4, so this copies them too.
    if state.get('exit_code')==0 and len(complete)==40 and (OUT/'campaign/reports/ten_k/SHA256.json').exists():break
    time.sleep(180)
