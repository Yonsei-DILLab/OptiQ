"""Run on dildata: read-only, scoped pull of one frozen experiment campaign."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/data1/heejoonorm/OptiQ/studies/20260921_trg_single_mc')
ROOT.mkdir(parents=True,exist_ok=True)
SSH='ssh -p 37047 -i /home/heejoonorm/.ssh/optiq_trg_103177249208 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o UserKnownHostsFile=/home/heejoonorm/.ssh/optiq_trg_known_hosts -o StrictHostKeyChecking=yes'
# Forced rrsync command restricts this remote root to this campaign only.
cmd=['rsync','-az','--timeout=90','--partial','--exclude=__pycache__/','--exclude=.pytest_cache/','-e',SSH,
     'heejoonorm@103.177.249.208:./',str(ROOT/'campaign')+'/']
while True:
    started=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
        result=dict(started=started,finished=time.time(),exit_code=r.returncode,stderr=r.stderr[-3000:])
    except Exception as e:result=dict(started=started,finished=time.time(),error=str(e))
    (ROOT/'SYNC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    time.sleep(180)
