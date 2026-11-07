"""Run on dildata using its preexisting key, with results-only read access."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--known-hosts',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    ssh=['ssh','-p','40326','-i','/home/heejoonorm/.ssh/optiq_trg_103177249208',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','UserKnownHostsFile='+str(a.known_hosts),'-o','ConnectTimeout=20']
    while True:
        try:
            result=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=*.tmp',
                '--exclude=__pycache__','--exclude=*.lock','-e',shlex.join(ssh),
                'heejoonorm@144.34.174.130:./',str(a.root/'campaign')+'/'],
                capture_output=True,text=True,timeout=1200)
            state=dict(time=time.time(),exit_code=result.returncode,stderr=result.stderr[-3000:])
        except Exception as e:state=dict(time=time.time(),error=repr(e))
        tmp=a.root/'SYNC_STATUS.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.root/'SYNC_STATUS.json')
        print(json.dumps(state),flush=True)
        if a.once:break
        time.sleep(180)
