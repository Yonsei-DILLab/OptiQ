"""dildata pull: existing results-only read-only authorization, no key transfer."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/data1/heejoonorm/OptiQ/studies/20260925_kl_six_highL')
OPS=Path('/data1/heejoonorm/OptiQ/studies/20260922_gmm40_bandit/operations')
SSH='ssh -p 60616 -i /home/heejoonorm/.ssh/optiq_trg_103177249208 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile='+str(OPS/'known_hosts')+' -o ConnectTimeout=20'
ROOT.mkdir(parents=True,exist_ok=True)
while True:
    try:
        p=subprocess.run(['rsync','-rtz','--timeout=120','--exclude=*.tmp','--exclude=__pycache__','--exclude=jax_cache','-e',SSH,
             'heejoonorm@38.49.42.46:kl_six_highL_20260925/',str(ROOT/'vast_campaign')+'/'],capture_output=True,text=True,timeout=1200)
        result=dict(time=time.time(),exit_code=p.returncode,stderr=p.stderr[-2500:])
    except Exception as e:result=dict(time=time.time(),error=repr(e))
    (ROOT/'VAST_SYNC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    allocation=ROOT/'vast_campaign/VAST_ALLOCATION.json'
    if result.get('exit_code')==0 and allocation.exists():
        cfg=json.loads(allocation.read_text());expected=sum(len(x) for x in cfg['workers'].values())
        if len(list((ROOT/'vast_campaign/runtime/confirm').glob('*/reverse_*/COMPLETE.json')))==expected:break
    time.sleep(180)
