"""Read-only status/provenance snapshot for the approved annealing and missing-baseline queue."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import json, subprocess
from pathlib import Path
r=Path('/home/heechan/optiq-experiments/antmaze-dense-anneal-baselines-s0-20260924')
old=Path('/home/heechan/optiq-experiments/antmaze-dense-off-16-current-s0-20260924')
data={}
paths=[r/n for n in ('manifest.json','status.json','registration.json','failure.json','network-recovery.json','wandb-sync-status.json')]
paths+=list((r/'jobs').glob('*.json'))
for phase in ('preflight','runs'):
 for run in (r/phase).glob('*'):
  paths += [run/n for n in ('config.json','progress.json','result.json','checkpoint-verification.json','optiq-profile-verification.json','wandb.json','failure.json')]
for p in paths:
 if p.is_file():data[str(p.relative_to(r))]=json.loads(p.read_text())
data['previous_status']=json.loads((old/'status.json').read_text())
p=old/'sac-user-cancellation.json'
if p.exists():data['sac_user_cancellation']=json.loads(p.read_text())
data['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True)
data['branch_head']=subprocess.check_output(['git','-C','/home/heechan/OptiQ-direct-gmm-trg-antmaze','rev-parse','HEAD'],text=True).strip()
data['branch_status']=subprocess.check_output(['git','-C','/home/heechan/OptiQ-direct-gmm-trg-antmaze','status','--porcelain'],text=True)
print(json.dumps(data))
'''


def collect(host):
    result=subprocess.run(['ssh',host,'python3','-'],input=REMOTE,text=True,
                          capture_output=True,check=True,timeout=45)
    data=json.loads(result.stdout)
    (ROOT/(host+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    return host,data


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for host,data in pool.map(collect,['vast-heechan-180','vast-heechan-199']):
            state=data['status.json']
            print(host,'failed=',state['failed'])
            for j in state['running']:
                key=j['id']; job=data['jobs/'+key+'.json']
                progress=data.get('runs/'+key+'/progress.json',{})
                proof=data.get('preflight/'+key+'/result.json',{})
                print(key,'GPU',j['gpu'],job['phase'],progress.get('step'),
                      'preflight_passed=',proof.get('completed',False))
    (ROOT/'collected-at.json').write_text(json.dumps({'unix_time':time.time()})+'\n')
